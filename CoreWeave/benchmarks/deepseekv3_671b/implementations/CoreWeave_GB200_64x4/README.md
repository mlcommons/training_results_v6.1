# Running NVIDIA Large MoE DeepSeek v3 671B PyTorch MLPerf Benchmark

Instructions for running the **NVIDIA DeepSeek v3 671B** PyTorch MLPerf Training
benchmark on NVIDIA hardware (MLPerf Training v6.1 reference stack).

## 1. Hardware requirements

- At least **1.5 TB** disk space.
- NVIDIA GPU with at least **190 GB** memory is strongly recommended (for example
  GB200).
- GPUs are **required** for checkpoint download processing and conversion.

## 2. Software requirements

- **Slurm** with **Pyxis** and **Enroot**
- **Docker** (for pulling the NGC container and optional local extract)

## 3. Set up

### 3.1.1 Pull the container

Log in to the NGC registry:

```bash
docker login nvcr.io
```

Set the container image (ARM example for GB200 / Grace Hopper):

```bash
export CONT=nvcr.io/nvdlfwea/mlperftv61/deepseekv3_671b-arm:20261006
```

Download:

```bash
docker pull "$CONT"
```

On Slurm sites that use Enroot squashfs instead of live `docker pull`, import
the image on a **compute** node; the login node may not have the privileges
Pyxis needs.

### 3.1.2 Extract configs and batch script

Training launch is two steps:

1. **Source** a `config_*.sh` file.
2. **Submit** `run.sub` to Slurm.

Extract launch assets from the container into your work directory
(`</path/to/your/workdir>`):

**Docker**

```bash
docker run -it --rm --network=host --ipc=host \
  --volume </path/to/your/workdir>:/mounted \
  "$CONT"
```

**Slurm (interactive)**

```bash
srun --nodes=1 -t 40:00 --pty \
  --container-image="${CONT}" \
  --container-mounts=</path/to/your/workdir>:/mounted \
  -p <partition> -A <account> \
  /bin/bash
```

Inside the container, copy files to the mount:

```bash
cp /workspace/llm/config_*.sh /mounted/ && chmod 664 /mounted/config_*.sh
cp /workspace/llm/*.sub /mounted/ && chmod 775 /mounted/run.sub
cp -r /workspace/llm/data_scripts /mounted && chmod 775 /mounted/data_scripts/*
exit
```

You should now have `config_*.sh`, `run.sub`, and `data_scripts/` in the
workdir.

## 3.2 Prepare dataset

DeepSeek v3 671B uses the **same preprocessed C4 dataset and tokenizer as Llama
3.1 8B**. If you already have that layout, skip download. Path names
intentionally refer to `8b/`; that is expected.

```bash
export DATADIR=<path/to/dataset>
bash data_scripts/download.sh
```

Expected layout under `8b/`:

```text
8b/
├── LICENSE.txt
├── NOTICE.txt
├── c4-train.en_6_text_document.bin
├── c4-train.en_6_text_document.idx
├── c4-validation-91205-samples.en_text_document.bin
├── c4-validation-91205-samples.en_text_document.idx
├── llama-3-1-8b-preprocessed-c4-dataset.md5
└── tokenizer/
    ├── LICENSE
    ├── README.md
    ├── USE_POLICY.md
    ├── llama-3-1-8b-tokenizer.md5
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

Set `DATADIR` to the **parent** of `8b/` (launch configs append `/8b` via mounts).
Training fails at data load if `8b/tokenizer/tokenizer.json` is missing.

## 3.3 Model and checkpoint preparation

### 3.3.1 Publication / attribution

Megatron is developed by the Applied Deep Learning Research team at NVIDIA.
This benchmark uses **Megatron-Core / Megatron-Bridge** with **NVIDIA
Transformer Engine** (FP8 / FP4 training on supported GPUs).

### 3.3.2 Model architecture

| Parameter | Value |
|---|---|
| Model size | 671B parameters |
| Architecture | Mixture of Experts (DeepSeek v3) |
| Sequence length | 4096 |
| Layers | 61 |

### 3.3.3 Checkpoint download

MLCommons hosts the starting checkpoint for members. Download with the R2
downloader (package is ~1.4 TB):

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "<checkpoint_dest>" \
  https://training.mlcommons-storage.org/metadata/deepseekv3_checkpoint_bf16.uri
```

### 3.3.4 Starting checkpoint

The distributed checkpoint is in **Hugging Face** format. It was produced by
loading the original DeepSeek v3 671B weights into Megatron-Bridge, training
**50** iterations with sequence auxiliary load-balancing loss weight **1e-2**,
then converting back to Hugging Face format. That warm-up adapts MoE routers
after switching to the **Llama 3.1 8B tokenizer** (token distribution changes
can imbalance routed experts).

### 3.3.5 Checkpoint preprocessing

Training starts from the converted Megatron checkpoint of the starting
checkpoint described in 3.3.4. Set `LOAD_CHECKPOINTS_DIR` to its directory
before `sbatch`.

## 4. Launch training

Training uses Slurm + Pyxis and MPI inside the container.

From the directory that contains `run.sub`:

```bash
export LOGDIR=</path/to/output/dir>
export DATADIR=<as/set/above>
export CONT=<as/set/above>

source config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh

sbatch -N "${DGXNNODES}" --time="${WALLTIME}" run.sub
# Add --account and --partition per site policy.
```

Config file naming (reference):

```text
config_<SYSTEM_NAME>_<NODES>x<GPUS_PER_NODE>x<BATCH_PER_GPU>xtp<TP>pp<PP>cp<CP>.sh
```

`TP`, `PP`, and `CP` are tensor, pipeline, and context parallel sizes.

## 5. Quality

| Item | Value |
|---|---|
| Metric | Log perplexity |
| Target | **≤ 3.6** |
| Evaluation frequency | **Every training step** (one step per GBS update) |
| Evaluation thoroughness | **1,024** validation sequences (~8.4M tokens) |

MLPerf reference convergence points (RCP) are keyed by **global batch size (GBS)**.
Pick `GBS`, learning rate, and warmup to match the published row for your scale.

## 6. Additional notes

### 6.1 Parallelism and batch size

```text
MP = TP × PP × CP
DP = WS // MP = (NNODES × GPUS_PER_NODE) / (TP × PP × CP)
miniBS = GBS // DP
```

| Symbol | Meaning |
|---|---|
| MP | Model parallelism |
| TP | Tensor parallelism |
| PP | Pipeline parallelism |
| CP | Context parallelism |
| DP | Data parallelism |
| WS | World size (nodes × GPUs per node) |
| GBS | Global batch size |

Changing `MICRO_BATCH_SIZE` does **not** change GBS or the relations above; it
sets gradient accumulation: `GA = miniBS // microBS`.

**Knob guidance**

- `GBS` should be divisible by `DP × VP`, where `VP` is virtual pipelining
  (`INTERLEAVED_PIPELINE`).
- Layer count (`OVERWRITTEN_NUM_LAYERS`, default **61**) should be divisible
  by `PP × VP`.
- If you reduce layers, set `LOAD_CHECKPOINT=""` so you do not load a checkpoint
  with more layers than the run config.

### 6.2 Seeds

NeMo builds dataset index shuffles on one rank and encodes `SEED` in index file
names. **All ranks must share the same `SEED`** or data load fails. Export
`SEED` before launch, or rely on the value set in `run.sub`.

## CoreWeave submission notes

The sections above are NVIDIA's generic instructions for this benchmark. This section records how CoreWeave ran the result submitted from this directory.

### Image and code

Container `deepseekv3_671b-arm:20261006`, NVIDIA's 2026-10-06 DeepSeek-V3 drop. `pretrain.py` in this directory is mounted over the image copy; `numabind.sh` binds each rank to the NUMA node of its GPU.

### Launch

The directory is copied to the cluster as `~/dsv3-671b`. `dsv3-64n.sh` sources `site.env` and `config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh`, mounts `pretrain.py` and `numabind.sh` into the container (`EXTRA_MOUNTS`), exports the MLPerf identity variables, requires `NODELIST` to name exactly 64 nodes, and submits `run.sub.use2a` (NVIDIA's `run.sub` with the IPv4 `MASTER_ADDR` resolution and the NCCL profiler plugin override) as one Slurm job that runs `NEXP` experiments back to back; `run.sub.use2a` sets `SEED` to `SEED_BASE - 1 + <experiment index>`. The submitted set was launched as

```bash
NODELIST=<64-node list> NEXP=3 bash dsv3-64n.sh
```

### Result logs

The three result files are the complete per-run logs (framework output included, with the tenant home directory and user name redacted as the rules allow).
