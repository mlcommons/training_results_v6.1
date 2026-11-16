# Running NVIDIA Large Language Model Llama 3.1 8B PyTorch MLPerf Benchmark

Instructions for running the **NVIDIA Llama 3.1 8B** PyTorch MLPerf Training
benchmark on NVIDIA hardware (MLPerf Training v6.1 reference stack).

## 1. Hardware requirements

- At least **100 GB** disk space.
- NVIDIA VR200 system with 4 GPUs per node.
- GPUs are **not** required for dataset preparation.

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
export CONT=nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm:20261012
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

```bash
export DATADIR=<path/to/dataset>
bash data_scripts/download_8b.sh
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
This benchmark uses **Megatron-Core / Megatron-Bridge** with **NVIDIA Transformer Engine** (FP4
/ FP8 training on supported GPUs).

### 3.3.2 Model architecture

| Parameter | Value |
|---|---|
| Model size | 8B parameters |
| Architecture | Llama 3.1 decoder-only transformer |
| Layers | 32 |
| Sequence length | 8192 (training benchmark) |
| Vocabulary | 128,000 tokens |

### 3.3.3 Model checkpoint

Llama 3.1 8B is trained **from scratch** for MLPerf; there is no starting
checkpoint.

## 4. Launch training

Training uses Slurm + Pyxis and MPI inside the container.

From the directory that contains `run.sub`:

```bash
export LOGDIR=</path/to/output/dir>
export DATADIR=<as/set/above>
export CONT=<as/set/above>

source config_VR200_18x4x1xtp1pp1cp1_8b.sh   # example: pick the config for your system

sbatch -N "${DGXNNODES}" --time="${WALLTIME}" run.sub
# Add --account and --partition per site policy.
```

Config file naming (reference):

```text
config_<SYSTEM_NAME>_<NODES>x<GPUS_PER_NODE>x<BATCH_PER_GPU>xtp<TP>pp<PP>cp<CP>_8b_<precision>.sh
```

`TP`, `PP`, and `CP` are tensor, pipeline, and context parallel sizes. Reference
configs encode precision (for example `fp4`).

## 5. Quality

| Item | Value |
|---|---|
| Metric | Log perplexity |
| Target | **≤ 3.3** |
| Evaluation frequency | Every **12,288** sequences (~100M tokens) |
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
- Layer count (`OVERWRITTEN_NUM_LAYERS`, default **32**) should be divisible
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

Container `llama31_8b-arm:20261012`, NVIDIA's 2026-10-12 Llama 3.1 8B drop; the Python files in this directory are that drop's, plus `patches/pretrain_avgcoll.py` (see below).

### Launch

The directory is copied to the cluster as the 20261012 code drop. `submit-18n.sh` checks the image, the dataset, the 18-node allocation and the site network settings, pins `NCCL_IB_HCA` in `site.env` to the explicit port list, and calls `llama-18n.sh` with `NEXP=10`, `SLOW_DATADIR_SRC=<DATADIR>` (node-local copy of the dataset) and `CONFIG_FILE=config_VR200_18x4x1xtp1pp1cp1_8b_ub_avgcoll.sh`. `llama-18n.sh` sources `site.env` and the config and submits `run.sub.vr` (NVIDIA's `run.sub` with the IPv4 `MASTER_ADDR` resolution and the NCCL profiler plugin override) as one Slurm job that runs `NEXP` experiments back to back. `run.sub` draws a new `SEED` (`$RANDOM`) for every experiment. The submitted set was launched as

```bash
NODES_18N=<18-node list> NEXP18=10 bash submit-18n.sh
```

### Config and the logged `config_filename`

The runs used `config_VR200_18x4x1xtp1pp1cp1_8b_ub_avgcoll.sh`, a chain of three files:

| File | Adds |
|---|---|
| `config_VR200_18x4x1xtp1pp1cp1_8b.sh` | NVIDIA's shipped 18x4 config (NVFP4 GEMM, FP8 attention, CUDA graphs, GBS 72, LR 8e-4, 64 warmup steps) |
| `config_VR200_18x4x1xtp1pp1cp1_8b_ub.sh` | `NCCL_UB_DP=True` |
| `config_VR200_18x4x1xtp1pp1cp1_8b_ub_avgcoll.sh` | `AVERAGE_IN_COLLECTIVE=True`, and `EXTRA_MOUNTS` placing `patches/pretrain_avgcoll.py` over `/workspace/llm/pretrain.py` |

`patches/pretrain_avgcoll.py` is the image's `pretrain.py` plus the lines that read `AVERAGE_IN_COLLECTIVE` into `DistributedDataParallelConfig.average_in_collective` (compare it with `pretrain.py` in this directory). `pretrain.py` logs `config_filename` as `config_${DGXSYSTEM}.sh`, and only the base file sets `DGXSYSTEM`, so the logs name `config_VR200_18x4x1xtp1pp1cp1_8b.sh` while the run used the derived file; the logged parallelism (TP 1, PP 1, CP 1) is the same in both.

### Result logs

The ten result files are the complete per-run logs (framework output included, with the tenant home directory and user name redacted as the rules allow).
