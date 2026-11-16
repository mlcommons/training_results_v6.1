# Running NVIDIA Small MoE GPT OSS 20B PyTorch MLPerf Benchmark

Instructions for running the **NVIDIA Small MoE GPT OSS 20B** PyTorch MLPerf
Training benchmark on NVIDIA hardware (MLPerf Training v6.1 reference stack).

## 1. Hardware requirements

- At least **100 GB** disk space.
- NVIDIA GPU with at least **80 GB** memory is strongly recommended.
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
export CONT=nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm:20261006
```

Download:

```bash
docker pull "$CONT"
```

On Slurm sites that use Enroot squashfs instead of live `docker pull`, import
the image on a **compute** node; the login node may
not have the privileges Pyxis needs.

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

GPT OSS 20B uses the **same preprocessed C4 dataset and tokenizer as Llama 3.1
8B**. If you already have the Llama 3.1 8B dataset layout, skip download. Path
names intentionally refer to `8b/`; that is expected.

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
This benchmark uses **Megatron-Core / Megatron-Bridge** with **NVIDIA Transformer Engine** (FP8
/ FP4 training on supported GPUs).

### 3.3.2 Model architecture

| Parameter | Value |
|---|---|
| Model size | 20B parameters |
| Architecture | GPT with Mixture of Experts |
| Sequence length | 8192 |
| Expert parallelism | 8 (in reference configs) |

### 3.3.3 Model checkpoint

GPT OSS 20B is trained **from scratch** for MLPerf; there is no starting
checkpoint.

## 4. Launch training

Training uses Slurm + Pyxis and MPI inside the container.

From the directory that contains `run.sub`:

```bash
export LOGDIR=</path/to/output/dir>
export DATADIR=<as/set/above>
export CONT=<as/set/above>

source config_GB200_18x4x1xtp1pp1cp2ep4_mxfp8_ub_tp2cp1_avgcoll_conn8.sh   # example: pick the config for your system

sbatch -N "${DGXNNODES}" --time="${WALLTIME}" run.sub
# Add --account and --partition per site policy.
```

Config file naming (reference):

```text
config_<SYSTEM_NAME>_<NODES>x<GPUS_PER_NODE>x<BATCH_PER_GPU>xtp<TP>pp<PP>cp<CP>[ep<EP>_]<precision>.sh
```

`TP`, `PP`, and `CP` are tensor, pipeline, and context parallel sizes. GPT OSS
reference configs also encode expert parallel (`ep`) and precision (for example
`mxfp8`).

## 5. Quality

| Item | Value |
|---|---|
| Metric | Log perplexity |
| Target | **≤ 3.34** |
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
- Layer count (`OVERWRITTEN_NUM_LAYERS`, default **24**) should be divisible
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

Container `gpt_oss_20b-arm:20261012`, NVIDIA's 2026-10-12 GPT-OSS drop. Its model definition matches the reference (SwiGLU, no linear or router bias, vanilla softmax, QK layernorm, sliding window (128, 0), `eod_mask_loss` on). The Python files in this directory (`pretrain.py`, `callback_warmup.py`, `conf/gpt_oss.yaml`, `opt_kernels/`) are the files of that drop.

### Launch

The directory is copied to the cluster as `~/gpt-oss-20b`. `site.env` sets `DATADIR` (C4 shards and tokenizer), `CONT` (the image) and `LOGDIR`. `gpt-18n.sh` sources `site.env`, the config and `gb200-opts.sh` (node-local copy of the dataset, optional warmup overrides), builds `run.sub.use2a` from NVIDIA's `run.sub` with `patch-run-sub-use2a.sh` (drops `--network=sharp`, removes the `MASTER_ADDR` container-env allowlist and resolves `MASTER_ADDR` to an IPv4 address, forwards the NCCL profiler plugin override), exports the MLPerf identity variables and submits one Slurm job that runs `NEXP` experiments back to back. `run.sub` draws a new `SEED` (`$RANDOM`) for every experiment. The submitted set was launched as

```bash
CONFIG_FILE=config_GB200_18x4x1xtp1pp1cp2ep4_mxfp8_ub_tp2cp1_avgcoll_conn8.sh NEXP=10 bash gpt-18n.sh
```

### Config

`config_GB200_18x4x1xtp1pp1cp2ep4_mxfp8_ub_tp2cp1_avgcoll_conn8.sh` sources NVIDIA's `config_GB300_18x4x1xtp1pp1cp2ep4_mxfp8.sh` and sets `NCCL_UB_DP=True`, `TENSOR_MODEL_PARALLEL=2`, `CONTEXT_PARALLEL=1`, `SEQ_PARALLEL=True`, `VAL_CHECK_INTERVAL=342`, `AVERAGE_IN_COLLECTIVE=True` and `CUDA_DEVICE_MAX_CONNECTIONS=8`. It sets its own `DGXSYSTEM`, so the logs' `config_filename` names this file. The hyperparameters (GBS 36, LR 4e-4, 256 warmup steps) are those of NVIDIA's 18x4 config.

### RCP

GBS 36 has no reference row; the checker interpolates the pruned rows at GBS 16 and 64 (RCP mean 238,933 samples, minimum 231,194). The ten runs converge in 209,304 to 233,928 samples, olympic mean 218,538, so the RCP test fails. `results/CoreWeave_GB200_18x4/gpt_oss_20b/package_checker_params` therefore carries `rcp-bypass`, as the MLCommons Training chair asked submitters to do on 2026-10-16 after NVIDIA's disclosure of the GPT-OSS reference mismatches (review repository issue #1). The handling is pending the review committee; scaling to the RCP mean would give 15.33 x 1.0933 = 16.76 min.

### Result logs

The ten result files are the complete per-run logs (framework output included, with the tenant home directory and user name redacted as the rules allow).
