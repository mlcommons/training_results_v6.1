# Lambda GPT-OSS 20B PyTorch MLPerf Training v6.1 Benchmark (GB300 NVL72, 16x4)

This directory contains the code for Lambda's closed-division GPT-OSS 20B result in
`results/gb300_n16_ngc26.09_pytorch/gpt_oss_20b`. The result was run on a Lambda 1-Click
Cluster using 16 compute trays (64 NVIDIA GB300 GPUs) of one GB300 NVL72 rack. The system
description is `systems/gb300_n16_ngc26.09_pytorch.json`.

The implementation is NVIDIA's MLPerf Training v6.1 GPT-OSS 20B code (Megatron-Bridge on
PyTorch), as shipped in NVIDIA's `gpt_oss_20b-arm:20261012` container. This directory holds the
files Lambda used to launch that container: the Dockerfile, the configs, the Slurm batch script
and a launch wrapper. The training code is not copied here. It is inside the container, and NVIDIA
publishes it in this repository at
`NVIDIA/benchmarks/gpt_oss_20b/implementations/theia-jhb_ngpu64_ngc26.09_nemo`. Sections 3 and
4 use that directory.

Lambda changed one line of `run.sub`, added a comment to the system config and added a launch
wrapper. Section 1 lists the changes.

## 1. Contents

| File | Origin |
|---|---|
| `Dockerfile` | NVIDIA, unchanged. Builds the container (section 3). |
| `config_common.sh`, `config_common_cg.sh`, `config_common_mxfp8.sh`, `config_mounts.sh` | NVIDIA, unchanged. Sourced by the system config. |
| `config_GB300_16x4x1xtp1pp1cp2ep4_mxfp8.sh` | NVIDIA's GB300 16x4 config from the same drop, with a comment added. This is the only config for this submission. |
| `run.sub` | NVIDIA's Slurm batch script, with one line changed. |
| `launch.sh` | Lambda's wrapper that sets paths and submits `run.sub`. |

Every file marked "unchanged" is byte-identical to the file of the same name in the
container's `/workspace/llm`.

Lambda's changes:

1. `run.sub`: `PYXIS_DEFAULTS` passes `--no-container-remap-root` instead of
   `--container-remap-root`. The cluster runs Slurm 23.11, whose PMIx server rejects clients
   remapped to uid 0, so MPI hung at init. This change does not touch training.
2. `config_GB300_16x4x1xtp1pp1cp2ep4_mxfp8.sh`: comment only, no setting changed. The comment
   says that HybridEP sometimes deadlocked on the first step with driver 580. After the driver
   and IMEX were upgraded to 615.71.09, runs with NVIDIA's default settings completed without
   deadlocks.
3. `launch.sh` (new): sets `CONT`, `DATADIR` and `LOGDIR` for Lambda's cluster, sources the
   config and calls `sbatch`. It also sets `NCCL_IB_HCA='^mlx5_2'` because one InfiniBand port on
   one node was down. NCCL aborts when nodes report different NIC counts, and collectives inside
   the NVL72 rack use multi-node NVLink, not InfiniBand. Other clusters do not need this setting.

## 2. Requirements

- NVIDIA GB300 NVL72: 16 nodes with 4 GPUs each, all in one NVLink domain. The config runs
  HybridEP over multi-node NVLink (`USE_MNNVL=1`).
- NVIDIA driver and IMEX 615.71.09, the version listed in the system description.
- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot).
- Docker, or another OCI image builder, on an arm64 host to build the container.
- About 100 GB of disk for the dataset. Dataset preparation does not need GPUs.

## 3. Build the container

The Dockerfile copies the training code from its build context into the image, so build it from
NVIDIA's implementation directory, which contains that code. The Dockerfile there is identical
to the one in this directory. From the repository root:

```bash
cd NVIDIA/benchmarks/gpt_oss_20b/implementations/theia-jhb_ngpu64_ngc26.09_nemo
docker build --pull -t <docker/registry>/mlperf-lambda:gpt_oss_20b-arm .
docker push <docker/registry>/mlperf-lambda:gpt_oss_20b-arm
export CONT=<docker/registry>/mlperf-lambda:gpt_oss_20b-arm
```

The training code in that directory is the code in the container. The Python files,
`requirements.txt` and `run_and_time.sh` are byte-identical to the container's. The `conf/` YAML
files there have their comments removed, but their settings are the same.

Build on an arm64 (Grace) host, or cross-build with `docker buildx build --platform linux/arm64`.
The Dockerfile:

- starts from `nvcr.io/nvidia/pytorch:26.09-py3`
- installs Megatron-Bridge `26.09-alpha.rc1` with its Megatron-LM submodule
- builds Transformer Engine `v2.19`
- builds DeepEP's `hybrid-ep` branch at commit `58397eac`
- installs the pinned packages in `requirements.txt`, including MLPerf logging `6.1.0-rc2`
- copies the build directory to `/workspace/llm`

The submitted runs used NVIDIA's prebuilt image
`nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm:20261012`, which is built from this Dockerfile and
NVIDIA's code. Lambda converted it to a squashfs file with Enroot, and `launch.sh` points `CONT`
at that file:

```bash
enroot import -o /data/containers/gpt_oss_20b-arm_20261012.sqsh \
    docker://nvcr.io#nvdlfwea/mlperftv61/gpt_oss_20b-arm:20261012
```

## 4. Prepare the dataset

GPT-OSS 20B uses the same preprocessed C4 dataset and tokenizer as Llama 3.1 8B. If you already
have that dataset, skip this step. Several paths contain `8b` on purpose.

The download scripts are in NVIDIA's implementation directory. From the repository root:

```bash
cd NVIDIA/benchmarks/gpt_oss_20b/implementations/theia-jhb_ngpu64_ngc26.09_nemo
export DATADIR=<path/to/dataset>
bash data_scripts/download.sh
```

The script downloads the dataset and tokenizer from MLCommons storage and arranges them as:

```text
$DATADIR/8b/
|-- LICENSE.txt
|-- NOTICE.txt
|-- c4-train.en_6_text_document.bin
|-- c4-train.en_6_text_document.idx
|-- c4-validation-91205-samples.en_text_document.bin
|-- c4-validation-91205-samples.en_text_document.idx
|-- llama-3-1-8b-preprocessed-c4-dataset.md5
`-- tokenizer
    |-- LICENSE
    |-- README.md
    |-- USE_POLICY.md
    |-- special_tokens_map.json
    |-- tokenizer.json
    `-- tokenizer_config.json
```

Set `DATADIR` to the parent of `8b/`. `config_mounts.sh` mounts `$DATADIR/8b` at `/preproc_data`
and `$DATADIR/8b/tokenizer` at `/workspace/llm/nemo_tokenizer`. Before training, `run.sub`
checks the mounted files against `expected-mounts.csv` in the container. Lambda stored the
dataset on the cluster's shared NFS at `/data/gpt_oss_20b/dataset`.

## 5. Launch training

Run these commands from this directory:

```bash
export CONT=<image or .sqsh file from section 3>
export DATADIR=<parent of 8b/ from section 4>
export LOGDIR=<path/to/output/dir>
export NEXP=10   # runs per job; set it before sourcing the config, which computes WALLTIME from it
source config_GB300_16x4x1xtp1pp1cp2ep4_mxfp8.sh
sbatch -N ${DGXNNODES} --gres=gpu:${DGXNGPU} --time=${WALLTIME} run.sub
# add --account / --partition as your site requires
```

Lambda launched the same command through `launch.sh`. To use it on another cluster, edit `CONT`
and `DATADIR` in the script and remove the `NCCL_IB_HCA` line, then run:

```bash
./launch.sh config_GB300_16x4x1xtp1pp1cp2ep4_mxfp8.sh 10 final   # <config> <NEXP> [tag]
```

`launch.sh` writes logs to `/data/gpt_oss_20b/results/<DGXSYSTEM>/<tag>_<timestamp>/`. You can pass
extra `sbatch` arguments in `SBATCH_EXTRA`. The ten submitted results come from one Slurm job
with `NEXP=10`.

In each job, `run.sub` does the following:

- runs an NCCL all-reduce test
- runs `NEXP` experiments back to back. Before each one it clears the page cache, then runs
  the container's `run_and_time.sh` on all 64 ranks and passes the log through the MLPerf
  compliance checker.
- writes each experiment's log to `${LOGDIR}/<DATESTAMP>_<NN>.log`

## 6. Configuration

This submission has one config, `config_GB300_16x4x1xtp1pp1cp2ep4_mxfp8.sh`. Config names follow
`config_<SYSTEM>_<NODES>x<GPUS/NODE>x<MINIBS>xtp<TP>pp<PP>cp<CP>ep<EP>_<PRECISION>.sh`. The config
sets `DGXSYSTEM` from its own file name.

| Setting | Value |
|---|---|
| Nodes x GPUs per node | 16 x 4 = 64 |
| Tensor / pipeline / context / expert parallel | 1 / 1 / 2 / 4 |
| Data parallel | 64 / (TP x PP x CP) = 32 |
| Batch per data-parallel rank (`MINIBS`) / micro-batch | 1 / 1 |
| Global batch size | 32 |
| Learning rate / warmup | 4e-4 / 256 steps |
| Evaluation interval | 384 steps = 12,288 sequences |
| Precision | MXFP8 linear layers (`config_common_mxfp8.sh`), BF16 attention |
| Other | full-iteration CUDA graphs (`config_common_cg.sh`), HybridEP over multi-node NVLink, CuTe DSL fused grouped MLP, NCCL user buffers for data parallel disabled |
| Time limit per run | 40 minutes (`WALLTIME_RUNANDTIME`) |

The global batch size is `GBS = MINIBS x DP`, where `DP = (NODES x GPUS/NODE) / (TP x PP x CP)`.
`MICRO_BATCH_SIZE` only sets gradient accumulation (`MINIBS / MICRO_BATCH_SIZE`) and does not
change the global batch size.

## 7. Model and quality

| Item | Value |
|---|---|
| Model | GPT-OSS 20B, mixture-of-experts transformer, 24 layers ([model card](https://arxiv.org/abs/2508.10925)) |
| Sequence length | 8192 |
| Checkpoint | none, trained from scratch |
| Quality metric | log perplexity |
| Quality target | 3.34 |
| Evaluation frequency | every 12,288 sequences (about 100M tokens) |
| Evaluation size | 1,024 validation sequences (about 8.4M tokens) |

The code uses [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) on
[Megatron-Core](https://github.com/NVIDIA/Megatron-LM) with
[Transformer Engine](https://github.com/NVIDIA/TransformerEngine).

## 8. Reference convergence points

All ten runs reached the target in 196,608 to 221,184 samples, with an olympic mean of 204,288.
At global batch size 32, the MLPerf logging package (training 6.1.0) gives a reference mean of
230,059 samples and a pruned minimum of 222,721. Every run converged in fewer samples than that
minimum, so the RCP test fails. For this reason,
`results/gb300_n16_ngc26.09_pytorch/gpt_oss_20b/package_checker_params` contains `rcp-bypass`.
With the bypass, the package checker accepts the result and flags it for review by the
submission committee. The reference runs at this batch size used a learning rate of 8e-4. This
submission uses NVIDIA's 4e-4.

## 9. Seeds

Megatron-Core builds the dataset sample index on one process and puts the `SEED` value in the
index file name. All ranks therefore need the same `SEED`. `run.sub` sets it if it is not
already in the environment, and it picks a new `SEED` for each experiment.
