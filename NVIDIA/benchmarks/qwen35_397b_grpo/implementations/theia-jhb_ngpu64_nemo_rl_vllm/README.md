## Running NVIDIA Qwen3.5 397B GRPO PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Qwen3.5 397B GRPO PyTorch MLPerf Benchmark on NVIDIA hardware.

The benchmark post-trains Qwen3.5-397B-A17B with Group Relative Policy Optimization (GRPO) on R2E-Gym software-engineering tasks. The policy solves each task through the OpenHands agent framework inside a sandboxed Apptainer container. Megatron-Core policy training and vLLM generation run on disaggregated node pools managed as one Ray cluster.

## 1. Hardware Requirements

- NVIDIA GB300 NVL72 (GB300) or NVIDIA VR NVL72, 4 GPUs per node. Every config splits its nodes into a policy-training pool and a generation pool; both counts are part of the config name (section 6).
- A parallel filesystem (Lustre or equivalent) mounted on every node, with at least 5 TB free. It must hold every path the launch exports: the checkpoint (`HF_CKPT_PATH`, about 800 GB) and the Megatron-format copy the first run writes (`NRL_MEGATRON_CHECKPOINT_DIR`), the task files (`NEMO_GYM_SWE_TRAIN_DATA_PATH`, `NEMO_GYM_SWE_VALIDATION_DATA_PATH`), the SIF images (`NEMO_GYM_SWE_SIF_DIR`, about 1.2 TB) and the output directory (`LOGDIR`), where each job also keeps its Ray and NeMo Gym scratch. Every node reads the checkpoint and the SIF images at start-up, so the filesystem's aggregate read bandwidth bounds the start-up time.
- `/dev/fuse` must be available on every compute node: the R2E-Gym task sandboxes run as Apptainer SIF images inside the training container.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot). Jobs are submitted with `sbatch --segment` to keep each expert-parallel group inside one NVLink domain (section 4).
- [Docker](https://www.docker.com/) on an aarch64 host for building the container.

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build from the directory that holds the `Dockerfile`:

```bash
docker build --build-arg GIT_COMMIT_ID=$(git rev-parse HEAD) -t <docker/registry>/mlperf-nvidia:qwen35_397b_grpo-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:qwen35_397b_grpo-pyt
export CONT=<docker/registry>/mlperf-nvidia:qwen35_397b_grpo-pyt
```

The image is `linux/arm64` only. `GIT_COMMIT_ID` is mandatory; it is recorded in the image labels. Optional build arguments are `MAX_JOBS` (parallelism of the source-built CUDA extensions) and `NVTE_CUDA_ARCHS` (Transformer Engine targets, default `100a;103a;107a`).

Make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

The benchmark reads three inputs hosted by MLCommons: the R2E-Gym training and validation task files, one Apptainer SIF image per task, and the Qwen3.5-397B-A17B checkpoint (section 3.3). All three are downloaded with the [MLCommons R2 Downloader](https://github.com/mlcommons/r2-downloader), which verifies every file against the package's MD5 manifest.

Set the directory for the data to be downloaded to:

```bash
export DATADIR=<path/to/dataset>
mkdir -p $DATADIR
```

#### 3.2.1 Task files

The training file holds 700 tasks and the validation file 251 held-out tasks, one JSON object per line. Row order is part of the benchmark input: the data loader does not shuffle.

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d $DATADIR/benchmark-r2e-gym-easy \
  https://training.mlcommons-storage.org/metadata/benchmark-r2e-gym-easy.uri

export NEMO_GYM_SWE_TRAIN_DATA_PATH=$DATADIR/benchmark-r2e-gym-easy/benchmark_r2e_gym_easy_train.jsonl
export NEMO_GYM_SWE_VALIDATION_DATA_PATH=$DATADIR/benchmark-r2e-gym-easy/benchmark_r2e_gym_easy_val.jsonl
```

#### 3.2.2 R2E-Gym SIF images

Every task runs in its own sandbox, an Apptainer SIF image named `<instance-id>.sif`. Download the prebuilt aarch64 bundle:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d $DATADIR/r2egym-sifs-aarch64 \
  https://training.mlcommons-storage.org/metadata/r2egym-sifs-aarch64.uri

export NEMO_GYM_SWE_SIF_DIR=$DATADIR/r2egym-sifs-aarch64/images
```

`data_scripts/build_r2e_gym_sif.sh` rebuilds the images from source instead, using Docker on an aarch64 host. It writes them to `${SIF_DIR}/r2egym/<instance-id>.sif`; the runtime accepts either layout.

At the end, the directory structure should look like:

```
$DATADIR/
|-- benchmark-r2e-gym-easy/
|   |-- ATTRIBUTION.md
|   |-- LICENSE
|   |-- LICENSES/
|   |-- README.md
|   |-- benchmark-r2e-gym-easy.md5
|   |-- benchmark_r2e_gym_easy_train.jsonl
|   `-- benchmark_r2e_gym_easy_val.jsonl
`-- r2egym-sifs-aarch64/
    |-- ATTRIBUTION.md
    |-- LICENSES/
    |-- r2egym-sifs-aarch64.md5
    `-- images/
        |-- <instance-id>.sif
        `-- ...
```

`run.sub` checks the mounted inputs against `expected-mounts.csv` before every run (`VERIFY_MOUNTS=1`, the default) and writes the result to `<LOGDIR>/*_mountcheck.log`.

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

The policy is [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), released by the Qwen team under the Apache-2.0 license and described in [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5). The tasks come from [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym), described in [R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents](https://arxiv.org/abs/2504.07164). The policy acts on them through the [OpenHands](https://github.com/All-Hands-AI/OpenHands) agent framework.

Training is implemented with [NeMo RL](https://github.com/NVIDIA-NeMo/RL) and [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) on top of [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) via [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Generation is implemented with [vLLM](https://github.com/vllm-project/vllm).

#### 3.3.2 Model Architecture

The model follows the [Qwen3.5-397B-A17B model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B). Only the language model is trained; the vision encoder is not used.

| Parameter | Value |
|-----------|-------|
| Model Size | 397B total parameters, 17B active |
| Architecture | Hybrid Gated DeltaNet / gated-attention decoder with Mixture of Experts |
| Number of Layers | 60: 15 blocks of three Gated DeltaNet layers followed by one gated full-attention layer, each with an MoE MLP |
| Hidden size | 4096 |
| Gated DeltaNet heads | 64 value / 16 key, head dimension 128 |
| Full-attention heads | 32 query / 2 key-value, head dimension 256 |
| Rotary dimension | 64 |
| Routed experts | 512, 10 active per token, plus 1 shared expert |
| Expert intermediate size | 1024 |
| Vocabulary size | 248,320 |
| Native context length | 262,144 tokens |
| Benchmark sequence length | 65,536 tokens (prompt and response) |
| Training method | GRPO; routed-expert GEMMs in MXFP8 (forward and backward) with BF16 Adam state |

#### 3.3.3 Model checkpoint

MLCommons hosts the initial policy checkpoint at [MLCommons storage](https://training.mlcommons-storage.org/index.html#qwen3-5-397b-benchmark). Download it next to the dataset and choose a writable directory for the Megatron-format copy:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d $DATADIR/qwen35-397b-a17b \
  https://training.mlcommons-storage.org/metadata/qwen35-397b-a17b.uri

export HF_CKPT_PATH=$DATADIR/qwen35-397b-a17b
export NRL_MEGATRON_CHECKPOINT_DIR=<path/to/megatron-checkpoint-cache>
mkdir -p $NRL_MEGATRON_CHECKPOINT_DIR
```

`HF_CKPT_PATH` must name the directory that holds `config.json` and the 94 weight shards; a HuggingFace hub cache `snapshots/<sha>` directory also works. Use `data_scripts/convert_ckpt.sh` to populate `NRL_MEGATRON_CHECKPOINT_DIR` before the first benchmark run; benchmark runs mount the cache read-only and reuse the converted copy.

At the end, the directory structure should look like:

```
$DATADIR/qwen35-397b-a17b/
|-- LICENSE
|-- README.md
|-- chat_template.jinja
|-- config.json
|-- generation_config.json
|-- merges.txt
|-- model.safetensors-00001-of-00094.safetensors
|-- ...
|-- model.safetensors-00094-of-00094.safetensors
|-- model.safetensors.index.json
|-- preprocessor_config.json
|-- qwen35-397b-a17b.md5
|-- tokenizer.json
|-- tokenizer_config.json
|-- video_preprocessor_config.json
`-- vocab.json
```

## 4. Launch training

For training, we use Slurm with the Pyxis extension. A run is a Ray cluster: `run.sub` starts one Ray node per Slurm node, waits until every GPU has joined, and runs the GRPO driver once on the head node. Megatron-Core policy training and vLLM generation use disaggregated node pools (`DGXNNODES = TRAIN_NODES + GEN_NODES`), and NeMo Gym schedules the agent rollouts against the generation pool.

Navigate to the directory where `run.sub` is stored. All input paths must be visible from every node.

The launch command structure:

```bash
export HF_CKPT_PATH=<as/set/above>
export NRL_MEGATRON_CHECKPOINT_DIR=<as/set/above>
export NEMO_GYM_SWE_TRAIN_DATA_PATH=<as/set/above>
export NEMO_GYM_SWE_VALIDATION_DATA_PATH=<as/set/above>
export NEMO_GYM_SWE_SIF_DIR=<as/set/above>
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export CONT=<as/set/above>
source config_GB300_16x4_t8g8_tp4pp2ep16gtp4_mxfp8.sh  # select config and source it
sbatch --no-requeue --segment=${SEGMENT} -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account, --partition and --gres=gpu:${DGXNGPU} here
```

`--no-requeue` is required: `run.sub` refuses to run in a job that Slurm may restart. `--segment=${SEGMENT}` keeps each expert-parallel group of `SEGMENT` nodes inside one NVLink domain; Slurm installations without this option need the site-specific option with the same effect. Asynchronous GRPO leaves the training pool idle while the rollout buffer fills, so on sites that terminate jobs with idle GPUs the job must be exempted through the site's own mechanism.

`run.sub` writes the MLPerf log to `<LOGDIR>/<datestamp>_<n>.log` and runs the compliance checker on it. When the job ends it removes the job's NeMo Gym and Ray scratch trees under `LOGDIR` (`nemo_logs_<job-id>/nemo_gym` and `ray-<job-id>.<restart>`); export `RETAIN_GRPO_SCRATCH=1` before `sbatch` to keep them for debugging. `grpo_teardown.sh`, `cleanup_grpo_scratch.sh` and `utility/slurm_helpers.sh` must stay next to `run.sub`.

## 5. Quality

### Quality metric
Grouped pass@4 on the validation tasks: a task passes when at least one of its four generated trajectories receives a positive reward from the task's test harness. The MLPerf log reports it as `eval_accuracy`.

### Quality target
0.69

### Evaluation frequency
After training, validation restores steps `VAL_START_AT` and `VAL_START_AT + 1`, stopping after the first checkpoint that reaches the target. These values are set by the batch-size schedule the config sources (section 6).

### Evaluation thoroughness
All 251 validation tasks with four generations each, i.e. 1,004 trajectories per evaluation, sampled at temperature 0.1 and top-p 0.95.

## 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM>_<NODES>x<GPUS/NODE>_t<TRAIN>g<GEN>_tp<TP>pp<PP>ep<EP>gtp<GTP>[_gbs<GBS>][_mxfp8].sh`, e.g. `config_GB300_16x4_t8g8_tp4pp2ep16gtp4_mxfp8.sh`, where:

```
SYSTEM     = target system (GB300 = NVIDIA GB300 NVL72, VR = NVIDIA VR NVL72)
NODES      = total number of nodes (DGXNNODES)
GPUS/NODE  = GPUs per node (DGXNGPU)
TRAIN      = policy-training nodes (TRAIN_NODES)
GEN        = vLLM generation nodes (GEN_NODES); NODES = TRAIN + GEN
TP, PP, EP = policy tensor, pipeline and expert parallelism (TP, PP, EP)
GTP        = generation tensor parallelism (GTP)
GBS        = optional global batch size override
mxfp8      = routed-expert MXFP8 forward and backward compute (recipe grpo_qwen35_397b_swe_openhands_async_mxfp8.yaml)
```

The global batch size is `NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT`. Each config sources one batch-size schedule, `config_GBS<GBS>.sh`, which sets the hyperparameters that depend on it: prompts and generations per step, learning rate (`LEARNING_RATE`), gradient clip (`MAX_GRAD_NORM`), first evaluation step (`VAL_START_AT`) and maximum number of steps (`MAX_STEPS`). The schedule a config uses is the one it sources. Profiles also set `SEGMENT` for Slurm allocation and `NEMORL_TRAIN_SEGMENT_SIZE` for NeMo-RL training placement. Both default to the profile's existing segment size and can be overridden independently before sourcing the config. Keep training expert-parallel groups within an NVLink domain when changing either setting. Configs do not export `SBATCH_SEGMENT_SIZE`; direct submissions must pass `--segment=${SEGMENT}` as shown above.

Configs of 64 nodes and more also source `config_gym_shards.sh`, which runs the NeMo Gym rollout server as `NEMO_GYM_REPLICAS` (default 4) actors.

### Seeds
`run.sub` gives each experiment its own seed, drawn from `$RANDOM`, and passes it to the driver as the GRPO seed; the value is recorded in the MLPerf log.
If the `SEED` environment variable is set prior to launching the job, the first experiment uses that value and later experiments still draw their own.
