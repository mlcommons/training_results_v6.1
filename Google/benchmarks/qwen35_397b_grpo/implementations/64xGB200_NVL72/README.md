# Qwen3.5-397B GRPO on Google Cloud A4X (64 × 4 GB200)

This file contains the instructions for running the NVIDIA Qwen3.5 397B GRPO PyTorch MLPerf Benchmark on Google A4X with TensorRT-LLM as the generation engine.

The benchmark post-trains Qwen3.5-397B-A17B with Group Relative Policy Optimization (GRPO) on R2E-Gym software-engineering tasks. The policy solves each task through the OpenHands agent framework inside a sandboxed Apptainer container. Megatron-Core policy training and TensorRT-LLM generation run on disaggregated node pools managed as one Ray cluster. 

## 1. Hardware Requirements

- 64 Google Cloud A4X nodes, with 4 NVIDIA GB200 GPUs per node: 256 GPUs total.
- Four NVLink domains, with 16 allocated nodes per domain.
- Shared Lustre mounted on the controller and every compute node. Allow space for the approximately 800 GB initial model, its converted Megatron checkpoint, approximately 1.2 TB of SIF images, saved training checkpoints, and logs. Budget at least 5 TB as a planning estimate; retained runs require additional capacity.
- Slurm with Pyxis and Enroot, GPU allocation, and the site's `--segment=16` support.
- Python 3, `unsquashfs`, `realpath`, and `sbatch` on the launch host.
- `/dev/fuse` on compute nodes for the task sandboxes.
- Permission to run the configured cache-clear command, `sudo -n /sbin/sysctl vm.drop_caches=3`, on compute nodes.

Docker is needed only when rebuilding the container. Existing squashfs images can be used directly.

## 2. Configuration

| Setting | Current GB200 launcher |
|---|---|
| Global batch size | 512: 32 prompts × 16 generations |
| Policy training | 32 nodes / 128 GPUs |
| Generation | 32 nodes / 128 GPUs; 32 TP4/EP4 engines |
| Training parallelism | TP4, PP2, EP32, expert TP1, CP1 |
| Placement | Slurm segment size 16; NeMo RL training segment size 8 |
| Policy parameters / master parameters | BF16 / FP32 |
| Training routed-expert GEMMs | MXFP8 forward and backward |
| Other Transformer Engine modules | BF16 |
| Adam first and second moments | BF16 |
| Sampler routed experts | MXFP8, CuTe DSL backend |
| Sampler KV and recurrent-state caches | BF16 |
| Sampler max batch / memory utilization | 256 / 0.7 |
| Maximum total sequence length | 65,536 tokens |
| Training microbatch / logprob batch | 1 / 1 |
| Learning rate | 1.4142135624e-6 |
| Gradient clipping | 0.08838834765 |
| Activation checkpointing | Enabled, full recomputation |
| Fine-grained activation offload | Disabled |
| Optimizer CPU offload | Disabled |
| Chunked optimizer-state offload | Disabled |
| Maximum trajectory age | 4 steps |
| NeMo Gym replicas / proxy workers | 4 replicas; 4 proxy workers per replica |
| Training / evaluation concurrency | 1536 / 1024 |
| Thinking mode | `enable_thinking=false` |
| Training sampling | Temperature 1.0, top-p 1.0 |
| Validation sampling | Temperature 0.1, top-p 0.95 |
| Evaluation | Deferred (`OFFLINE_EVAL=1`), described below |
| W&B / compliance checker | Disabled / enabled |

The initial Hugging Face checkpoint is BF16. MXFP8 is selected by the training and generation recipes; an FP8 Hugging Face checkpoint is not required.


## 3. Prepare shared inputs

### 3.1 Site configuration

Create a writable project directory on Lustre. On a newly provisioned filesystem, an administrator may need to create and assign ownership of the user directory first:

```bash
sudo mkdir -p "/lustre/$USER"
sudo chown "$USER:$(id -gn)" "/lustre/$USER"

mkdir -p "/lustre/$USER/mlperf-rl"/{data,images,results}
```

Create the site file that the launcher reads:

```bash
cat > "/lustre/$USER/mlperf-rl/site.sh" <<'SITE'
export RL_ROOT="/lustre/$USER/mlperf-rl"
export DATADIR="$RL_ROOT/data"
export HF_CKPT_PATH="$DATADIR/qwen35-397b-a17b"
export NEMO_GYM_SWE_TRAIN_DATA_PATH="$DATADIR/benchmark-r2e-gym-easy/benchmark_r2e_gym_easy_train.jsonl"
export NEMO_GYM_SWE_VALIDATION_DATA_PATH="$DATADIR/benchmark-r2e-gym-easy/benchmark_r2e_gym_easy_val.jsonl"
export NEMO_GYM_SWE_SIF_DIR="$DATADIR/r2egym-sifs-aarch64/images"
SITE

source "/lustre/$USER/mlperf-rl/site.sh"
```

Use absolute paths visible on every node, without spaces, commas, or colons.

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

Training is implemented with [NeMo RL](https://github.com/NVIDIA-NeMo/RL) and [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) on top of [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) via [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Generation is implemented with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

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

`HF_CKPT_PATH` must name the directory that holds `config.json` and the 94 weight shards; a HuggingFace hub cache `snapshots/<sha>` directory also works. The first run converts the checkpoint to Megatron format into `NRL_MEGATRON_CHECKPOINT_DIR`, outside the timed region; later runs reuse the converted copy.

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

## 4. Launch on GB200

From the repository checkout:

```bash
cd "$HOME/MLPerf-v6.1-Google-GPU/benchmarks/qwen35_397b_grpo/implementations/64xGB200_NVL72"

bash launch_gb200_train64.sh \
  "$HOME/images/qwen35_397b_grpo-arm+20261013.sqsh"
```

```text
partition=a4x
nodes=64
segment=16
tasks-per-node=1
gpus-per-node=4 (--gres=gpu:4)
exclusive allocation, all host memory, no requeue
allocation time limit=10 hours
```

### What the launcher does

1. Reads `site.sh`, creates a fresh run directory, and stages the supplied image if necessary.
2. Extracts the benchmark launch tree from that image.
3. Draws a seed and writes the complete run configuration.
4. Creates a per-run `run.gb200.sub` with two Slurm adaptations: optional account handling and bounded topology ranks 1–64. Physical NVLink grouping still comes from the topology probe's ClusterUUID.
5. Submits the 64-node allocation, starts the Ray cluster, runs training and deferred evaluation, and invokes compliance checking.

The benchmark code executed by this entry point comes from the **supplied image**. Editing the checked-in recipe or `run.gb200.sub` alone does not change that code; source changes must be included in the image used for the run.

## 5. Quality

### Quality metric
Grouped pass@4 on the validation tasks: a task passes when at least one of its four generated trajectories receives a positive reward from the task's test harness. The MLPerf log reports it as `eval_accuracy`.

### Quality target
0.69

### Evaluation frequency
Every training step from step `VAL_START_AT` on. A run stops when it reaches the target or after `MAX_STEPS` steps. Both are set by the batch-size schedule the config sources (section 6).

### Evaluation thoroughness
All 251 validation tasks with four generations each, i.e. 1,004 trajectories per evaluation, sampled at temperature 0.1 and top-p 0.95.

## 6. Additional notes

### Config naming convention

Configuration files use two formats, depending on whether generation combines prefill and decode in each engine or uses separate engines:

- Aggregated generation: `config_<SYSTEM>_<NODES>x<GPUS/NODE>_t<TRAIN>g<GEN>_tp<TP>pp<PP>ep<EP>gtp<GTP>[_mxfp8].sh`.
- Disaggregated prefill/decode: `config_<SYSTEM>_<NODES>x<GPUS/NODE>_t<TRAIN>g<GEN>_tp<TP>pp<PP>ep<EP>_<P_ENGINES>p<PREFILL>d<DECODE>[_mxfp8].sh`, e.g. `config_VR200_16x4_t8g8_tp4pp2ep16_2p8d16_mxfp8.sh`.

The fields mean:

```
SYSTEM     = target system (GB300 = NVIDIA GB300 NVL72, VR200 = NVIDIA Vera Rubin NVL72)
NODES      = total number of nodes (DGXNNODES)
GPUS/NODE  = GPUs per node (DGXNGPU)
TRAIN      = policy-training nodes (TRAIN_NODES)
GEN        = TensorRT-LLM generation nodes (GEN_NODES); NODES = TRAIN + GEN
TP, PP, EP = policy tensor, pipeline and expert parallelism (TP, PP, EP)
GTP        = aggregated generation tensor parallelism (TRTLLM_TP)
P_ENGINES  = number of prefill engines (CTX_ENGINES)
PREFILL    = width of each prefill engine (CTX_TP, CTX_MEP)
DECODE     = width of the decode engine (GEN_TP, GEN_MEP)
mxfp8      = routed-expert MXFP8 forward and backward compute
```

The global batch size is `NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT`. Each config sources one batch-size schedule, `config_GBS<GBS>.sh`, which sets the hyperparameters that depend on it: prompts and generations per step, learning rate (`LEARNING_RATE`), gradient clip (`MAX_GRAD_NORM`), first evaluation step (`VAL_START_AT`) and maximum number of steps (`MAX_STEPS`). The schedule a config uses is the one it sources. Profiles also set `SEGMENT` for Slurm allocation and `NEMORL_TRAIN_SEGMENT_SIZE` for NeMo-RL training placement. Both default to the profile's existing segment size and can be overridden independently before sourcing the config. Keep training expert-parallel groups within an NVLink domain when changing either setting. Configs do not export `SBATCH_SEGMENT_SIZE`; direct submissions must pass `--segment=${SEGMENT}` as shown above.

### Seeds
`run.sub` gives each experiment its own seed, drawn from `$RANDOM`, and passes it to the driver as the GRPO seed; the value is recorded in the MLPerf log.
If the `SEED` environment variable is set prior to launching the job, the first experiment uses that value and later experiments still draw their own.
