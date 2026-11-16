# Qwen3.5-397B GRPO on Google Cloud A4X Max (14 × 4 GB300)

This file contains the instructions for running the NVIDIA Qwen3.5 397B GRPO PyTorch MLPerf Benchmark on Google Cloud A4X Max with TensorRT-LLM as the generation engine.

The benchmark post-trains Qwen3.5-397B-A17B with Group Relative Policy Optimization (GRPO) on R2E-Gym software-engineering tasks. The policy solves each task through the OpenHands agent framework inside a sandboxed Apptainer container. Megatron-Core policy training and disaggregated TensorRT-LLM generation (separate prefill and decode engines) run on separate node pools managed as one Ray cluster.

## 1. Hardware Requirements

- 14 Google Cloud A4X Max nodes (`a4x-maxgpu-4g-metal`), with 4 NVIDIA GB300 GPUs per node: 56 GPUs total.
- All 14 nodes in one NVLink (NVL72) domain, allocated as a single Slurm segment of 14 nodes.
- Shared Lustre mounted on the controller and every compute node. Allow space for the approximately 800 GB initial model, its converted Megatron checkpoint, approximately 1.2 TB of SIF images, saved training checkpoints, and logs. Budget at least 5 TB as a planning estimate; retained runs require additional capacity.
- Slurm with Pyxis and Enroot, GPU allocation, and the site's `--segment=14` support.
- Python 3, `unsquashfs`, `realpath`, and `sbatch` on the launch host.
- `/dev/fuse` on compute nodes for the task sandboxes.
- Permission to run the configured cache-clear command, `sudo -n /sbin/sysctl vm.drop_caches=3`, on compute nodes. `setup_gb300_nodes.sh` grants exactly this permission to the submitting user (section 4).

Docker is needed only when rebuilding the container. Existing squashfs images can be used directly.

## 2. Configuration

The launcher uses NVIDIA's 14-node GB300 profile, `config_GB300_14x4_t8g6_tp4pp2ep16_2p4d16_mxfp8.sh`, from the supplied image. It does not change the profile's model, topology, precision, or offload settings.

| Setting | Current GB300 launcher |
|---|---|
| Global batch size | 512: 32 prompts × 16 generations |
| Policy training | 8 nodes / 32 GPUs |
| Generation | 6 nodes / 24 GPUs; disaggregated: 2 TP4/EP4 prefill (context) engines and 1 TP16/EP16 decode engine, attention DP, NIXL KV-cache transfer |
| Training parallelism | TP4, PP2 (31 layers in the first pipeline stage), EP16, expert TP1, CP1 |
| Placement | Slurm segment size 14; NeMo RL training segment size 8 |
| Policy parameters / master parameters | BF16 / FP32 |
| Training routed-expert GEMMs | MXFP8 forward and backward |
| Other Transformer Engine modules | BF16 |
| Adam first and second moments | BF16 |
| MoE token dispatcher | `flex` with the `hybridep` backend; shared-expert overlap disabled |
| Sampler routed experts | MXFP8, CuTe DSL backend |
| Sampler KV and recurrent-state caches | BF16 |
| Sampler max batch / memory utilization | Prefill engines 64 / 0.8; decode engine 256 / 0.8 |
| Maximum total sequence length | 65,536 tokens |
| Training microbatch / logprob batch | 1 / 1 |
| Learning rate | 1.4142135624e-6 |
| Gradient clipping | 0.08838834765 |
| Activation checkpointing | Enabled, selective recomputation of `layernorm`, `attention`, `gdn`, `moe_act`, `shared_experts` |
| Fine-grained activation offload | Disabled |
| Optimizer CPU offload | Disabled |
| Chunked optimizer-state offload | Enabled (offload fraction 1.0, 256 MB chunks) |
| Maximum trajectory age | 4 steps |
| NeMo Gym replicas / proxy workers | 1 replica; 16 proxy workers per replica |
| Training / evaluation concurrency | 512 / 256 |
| Validation batching | Waves of at most 126 prompts (`VAL_BATCH_SIZE=126`) |
| Thinking mode | `enable_thinking=false` |
| Training sampling | Temperature 1.0, top-p 1.0 |
| Validation sampling | Temperature 0.1, top-p 0.95 |
| Evaluation | Deferred (`OFFLINE_EVAL=1`), described below |
| W&B / compliance checker | Disabled / enabled |

The initial Hugging Face checkpoint is BF16. MXFP8 is selected by the training and generation recipes; an FP8 Hugging Face checkpoint is not required.

Besides site paths and the Slurm adaptations described in section 4, the launcher's `config.sh` adds only site settings: GCP network selection (`UCX_TLS=all UCX_NET_DEVICES=all UCX_TCP_AF_PRIO=inet`), the actual Slurm CPU allocation instead of the profile's 144-CPU preset, MLPerf metadata (`Google`, `GB300`), and `NCCL_RAS_ENABLE=0`. The last setting works around an NCCL 2.30 RAS use-after-free crash (`ras/peers.cc`, `rasMsgHandlePeersUpdate`) observed on GB300.

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

Before every submission, the launcher runs `validate_gb300_inputs.py`, a read-only audit of the inputs. It checks official file sizes and sparse fingerprints, the model shard references, the 700 training and 251 validation tasks, and the SIF images they require. The audit uses NVIDIA's unmodified `expected-mounts.csv` and `mountcheck.py`, bundled under `assets/` with pinned SHA-256 hashes, and writes its report to `<run directory>/assets_check.json`. Because inputs are audited before submission, `config.sh` sets `VERIFY_MOUNTS=0` for `run.sub`.

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

MLCommons hosts the initial policy checkpoint at [MLCommons storage](https://training.mlcommons-storage.org/index.html#qwen3-5-397b-benchmark). Download it next to the dataset:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d $DATADIR/qwen35-397b-a17b \
  https://training.mlcommons-storage.org/metadata/qwen35-397b-a17b.uri

export HF_CKPT_PATH=$DATADIR/qwen35-397b-a17b
```

`HF_CKPT_PATH` must name the directory that holds `config.json` and the 94 weight shards; a HuggingFace hub cache `snapshots/<sha>` directory also works. The launcher sets the Megatron-format copy location to `NRL_MEGATRON_CHECKPOINT_DIR=$RL_ROOT/mcore_ckpt_1013_gb300_tp4pp2ep16`. The first run converts the checkpoint into that directory, outside the timed region; later runs reuse the converted copy.

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

## 4. Launch on GB300

From the repository checkout, prepare the nodes once. This step needs sudo on the controller. It runs one root Slurm step on the 14 nodes, validates Lustre access and Enroot, and grants the submitting user passwordless access only to `/sbin/sysctl vm.drop_caches=3`. Training itself runs as the normal user.

```bash
cd "$HOME/MLPerf-v6.1-Google-GPU/benchmarks/qwen35_397b_grpo/implementations/14xGB300_NVL72"

bash setup_gb300_nodes.sh
```

Then launch:

```bash
bash launch_gb300_train14.sh \
  "$HOME/images/qwen35_397b_grpo-arm+20261013.sqsh"
```

```text
partition=a4xmaxpartition
nodes=14
segment=14
tasks-per-node=1
gpus-per-node=4 (--gpus-per-node=4)
exclusive allocation, all host memory, no requeue
allocation time limit=10 hours
```

`launch_gb300_train14.sh` is the launcher that produced the recorded runs; it was named `launch_gb300_train14_1013.sh` at run time. Its node list (`xtyslurma4-a4xmaxnodeset1-[0,2-14]`) and partition are site-specific, and `setup_gb300_nodes.sh` uses the same node list.

### What the launcher does

1. Reads `site.sh`, creates a fresh run directory, and runs the read-only input audit (`validate_gb300_inputs.py`, section 3.2).
2. Uses the supplied image from Lustre, staging it onto Lustre first if necessary, and extracts the benchmark launch tree from that image.
3. Draws a seed and resolves NVIDIA's 14-node profile in a clean environment, then freezes the resolved settings in `profile.env.sh` and `profile.forward`. The launch stops if the resolved topology is not 8 training + 6 generation nodes, TP4/PP2/EP16/ETP1, two TP4 prefill engines, and one TP16 decode engine.
4. Writes `paths.sh`, `config.sh`, and `job.sh`, and creates a per-run `run.gb300.sub` with two Slurm adaptations: optional account handling and bounded topology ranks 1–14. Physical NVLink grouping still comes from the topology probe's ClusterUUID.
5. Submits the 14-node allocation, starts the Ray cluster, runs training and deferred evaluation, and invokes compliance checking.

The benchmark code executed by this entry point comes from the **supplied image**. Editing the checked-in recipe or `run.gb300.sub` alone does not change that code; source changes must be included in the image used for the run.

`config.sh`, `job.sh`, `paths.sh`, `profile.env.sh`, `profile.forward`, and `run.gb300.sub` in this directory are the files generated for the `result_1` run (seed 1792220389). The files for the other recorded runs differ only in `SEED` and in the run-directory paths.

## 5. Quality

### Quality metric
Grouped pass@4 on the validation tasks: a task passes when at least one of its four generated trajectories receives a positive reward from the task's test harness. The MLPerf log reports it as `eval_accuracy`.

### Quality target
0.69

### Evaluation frequency
Checkpoints are evaluated from step `VAL_START_AT` (10, set by `config_GBS512.sh`) on. With deferred evaluation (`OFFLINE_EVAL=1`), training continues through step 11 (the MLPerf log reports `max_steps=11`), and the saved checkpoints are evaluated afterwards. `run_stop` is timestamped when the policy update of the first step whose checkpoint reaches the target finishes. In all recorded runs, the step-10 checkpoint reached the target.

### Evaluation thoroughness
All 251 validation tasks with four generations each, i.e. 1,004 trajectories per evaluation, sampled at temperature 0.1 and top-p 0.95.

## 6. Additional notes

### Config naming convention

Configuration files use two formats, depending on whether generation combines prefill and decode in each engine or uses separate engines:

- Aggregated generation: `config_<SYSTEM>_<NODES>x<GPUS/NODE>_t<TRAIN>g<GEN>_tp<TP>pp<PP>ep<EP>gtp<GTP>[_mxfp8].sh`.
- Disaggregated prefill/decode: `config_<SYSTEM>_<NODES>x<GPUS/NODE>_t<TRAIN>g<GEN>_tp<TP>pp<PP>ep<EP>_<P_ENGINES>p<PREFILL>d<DECODE>[_mxfp8].sh`, e.g. `config_GB300_14x4_t8g6_tp4pp2ep16_2p4d16_mxfp8.sh`.

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

The global batch size is `NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT`. Each config sources one batch-size schedule, `config_GBS<GBS>.sh`, which sets the hyperparameters that depend on it: prompts and generations per step, learning rate (`LEARNING_RATE`), gradient clip (`MAX_GRAD_NORM`), first evaluation step (`VAL_START_AT`) and maximum number of steps (`MAX_STEPS`). Profiles also set `SEGMENT` for Slurm allocation and `NEMORL_TRAIN_SEGMENT_SIZE` for NeMo-RL training placement. Configs do not export `SBATCH_SEGMENT_SIZE`; this launcher passes `--segment=14` to `sbatch`, so the whole 14-node allocation stays in one NVLink domain, and NeMo RL places training in segments of 8 nodes. Keep training expert-parallel groups within an NVLink domain when changing either setting.

### Seeds
The launcher draws one seed per launch from the operating system's random number generator (`secrets.randbelow(2**31)`), records it in `seed.txt` and `profile.env.sh`, and exports it as `SEED`. With one experiment per job (`NEXP=1`), `run.sub` passes this value to the driver as the GRPO seed, and the value is recorded in the MLPerf log.
