## Running Azure Large MoE DeepSeek v3 671B PyTorch MLPerf Training v6.1 Benchmark

This file adapts the README supplied in the NVIDIA training container for running DeepSeek v3 671B on Azure GB200 with MRC.

## 1. Hardware Requirements

- At least 1.5TB disk space is required.
- NVIDIA GPU with at least 190GB memory is strongly recommended (e.g. GB200)
- GPUs are required for the checkpoint processing.
- This configuration uses 2560 nodes with four GPUs per node (10240 GPUs). Allow additional disk space for the dataset, converted checkpoint, image and outputs.

## 2. Software Requirements

- Linux/aarch64, Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot), and PMI2 support
- Python 3.9+, Bash, GNU coreutils and `curl`
- Azure MRC/NCCL, verbs-shim, DOCA and HPC-X packages
- Noninteractive permission for `sudo /sbin/sysctl vm.drop_caches=3`
- [Docker](https://www.docker.com/) for building the container

Use dedicated allocations: the original batch cleanup removes all Enroot containers listed by the running user on the allocated nodes.

## 3. Set up

### 3.1 Build the container

On an aarch64 build host, run the following from the directory containing `Dockerfile`. Replace `your-registry` with your container registry:

```bash
export IMAGE=your-registry/azure-deepseek:v6.1
BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
docker pull --platform=linux/arm64 "$BASE_IMAGE"
FROM_IMAGE_NAME="$(docker image inspect "$BASE_IMAGE" --format '{{index .RepoDigests 0}}')"

docker build --build-arg FROM_IMAGE_NAME="$FROM_IMAGE_NAME" \
    -t "$IMAGE" .
docker push "$IMAGE"
```

The base-image lookup supplies the immutable reference required by the Dockerfile.
On the Slurm system, import the built image with Enroot:

```bash
export IMAGE=your-registry/azure-deepseek:v6.1
enroot import -o deepseek-v6.1.sqsh "docker://${IMAGE}"
export CONT="$(realpath deepseek-v6.1.sqsh)"
```

Make the image accessible at the same absolute path on every training node. A rebuilt image must be qualified before using it for benchmark results.

### 3.2 Prepare dataset

**DeepSeek v3 671B uses the same dataset as LLama3.1 8b!**
* **If you already have the LLama3.1 8b dataset you can skip this part**
* **If you don't, please mind that there are pointers/names related to 8b - this has been done on purpose**

Set the directory for the data to be downloaded to:
```bash
export DATADIR=/path/to/dataset
export HF_DIR="$DATADIR"/config  # the below script will download a HF model config to `HF_DIR`
```

To download the dataset and align the directories with the layout the benchmark expects, run:

```bash
bash data_scripts/download.sh
```

Run this from the implementation directory against a fresh download directory. Its cleanup step deletes unneeded downloaded files and moves the dataset/tokenizer directories.

At the end, the directory structure should look like:

```text
$DATADIR/
|-- 8b
|   |-- LICENSE.txt
|   |-- NOTICE.txt
|   |-- c4-train.en_6_text_document.bin
|   |-- c4-train.en_6_text_document.idx
|   |-- c4-validation-91205-samples.en_text_document.bin
|   |-- c4-validation-91205-samples.en_text_document.idx
|   |-- llama-3-1-8b-preprocessed-c4-dataset.md5
|   `-- tokenizer
|       |-- LICENSE
|       |-- README.md
|       |-- USE_POLICY.md
|       |-- llama-3-1-8b-tokenizer.md5
|       |-- special_tokens_map.json
|       |-- tokenizer.json
|       `-- tokenizer_config.json
`-- config
    |-- hub
    |   `-- models--deepseek-ai--DeepSeek-V3
    |       |-- blobs
    |       |-- refs
    |       |   `-- main
    |       `-- snapshots
    |           `-- e815299b0bcbac849fa540c768ef21845365c9eb
    |               |-- config.json
    |               `-- configuration_deepseek.py
    `-- modules
        |-- __init__.py
        `-- transformers_modules
            |-- __init__.py
            `-- deepseek_hyphen_ai
                |-- __init__.py
                `-- DeepSeek_hyphen_V3
                    |-- __init__.py
                    `-- e815299b0bcbac849fa540c768ef21845365c9eb
                        |-- __init__.py
                        |-- config.json
                        `-- configuration_deepseek.py
```

Stage the prepared `8b` directory on every node. Training mounts it at `/preproc_data` and its tokenizer at `/tokenizer`. The supplied image already contains its offline model-config cache; the host `HF_DIR` is not mounted during training.

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

This implementation uses [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), Megatron-Core and [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8/FP4 training on NVIDIA GPUs.

#### 3.3.2 Model Architecture

The model largely follows the paper titled [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).

| Parameter | Value                       |
|-----------|-----------------------------|
| Model Size | 671B parameters             |
| Architecture | Mixture of Experts |
| Sequence Length | 4096                        |



#### 3.3.3 Checkpoint download

MLCommons hosts the checkpoint for download **exclusively by MLCommons Members** at [MLCommons storage](https://training.mlcommons-storage.org/index.html#deepseekv3-benchmark).

To download, use the [MLCommons R2 Downloader](https://github.com/mlcommons/r2-downloader):

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/deepseekv3_checkpoint_bf16.uri
```

The checkpoint is ~1.4TB.

#### 3.3.4 Starting checkpoint

The checkpoint distributed for this benchmark is provided in HuggingFace format.

It was produced as follows: the original HuggingFace DeepSeek V3 671B checkpoint was loaded into Megatron-Bridge and trained for **50 iterations** with a sequence auxiliary load balancing loss weight of **1e-2**, then converted back to HuggingFace format. This warm-up step is necessary because this benchmark uses the Llama 3.1 8B tokenizer instead of the original DeepSeek tokenizer -- the change in token distribution causes MoE router load imbalance, and the 50 iterations allow the router to adapt before the main benchmark training begins.

#### 3.3.5 Checkpoint preprocessing

The distributed checkpoint is in HuggingFace format and must be converted to Megatron-LM format before training. Follow [the instructions](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo#checkpoint-preprocessing) from the reference implementation.

After conversion, export `LOAD_CHECKPOINTS_DIR` as the absolute path of the converted checkpoint root containing `iter_0000000`. The production config loads iteration 0 from `/checkpoints`.

### 3.4 Azure MRC setup

Provision the communication packages outside the code repository and set `MLPERF_TRAIN_DIR` to their root:

```text
$MLPERF_TRAIN_DIR/
|-- nccl/lib/        # MRC-compatible NCCL libraries
|-- hpcx/            # HPC-X 2.23.1 aarch64 distribution
|-- verbs/lib/       # NCCL verbs shim
`-- host-verbs/lib/  # Host MRC verbs shim
```

Preserve library symlinks. Host DOCA/MRC and NVIDIA driver libraries and `/dev/nvidia-caps-imex-channels/channel0` must be available. The production config sets the MRC library paths, NIC selection and communication environment.

`configs/mrc_mounts.sh` supplies additional host mounts; `MRC_MOUNTS_FILE` can select an equivalent site manifest. Its size requires sufficient kernel argument/environment limits; 4-KiB-page kernels need a compact equivalent. These packages are external prerequisites, not built by this repository; provide the exact distributions or versioned build instructions to the reproducing party.

## 4. Launch training

For training, we use Slurm with the Pyxis extension and PMI2 support to run the container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export CONT=/path/to/deepseek-v6.1.sqsh
export DATADIR=/path/to/dataset
export LOAD_CHECKPOINTS_DIR=/path/to/converted-checkpoint
export MLPERF_TRAIN_DIR=/shared/MLPERF_train
export LOGROOT=/shared/results/deepseek
export NODE_LOGDIR=/local/logs/deepseek
export NODE_TMPDIR=/local/tmp/deepseek
export NEXP=1

python3 launch.py --hostfile /shared/hostfiles/gb200 --seed 1234 \
    --account YOUR_ACCOUNT --partition YOUR_PARTITION --time 60
```

Replace the example paths, seed and Slurm identifiers with the actual run values. The hostfile must contain 2560 distinct hostnames, one per line. The implementation and `LOGROOT` must be shared at identical paths. Create writable `NODE_LOGDIR` and `NODE_TMPDIR` on every node before submission.

The launcher sources `configs/config_GB200_2560x4x12xtp2pp4ep32cp1_mxfp8_full_cg_mrc.sh` before submitting the original experiment-based `run.sub`. It creates a timestamped `LOGDIR` beneath `LOGROOT`. Add `--dry-run` to print the launch without calling Slurm. Allocation time defaults to 60 minutes; the training step limit is 50 minutes.

The selected configuration uses TP=2, PP=4, CP=1, EP=32, four virtual pipeline stages, MINIBS=12, GBS=15360 and MXFP8. Use a clean shell to avoid unrelated environment overrides.

# 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.6

### Evaluation frequency
Evaluate after every training step beginning at iteration 43 (`SKIP_FIRST_EVALS=1`).

### Evaluation thoroughness
Nominal evaluation batch: 1024 sequences of length 4096 (4,194,304 tokens). The evaluation callback pads the pipeline schedule with zero-weight microbatches. Keep `EVAL_BATCH_SIZE` unset as in the production configuration.


# 6. Additional notes

### Config naming convention

The production filename encodes nodes, GPUs per node, MINIBS, and tensor/pipeline/expert/context parallelism. `2560` denotes nodes, not GPUs.

Notice here that: 

```
MP = TP * PP * CP
DP = WS // MP = (NNODES * GPUS_PER_NODE) / (TP * PP * CP)
miniBS = GBS // DP
```

where: 
```
MP = model parallelism
TP = tensor parallelism
PP = pipeline parallelism
DP = data parallelism
WS = world size (number of nodes x number of gpus per node)
GBS = global batch size
```

Note: changing `MICRO_BATCH_SIZE` doesn't affect GBS or any of the above parameters.
Effectively it controls gradient accumulation (`GA = miniBS // microBS`).

For this configuration, WS=10240, MP=8, DP=1280 and GBS=15360. Preserve the supplied pipeline layout and batch settings when reproducing results.


### Seeds
All ranks must use the same dataset seed. Supply the actual per-run seed through `launch.py --seed`; `1234` above is only an example. Submit measured trials as separate jobs with `NEXP=1` to retain their node-local logs separately.

### Runtime source and logs

The supplied image provides the training callbacks, Hydra configuration, frameworks and offline cache. Only the experiment-based `run_and_time.sh` and `pretrain.py` are mounted from this repository; the latter adds the `config_filename` event. Other checked-in Python/YAML files are source snapshots, not overrides of the supplied image.

Rank-zero training output is written on its host to `${NODE_LOGDIR}/slurm-<job-id>_train_0.out` and `_train_0.err`. Cache-clear events are in the node-local `cachelog` files. Collect these original records before node-local storage is reclaimed; the shared orchestration log is not the training result log. Review environment snapshots before publishing.

Use MLPerf `run_start` and successful `run_stop` events for timing. A Slurm `COMPLETED` status is not proof of convergence or compliance. The production config disables the legacy inline checker; run the official v6.1 compliance/package checks separately on the collected results and include the required system description and actual seeds.
