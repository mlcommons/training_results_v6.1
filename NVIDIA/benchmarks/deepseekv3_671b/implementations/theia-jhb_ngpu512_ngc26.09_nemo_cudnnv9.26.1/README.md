## Running NVIDIA Large MoE DeepSeek v3 671B PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Large MoE DeepSeek v3 671B PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- NVIDIA GB300 NVL72 (GB300) or NVIDIA VR NVL72, 4 GPUs per node.
- At least 1.5TB disk space is required (the starting checkpoint alone is ~1.4TB). The dataset download needs about 120GB of scratch, 80GB once its cleanup step has run.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt
export CONT=<docker/registry>/mlperf-nvidia:deepseekv3_671b-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

**DeepSeek v3 671B uses the same dataset as LLama3.1 8b!**
* **If you already have the LLama3.1 8b dataset you can skip this part**
* **If you don't, please mind that there are pointers/names related to 8b - this has been done on purpose**

Set the directory for the data to be downloaded to:
```bash
export DATADIR=<path/to/dataset>
```

From the benchmark directory (`/workspace/llm` inside the container), download the dataset and align the directories with the layout the benchmark expects:

```bash
bash data_scripts/download.sh
```

`data_scripts/download.sub` runs the same steps as a Slurm job inside the container; set `CONT`, `DATADIR` and `HF_DIR`, a host directory that receives the HuggingFace model config. The script also refreshes the HuggingFace model config under `hf_hub/` in the working copy; the container already ships that copy, so nothing else needs to be mounted.

At the end, the directory structure should look like:

```bash
$tree $DATADIR/8b/
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
    |-- llama-3-1-8b-tokenizer.md5
    |-- special_tokens_map.json
    |-- tokenizer.json
    `-- tokenizer_config.json

2 directories, 14 files
```

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

[Megatron-Core](https://github.com/NVIDIA/Megatron-LM) is a large, powerful transformer library developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) on top of Megatron-Core. Megatron-Core has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8/FP4 training on NVIDIA GPUs.

#### 3.3.2 Model Architecture

The model largely follows the paper titled [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).

| Parameter | Value                       |
|-----------|-----------------------------|
| Model Size | 671B parameters             |
| Architecture | Mixture of Experts |
| Sequence Length | 4096                        |

#### 3.3.3 Model checkpoint

MLCommons hosts the checkpoint for download **exclusively by MLCommons Members** at [MLCommons storage](https://training.mlcommons-storage.org/index.html#deepseekv3-benchmark).

To download, use the [MLCommons R2 Downloader](https://github.com/mlcommons/r2-downloader):

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/deepseekv3_checkpoint_bf16_megatron_bridge.uri
```

The checkpoint is ~1.4TB and is distributed as a Megatron-Bridge distributed checkpoint. The downloader writes it into the current working directory, which then holds `latest_train_state.pt` and the `iter_0000000/` directory:

```bash
export LOAD_CHECKPOINTS_DIR=</path/to/checkpoint>  # host directory that contains latest_train_state.pt and iter_0000000/; mounted at /checkpoints
export LOAD_CHECKPOINT=/checkpoints                # container path to load from; point it at the subdirectory if you downloaded into one
```

Setting `LOAD_CHECKPOINT` also adds the checkpoint files to the pre-run mount check, which logs missing files to `*_mountcheck.log` but does not stop the job; if the path holds no checkpoint, training then starts from random weights.

The checkpoint was produced as follows: the original HuggingFace DeepSeek V3 671B checkpoint was loaded into Megatron-Bridge and trained for **50 iterations** with a sequence auxiliary load balancing loss weight of **1e-2**. This warm-up step is necessary because this benchmark uses the Llama 3.1 8B tokenizer instead of the original DeepSeek tokenizer - the change in token distribution causes MoE router load imbalance, and the 50 iterations allow the router to adapt before the main benchmark training begins.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export DATADIR=<as/set/above>
export LOAD_CHECKPOINTS_DIR=<as/set/above>
export LOAD_CHECKPOINT=<as/set/above>
export CONT=<as/set/above>
source config_GB300_128x4x120xtp1pp4ep32cp1_mxfp8_full_cg.sh  # select config and source it
sbatch --segment=${SEGMENT} -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

## 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.6

### Evaluation frequency
Evaluate after every one step

### Evaluation thoroughness
Evaluation on the validation subset that consists of 1024 sequences (=4.2M tokens).


## 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/DP-RANK>xtp<TP>pp<PP>ep<EP>cp<CP>[_<precision>][_full_cg].sh`, where TP is tensor parallel, PP is pipeline parallel, EP is expert parallel, CP is context parallel, `<precision>` is `mxfp8` or `fp4` (bf16 when absent) and `_full_cg` marks full-iteration CUDA graphs. For example `config_GB300_128x4x120xtp1pp4ep32cp1_mxfp8_full_cg.sh` runs 64 VR nodes x 4 GPUs with 240 sequences per data-parallel rank, TP1 PP4 EP32 CP1, MXFP8 and full-iteration CUDA graphs. `<SYSTEM_NAME>` is `GB300` or `VR`.

Notice here that:

```
MP = TP * PP * CP
DP = WS // MP = (NNODES * GPUS_PER_NODE) / (TP * PP * CP)
GBS = miniBS * DP
```

where:
```
MP = model parallelism
TP = tensor parallelism
PP = pipeline parallelism
CP = context parallelism
DP = data parallelism
WS = world size (number of nodes x number of gpus per node)
miniBS = batch per data-parallel rank (the <BATCH/DP-RANK> token, `MINIBS`)
GBS = global batch size
```

Note: changing `MICRO_BATCH_SIZE` doesn't affect GBS or any of the above parameters.
Effectively it controls gradient accumulation (`GA = miniBS // microBS`).

Recommendation on adjusting the knobs:
1. GBS should be divisible by `DP * VP`, where VP represents Virtual Pipelining, controlled by environment variable `INTERLEAVED_PIPELINE`.
2. DeepSeek V3 671B needs to run at GBS=15360 or above. To use more GPUs, increase `DGXNNODES` and decrease `MINIBS` proportionally so that GBS stays at 15360. When `MINIBS` does not divide proportionally at GBS=15360, switch to GBS=16384.


### Seeds
Megatron-Core builds the dataset sample index on one process only and derives the index file name from the `SEED` value.
Thus, all processes need to have the same value of `SEED` otherwise will not be able to read the data.
The `SEED` environment variable can be set prior to launching the job, otherwise it is set in `run.sub`.
