## Running NVIDIA Large Language Model Llama 3.1 8B PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Large Language Model Llama 3.1 8B PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- NVIDIA VR NVL72, 4 GPUs per node.
- At least 100GB disk space is required.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:llama31_8b-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:llama31_8b-pyt
export CONT=<docker/registry>/mlperf-nvidia:llama31_8b-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

Set the directory for the data to be downloaded to:
```bash
export DATADIR=<path/to/dataset>
```

To download the dataset and align the directories with the layout the benchmark expects, run:

```bash
bash data_scripts/download_8b.sh
```

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
    |-- special_tokens_map.json
    |-- tokenizer.json
    `-- tokenizer_config.json

2 directories, 13 files
```

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

[Megatron-Core](https://github.com/NVIDIA/Megatron-LM) is a large, powerful transformer library developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) on top of Megatron-Core. Megatron-Core has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8/FP4 training on NVIDIA GPUs.

#### 3.3.2 Model Architecture

The model largely follows the paper titled [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).

| Parameter | Value |
|-----------|-------|
| Model Size | 8B parameters |
| Architecture | Dense decoder-only Transformer |
| Number of Layers | 32 |
| Sequence Length | 8192 |

#### 3.3.3 Model checkpoint

The Llama 3.1 8B model is trained from scratch and is not using a checkpoint.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export DATADIR=<as/set/above>
export CONT=<as/set/above>
source config_VR_2x4x2xtp1pp1cp1_8b.sh  # select config and source it
sbatch --segment=${SEGMENT} -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

## 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.3

### Evaluation frequency
Evaluate after every 12,288 sequences (=100M tokens)

### Evaluation thoroughness
Evaluation on the validation subset that consists of 1024 sequences (=8.4M tokens).

## 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<system>_<nodes>x<gpus>x<batch>xtp<X>pp<Y>cp<Z>_8b[_<precision>].sh`, e.g. `config_VR_2x4x2xtp1pp1cp1_8b.sh`, where:

- `<system>` is the target system: `VR` for NVIDIA VR NVL72,
- `<nodes>` is the number of nodes (`DGXNNODES`),
- `<gpus>` is the number of GPUs per node (`DGXNGPU`),
- `<batch>` is the batch size per data-parallel rank (`MINIBS`, see below),
- `X`, `Y`, `Z` are the tensor parallel (TP), pipeline parallel (PP) and context parallel (CP) sizes,
- `_8b` is the model size,
- `_<precision>` is an optional suffix for configs that deviate from the default precision; the shipped VR configs use NVFP4 (`config_common_fp4.sh`) and carry no suffix.

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

Recommendation on adjusting the knobs:
1. GBS should be divisible by `DP * VP`, where VP represents Virtual Pipelining, controlled by environment variable `INTERLEAVED_PIPELINE`.
2. Model's number of layers, controlled by `OVERWRITTEN_NUM_LAYERS` knob (set to 32 in `config_common_8b.sh`), should be divisible by PP * VP.

### Seeds
Megatron-Core builds the dataset sample index on one process only and derives the index file name from the `SEED` value.
Thus, all processes need to have the same value of `SEED` otherwise will not be able to read the data.
The `SEED` environment variable can be set prior to launching the job, otherwise it is set in `run.sub`.
