## Running NVIDIA Small MoE GPT OSS 20B PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Small MoE GPT OSS 20B PyTorch MLPerf Benchmark on NVIDIA hardware.

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
docker build -t <docker/registry>/mlperf-nvidia:gpt_oss_20b-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:gpt_oss_20b-pyt
export CONT=<docker/registry>/mlperf-nvidia:gpt_oss_20b-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

**GPT OSS 20B uses the same dataset as LLama3.1 8b!**
* **If you already have the LLama3.1 8b dataset you can skip this part**
* **If you don't, please mind that there are pointers/names related to 8b - this has been done on purpose**

Set the directory for the data to be downloaded to:
```bash
export DATADIR=<path/to/dataset>
```

To download the dataset and align the directories with the layout the benchmark expects, run:

```bash
bash data_scripts/download.sh
```

At the end, the directory structure should look like:

```bash
$tree 8b/
8b/
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

The model largely follows the paper titled [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925).

| Parameter | Value |
|-----------|-------|
| Model Size | 20B parameters |
| Architecture | GPT with Mixture of Experts |
| Sequence Length | 8192 |


#### 3.3.3 Model checkpoint

The GPT OSS 20B model is trained from scratch and is not using a checkpoint.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export DATADIR=<as/set/above>
export CONT=<as/set/above>
source config_VR_2x4x3xtp1pp1cp1ep1_mxfp8.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

## 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.34

### Evaluation frequency
Evaluate after every 12,288 sequences (=100M tokens)

### Evaluation thoroughness
Evaluation on the validation subset that consists of 1024 sequences (=8.4M tokens).


## 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM>_<NODES>x<GPUS/NODE>x<MINIBS>xtp<TP>pp<PP>cp<CP>ep<EP>[_<PRECISION>].sh`, e.g. `config_VR_18x4x1xtp1pp1cp2ep4_mxfp8.sh`, where:

```
SYSTEM    = target system (VR = NVIDIA VR NVL72)
NODES     = number of nodes (DGXNNODES)
GPUS/NODE = GPUs per node (DGXNGPU)
MINIBS    = batch size per data-parallel rank (MINIBS)
TP        = tensor parallelism (TENSOR_MODEL_PARALLEL)
PP        = pipeline parallelism (PIPELINE_MODEL_PARALLEL)
CP        = context parallelism (CONTEXT_PARALLEL)
EP        = expert parallelism (EXPERT_PARALLEL)
PRECISION = low-precision recipe suffix; mxfp8 sources config_common_mxfp8.sh
```

Notice here that: 

```
MP = TP * PP * CP
DP = WS // MP = (NODES * GPUS/NODE) / (TP * PP * CP)
GBS = MINIBS * DP
```

where: 
```
MP = model parallelism
DP = data parallelism
WS = world size (number of nodes x number of gpus per node)
GBS = global batch size
```

Note: changing `MICRO_BATCH_SIZE` doesn't affect GBS or any of the above parameters.
Effectively it controls gradient accumulation (`GA = MINIBS // MICRO_BATCH_SIZE`).

Recommendation on adjusting the knobs: 
1. GBS should be divisible by `DP * VP`, where VP represents Virtual Pipelining, controlled by environment variable `INTERLEAVED_PIPELINE`. 
2. Model's number of layers, controlled by `OVERWRITTEN_NUM_LAYERS` knob (with default 24), should be divisible by `PP * VP`. 


### Seeds
Megatron-Core builds the dataset sample index on one process only and derives the index file name from the `SEED` value.
Thus, all processes need to have the same value of `SEED` otherwise they will not be able to read the data.
The `SEED` environment variable can be set prior to launching the job, otherwise it is set in `run.sub`.
