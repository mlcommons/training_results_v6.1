## Running NVIDIA Llama 2 70B LoRA PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Llama 2 70B LoRA PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- NVIDIA VR NVL72, 4 GPUs per node.
- At least 300GB disk space is required.
- GPUs are not needed for the download/preprocessing scripts, but are needed for training.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
export CONT=<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.

The dataset download/preprocessing script is included in the container. To invoke it, you need either a docker or slurm/enroot environment. Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset and the model checkpoint:

```bash
docker run -it --rm --network=host --ipc=host --volume </path/to/dataset>:/data $CONT

# now you should be inside the container in the /workspace/ft-llm directory
python scripts/download_dataset.py --data_dir /data/gov_report  # download and preprocess dataset; takes less than 1 minute
```

At the end, the directory structure should look like:

```
$tree /data/gov_report
/data/gov_report
|-- train.npy
`-- validation.npy
```

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

The model is [Llama 2 70B](https://huggingface.co/meta-llama/Llama-2-70b-hf), described in [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Touvron et al., 2023) and distributed by Meta under the [Llama 2 Community License](https://ai.meta.com/llama/license/). The fine-tuning dataset is [GovReport](https://gov-report-data.github.io/), described in [Efficient Attentions for Long Document Summarization](https://arxiv.org/abs/2104.02112) (Huang et al., 2021).

[Megatron-Core](https://github.com/NVIDIA/Megatron-LM) is a library of GPU-optimized transformer building blocks developed by NVIDIA. This benchmark fine-tunes the model with [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) on top of Megatron-Core, integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP4/FP8 training on NVIDIA GPUs.

#### 3.3.2 Model Architecture

The model follows the paper titled [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288). The base weights are frozen; only LoRA adapters on the attention QKV and output projections are trained.

| Parameter | Value |
|-----------|-------|
| Model Size | 70B parameters |
| Architecture | Dense decoder-only transformer, 80 layers |
| Sequence Length | 8192 |
| LoRA rank / alpha | 16 / 32 |
| LoRA target modules | `linear_qkv`, `linear_proj` |

#### 3.3.3 Model checkpoint

Training starts from the Llama 2 70B weights converted to a Megatron-Bridge distributed checkpoint. From inside the container started in 3.2, download the converted checkpoint next to the dataset:

```bash
python scripts/download_model.py --model_dir /data/model  # download model checkpoint used for initialization; could take up to 30 minutes
```

or simply run the download/preprocessing slurm script, which downloads both the dataset and the checkpoint:

```bash
export DATADIR="</path/to/dataset>/gov_report"  # set your </path/to/dataset>
export MODEL="</path/to/dataset>/model"  # set your </path/to/dataset>
sbatch -N1 -t 4:00:00 scripts/download.sub
```

You can also use the `scripts/convert_model.py` script, which downloads the original Llama 2 70B model and converts it to a Megatron-Bridge distributed checkpoint, e.g. `python scripts/convert_model.py --save_path /data/model`. This script requires that you either: have set the `HF_TOKEN` with granted access to `meta-llama/Llama-2-70B-hf`, or have already downloaded the Llama 2 70B checkpoint and set `HF_HOME` to its location.

At the end, the directory structure should look like:

```
$tree /data/model
/data/model
|-- iter_0000000
|   |-- __0_0.distcp
|   |-- __0_1.distcp
|   |-- common.pt
|   |-- metadata.json
|   |-- modelopt_run_config.yaml
|   |-- run_config.yaml
|   |-- tokenizer
|   |   |-- special_tokens_map.json
|   |   |-- tokenizer.json
|   |   |-- tokenizer.model
|   |   `-- tokenizer_config.json
|   `-- train_state.pt
|-- latest_checkpointed_iteration.txt
`-- latest_train_state.pt
```

Exit the container.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export DATADIR="</path/to/dataset>/gov_report"  # set your </path/to/dataset>
export MODEL="</path/to/dataset>/model"  # set your </path/to/dataset>
export LOGDIR="</path/to/output_logdir>"  # set the place where the output logs will be saved
export CONT=<as/set/above>
source config_VR_2x4x1xtp1pp1cp1_fp4.sh  # select config and source it
sbatch --segment=${SEGMENT} -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

## 5. Quality

### Quality metric
Cross entropy loss

### Quality target
0.925

### Evaluation frequency
Every 384 sequences, i.e. every CEIL(384 / global_batch_size) steps. The first FLOOR(0.125 * global_batch_size + 2) evaluations are skipped.

### Evaluation thoroughness
Evaluation on the validation subset that consists of 173 examples.

## 6. Additional notes

### Fused kernels and TE patches

The shipped config enables the optional `LORA_PATCH_*` kernels under `lora_kernels/` and the TE patch under `te_patches/`; see `lora_kernels/README.md` for what each one replaces and how it relates numerically to the stock path.

### Config naming convention

Configuration files follow the format `config_<system>_<nodes>x<gpus>x<batch>xtp<X>pp<Y>cp<Z>[_<precision>].sh`, e.g. `config_VR_2x4x1xtp1pp1cp1_fp4.sh`, where:

- `<system>` is the target system: `VR` for NVIDIA VR NVL72,
- `<nodes>` is the number of nodes (`DGXNNODES`),
- `<gpus>` is the number of GPUs per node (`DGXNGPU`),
- `<batch>` is the batch size per data-parallel rank (`MINIBS`, see below),
- `X`, `Y`, `Z` are the tensor parallel (TP), pipeline parallel (PP) and context parallel (CP) sizes,
- `_<precision>` is an optional suffix: `_fp4` trains the linear layers in NVFP4 (the config additionally sources `config_common_fp4.sh`); no suffix means FP8.

Notice here that:

```
MP = TP * PP * CP
DP = WS // MP = (NNODES * GPUS_PER_NODE) / (TP * PP * CP)
GBS = MINIBS * DP
```

where:
```
MP = model parallelism
DP = data parallelism
WS = world size (number of nodes x number of gpus per node)
GBS = global batch size
```

Note: changing `MBS` (micro batch size) doesn't affect GBS or any of the above parameters.
Effectively it controls gradient accumulation (`GA = MINIBS // MBS`).

### Seeds
The `SEED` environment variable can be set prior to launching the job, otherwise `run.sub` draws a random one.
`run.sub` derives one seed per experiment (`SEED_BASE - 1 + experiment_index`) and exports it as `SEED` to every rank, where it seeds both the model RNG and the data loader.
