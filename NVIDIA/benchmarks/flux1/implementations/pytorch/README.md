## Running NVIDIA text-to-image FLUX.1 PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA text-to-image FLUX.1 PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- NVIDIA GB300 NVL72 (GB300) or NVIDIA VR NVL72, 4 GPUs per node.
- At least 6TB disk space is required.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:flux1-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:flux1-pyt
export CONT=<docker/registry>/mlperf-nvidia:flux1-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

The dataset download/preprocessing scripts are included in the container. To invoke them, you need either a docker or slurm/enroot environment. Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset:

```bash
docker run -it --rm --network=host --ipc=host --volume </path/to/dataset>:/dataset $CONT
# if using srun, instead do: srun --nodes=1 -t 24:00:00 --pty --container-image=${CONT} --container-mounts=</path/to/dataset>:/dataset -p <partition> -A <account> /bin/bash
# now you should be inside the container in the /workspace/flux directory
pip install datasets  # HF datasets, read by scripts/to_webdataset.py

# download dataset and empty encodings
cd /dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri

# convert to webdataset format
mkdir energon
python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/cc12m_preprocessed --output_path /dataset/energon/train --num_workers 8
python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/coco_preprocessed --output_path /dataset/energon/val --num_workers 8

# build the per-shard .tar.idx indices read by the data loader
# (megatron-energon is only needed for this step and is not shipped in the container)
pip install megatron-energon==7.2.2
cd energon
python /workspace/flux/scripts/energon_prepare.py ./ --num-workers 8 --template-dir /workspace/flux/scripts/dataset_template/

# copy over empty_encodings
cp -r ../empty_encodings .

# (Optional) to reclaim space delete /dataset/cc12m_preprocessed and /dataset/coco_preprocessed
```

or simply run the download/preprocessing slurm script:

```bash
export DATADIR="</path/to/dataset>"  # set your </path/to/dataset>
export CONT=<as/set/above>
sbatch -N1 -t 24:00:00 scripts/download.sub --download --preprocess  # you may be required to set --account and --partition here
```

After, the data structure should look like:

```
/dataset
└── energon
    ├── .nv-meta/                  # written by scripts/energon_prepare.py, not read by training
    ├── train/                     # 110 shards
    │   ├── index.txt
    │   ├── shard_000000.tar
    │   ├── shard_000000.tar.idx
    │   └── ...
    ├── val/                       # 3 shards
    │   ├── index.txt
    │   ├── shard_000000.tar
    │   ├── shard_000000.tar.idx
    │   └── ...
    └── empty_encodings/
        ├── clip_empty.npy
        └── t5_empty.npy
```

Training mounts only the `energon` subdirectory: export it as `DATAROOT` when launching (section 4). `run.sub` verifies this layout against `expected-mounts.csv` before every run (`VERIFY_MOUNTS=1`, the default), so a mismatch is reported up front instead of as a training failure.

Exit the container.

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

This benchmark trains the [FLUX.1](https://github.com/black-forest-labs/flux) rectified-flow transformer by Black Forest Labs, following [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206). The model code is derived from the FLUX.1 implementation in [TorchTitan](https://github.com/pytorch/torchtitan) (BSD-style license) and has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8/FP4 training on NVIDIA GPUs.

#### 3.3.2 Model Architecture

The model is the FLUX.1 transformer, trained from scratch with the parameters below. The text encoders (T5, CLIP) and the VAE are not trained: their outputs are precomputed and shipped in the dataset prepared in section 3.2.

| Parameter | Value |
|-----------|-------|
| Model Size | ~11.9B parameters |
| Hidden size | 3072 |
| Attention heads | 24 |
| Double-stream blocks | 19 |
| Single-stream blocks | 38 |
| Text context dimension (T5) | 4096 |
| Pooled text dimension (CLIP) | 768 |
| Latent input channels | 64 |

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export DATAROOT="</path/to/dataset>/energon"  # the energon subdirectory prepared above
export LOGDIR="</path/to/output_logdir>"  # set the place where the output logs will be saved
export CONT=<as/set/above>
source config_VR_02x04x64.sh  # select config and source it
sbatch --segment=${SEGMENT} -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

## 5. Quality

### Quality metric
Validation loss averaged over 8 equidistant time steps [0, 7/8], as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206).
Each validation sample carries a pre-assigned integer timestep from 0 to 7 inclusive, which is divided by `8.0` to obtain the timestep. The loss is the mean squared error between the predicted and the target velocity. Because the validation set holds an equal number of samples per timestep, the plain mean of the per-sample losses equals the mean of the per-timestep means, and that is what the benchmark reports.

### Quality target
0.586

### Evaluation frequency
Every 262,144 training samples, rounded up to a whole number of steps.

### Evaluation thoroughness
29,696 samples

## 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>.sh`, where `<SYSTEM_NAME>` is `GB300` or `VR`; e.g. `config_VR_18x04x16.sh` runs on 18 VR nodes with 4 GPUs each and 16 samples per GPU.

Notice here that:

```
GBS = NODES * GPUS/NODE * BATCH/GPU
```

where GBS is the global batch size (`BATCHSIZE` in the config). The model is trained data-parallel only, with no tensor, pipeline or context parallelism and no gradient accumulation. Each config sources `config_common_gbs_<GBS>.sh` for the learning rate and warm-up steps that belong to that batch size.

### Seeds
`run.sub` gives each experiment its own `SEED`, drawn from `$RANDOM`. If `SEED` is exported before launching the job, the first experiment uses that value and later experiments still draw their own. The training script seeds model initialisation and the data pipeline from that single value, so all ranks must receive the same `SEED`; `run.sub` takes care of this. The seed used is recorded in the MLPerf log.
