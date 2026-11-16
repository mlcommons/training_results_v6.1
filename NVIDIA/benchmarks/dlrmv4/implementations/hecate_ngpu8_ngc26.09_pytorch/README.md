## Running NVIDIA Generative Recommender DLRMv4 (HSTU) PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Generative
Recommender DLRMv4 (HSTU) PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- NVIDIA GB200 NVL72 (GB200), NVIDIA GB300 NVL72 (GB300) or NVIDIA VR NVL72 (VR), 4 GPUs per node.
- At least 300GB disk space is required for the preprocessed dataset, plus
  another 200GB for the HuggingFace download/conversion cache (`HF_HOME`,
  see below) unless it shares the same volume.
- Dataset preparation needs no GPU but about 900GB of host RAM: the preprocessing
  step loads the whole 4.8B-event stream into memory and peaks at ~870GB RSS (it is
  OOM-killed on a 188GB node). Download takes ~1 hour, preprocessing ~40 minutes on a
  920GB node.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:dlrmv4-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:dlrmv4-pyt
export CONT=<docker/registry>/mlperf-nvidia:dlrmv4-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

Set the directory for the data to be downloaded to:

```bash
export DATADIR=<path/to/dataset>
```

Also redirect the HuggingFace cache off your home directory — the
raw-to-parquet conversion step can use 200GB+ there, which easily exceeds a
typical home quota:

```bash
export HF_HOME=<path/to/hf_cache>
```

Download and preprocess Yambda-5b (from HuggingFace):

```bash
./download_dataset.sh
./verify_dataset.sh
```

At the end, the directory structure should look like:

```
raw/5b/multi_event.parquet
shared_metadata/{artist_item_mapping,album_item_mapping,embeddings}.parquet
processed_5b/{train_sessions,test_events,session_index}.parquet
processed_5b/item_popularity.npy
processed_5b/split_meta.json
```

At the start of the first training run, the benchmark builds a shared mmap
cache from `train_sessions.parquet`. Offline time-window indices are enabled by
default and are generated in the same cache. With the default history length
and one-day time windows, the relevant layout is:

```text
processed_5b/hstu_cache/
├── _READY
├── store_meta.json
├── flat_*.npy
├── user_start.npy
├── user_end.npy
├── unique_uids.npy
├── positions_L4086.npy
├── ts_indices_L4086_W86400.npy
└── ts_offsets_L4086_W86400.npy
```

The flat event store is shared across history lengths. Derived files include
their history length and time-window size in the filename. Subsequent runs mmap
and reuse these files instead of rebuilding them, so `DATADIR` must be writable
during the first run.

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

The model is **HSTU** (Hierarchical Sequential Transduction Units), the
generative-recommender architecture from Meta's ICML'24 paper *Actions Speak
Louder than Words: Trillion-Parameter Sequential Transducers for Generative
Recommendations* ([arXiv:2402.17152](https://arxiv.org/abs/2402.17152)).

The benchmark code is vendored from the MLPerf Training reference
[mlcommons/training](https://github.com/mlcommons/training/tree/master/recommendation),
originally added via [mlcommons/training#889](https://github.com/mlcommons/training/pull/889).

#### 3.3.2 Model Architecture

HSTU replaces the feature-interaction stack of a classic DLRM with a stack of
pointwise-attention "transducer" layers operating over the user's interaction
sequence: sparse embedding tables (sharded with TorchRec
`DistributedModelParallel`) feed an HSTU attention stack, computed with a
fused jagged-attention kernel in bf16, trained against a single `listen_plus`
binary task.

#### 3.3.3 Model checkpoint

DLRMv4 is trained from scratch and does not use a starting checkpoint.

## 4. Launch training

```bash
export DATADIR=<as/set/above>
export LOGDIR=</path/to/output/dir>
export CONT=<as/set/above>
source config_VR_2x4x1024.sh  # select config and source it
sbatch --segment=${SEGMENT} -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

## 5. Quality

### Quality metric

`window_auc`: AUC on the held-out future evaluation window for the
`listen_plus` task.

### Quality target

0.75

### Evaluation frequency

Evaluate every 0.1% of the training stream (`EVAL_EVERY_DATA_PCT=0.001`).

### Evaluation thoroughness

Every evaluation scores the entire held-out future window, with no
subsampling.

## 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM>_<NODES>x<GPUS/NODE>x<BATCH/GPU>.sh`, where
`<SYSTEM>` is `GB200`, `GB300` or `VR`; e.g. `config_VR_2x4x1024.sh` runs on 2 nodes with 4 GPUs each and
1024 samples per GPU, a global batch size of 8192. The model is sharded with TorchRec
`DistributedModelParallel` only, so there is no tensor, pipeline or context parallelism token.
`DENSE_LR` and `SPARSE_LR` are the only tunable hyperparameters and scale linearly with the global batch size.

### Seeds

The `SEED` environment variable can be set prior to launching the job, otherwise `run.sub` draws a random one
and derives one seed per experiment from it. `SEED` seeds each embedding table's initializer deterministically
and must be identical across every rank for initialization to match.
