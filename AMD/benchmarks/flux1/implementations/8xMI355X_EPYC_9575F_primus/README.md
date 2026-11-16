# Flux1 Primus Training

MLPerf Training benchmark for Flux1 (text-to-image diffusion transformer) on the
Primus diffusion backend.

## Build and run

To build the container and run the training, follow these steps:

```bash
cd flux1/primus
DOCKER_BUILDKIT=1 docker build -t mlperf_flux1:integrated -f Dockerfile .
```


## Data

Download once to shared storage that every node can read:

```bash
PREPROCESSED_PATH=/shared/path/to/flux1/data ./scripts/download_ref_data.sh
```

`DATADIR` must contain the three downloaded directories, which training reads as-is:

```
$DATADIR/
├── cc12m_preprocessed
├── coco_preprocessed
└── empty_encodings
```


## Set Configuration

Set appropriate configuration and system-specific hyperparameters.
MI355X submission configurations are in `config_MI355X_*.sh`:

| Config File | System | Nodes × GPUs | Global Batch |
|-------------|--------|--------------|--------------|
| `config_MI355X_1x8x32.sh` | MI355X | 1 × 8 | 512 |
| `config_MI355X_4x8x32.sh` | MI355X | 4 × 8 | 1024 |
| `config_MI355X_8x8x32.sh` | MI355X | 8 × 8 | 2048 |

Every Flux implementation folder retains all available platform/node configs,
regardless of the system named by the folder. Currently, Flux Primus provides
the three MI355X configs above. Source the config for your system before
launching; it sets the node count `NNODES`.

## Launch Training

Edit the `#SBATCH` header in `job.sh` (partition, and nodelist if needed), then
submit with the node count from the sourced config:

```bash
export CONT=mlperf_flux1:integrated
export DATADIR=/path/to/flux1/data
export LOGDIR=/path/to/results
source config_MI355X_8x8x32.sh      # pick appropriate node config
sbatch --nodes=$NNODES job.sh
```

`job.sh` exits if no config was sourced or if the allocation does not match the
config's node count.

`job.sh` runs 10 trials back to back with compliance checking, as a submission
requires. For a single trial on a single node, set `NEXP=1` after sourcing
the appropriate config:

```bash
source config_MI355X_1x8x32.sh
NEXP=1 sbatch --nodes=$NNODES job.sh
```

Submission logs are written per trial to `$LOGDIR/<DATESTAMP>_<N>.log`.

## Directory Structure

```
flux1/primus/
├── Dockerfile
├── job.sh
├── config_MI355X_8x8x32.sh
├── config_MI355X_4x8x32.sh
├── config_MI355X_1x8x32.sh
├── config_common.sh
├── run_and_time.sh
├── run_with_docker_slurm.sh
├── run_with_docker.sh
├── scripts/
├── rocr/libhsa-runtime64.so.1
└── conf/flux.1_schnell_t2i-pretrain.yaml
```

## Model Details

- **Flux Schnell**: 19 double-stream (MMDiT) layers + 38 single-stream layers
- **Hidden size**: 3072, **Attention heads**: 24
- **Target**: Validation loss ≤ 0.586
