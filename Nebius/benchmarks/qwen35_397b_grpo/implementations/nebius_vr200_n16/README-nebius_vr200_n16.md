## Steps to launch training

### nebius_vr200_n16

Launch configuration and system-specific hyperparameters for the
nebius_vr200_n16 submission are in the
`benchmarks/qwen35_397b_grpo/implementations/nebius_vr200_n16/config_VR_16x4_t8g8_tp4pp2ep16_2p8d16_mxfp8.sh` script.

Steps required to launch training for nebius_vr200_n16. The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_VR_16x4_t8g8_tp4pp2ep16_2p8d16_mxfp8.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
