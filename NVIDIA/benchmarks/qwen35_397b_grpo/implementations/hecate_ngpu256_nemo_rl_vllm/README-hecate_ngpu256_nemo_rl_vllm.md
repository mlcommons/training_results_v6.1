## Steps to launch training

### hecate_ngpu256_nemo_rl_vllm

Launch configuration and system-specific hyperparameters for the
hecate_ngpu256_nemo_rl_vllm submission are in the
`benchmarks/qwen35_397b_grpo/implementations/hecate_ngpu256_nemo_rl_vllm/config_VR_64x4_t32g32_tp4pp2ep32gtp4_mxfp8.sh` script.

Steps required to launch training for hecate_ngpu256_nemo_rl_vllm.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_VR_64x4_t32g32_tp4pp2ep32gtp4_mxfp8.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch --segment=${SEGMENT} -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
