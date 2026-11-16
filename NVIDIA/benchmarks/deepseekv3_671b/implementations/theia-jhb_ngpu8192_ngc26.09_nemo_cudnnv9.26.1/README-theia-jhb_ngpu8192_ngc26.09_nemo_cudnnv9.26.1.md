## Steps to launch training

### theia-jhb_ngpu8192_ngc26.09_nemo_cudnnv9.26.1

Launch configuration and system-specific hyperparameters for the
theia-jhb_ngpu8192_ngc26.09_nemo_cudnnv9.26.1 submission are in the
`benchmarks/deepseekv3_671b/implementations/theia-jhb_ngpu8192_ngc26.09_nemo_cudnnv9.26.1/config_GB300_2048x4x8xtp1pp4ep32cp1_mxfp8_full_cg.sh` script.

Steps required to launch training for theia-jhb_ngpu8192_ngc26.09_nemo_cudnnv9.26.1.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB300_2048x4x8xtp1pp4ep32cp1_mxfp8_full_cg.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch --segment=${SEGMENT} -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
