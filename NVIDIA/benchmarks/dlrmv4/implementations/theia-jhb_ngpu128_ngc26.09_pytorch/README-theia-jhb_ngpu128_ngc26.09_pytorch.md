## Steps to launch training

### theia-jhb_ngpu128_ngc26.09_pytorch

Launch configuration and system-specific hyperparameters for the
theia-jhb_ngpu128_ngc26.09_pytorch submission are in the
`benchmarks/dlrmv4/implementations/theia-jhb_ngpu128_ngc26.09_pytorch/config_GB300_32x4x256.sh` script.

Steps required to launch training for theia-jhb_ngpu128_ngc26.09_pytorch.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB300_32x4x256.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch --segment=${SEGMENT} -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
