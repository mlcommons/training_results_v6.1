## Steps to launch training

### hecate_ngpu512_ngc26.09_nemo

Launch configuration and system-specific hyperparameters for the
hecate_ngpu512_ngc26.09_nemo submission are in the
`benchmarks/llama31_8b/implementations/hecate_ngpu512_ngc26.09_nemo/config_VR_128x4x1xtp2pp1cp4_8b.sh` script.

Steps required to launch training for hecate_ngpu512_ngc26.09_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_VR_128x4x1xtp2pp1cp4_8b.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch --segment=${SEGMENT} -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
