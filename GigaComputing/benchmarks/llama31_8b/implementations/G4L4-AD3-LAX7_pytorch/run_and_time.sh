#!/bin/bash

cd ../pytorch
export CONT=llama31-8b-x86.sqsh
source config_G4L4-AD3-LAX7_1x8x2xtp1pp1cp1_8b.sh
export DATADIR=/path/to/datasets
export LOGDIR=/path/to/logdir
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G4L4-AD3-LAX7

sbatch -N $DGXNNODES -t $WALLTIME run.sub
