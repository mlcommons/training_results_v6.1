#!/bin/bash

cd ../pytorch
export CONT=./nvdlfwea+mlperftv60+llama2_70b_lora-amd+.sqsh
source config_G4L4-AD3-LAX7_1x8x1xtp1pp1cp1_fp4.sh 
export DATADIR=/path/to/datasets
export MODEL=/path/to/model
export LOGDIR=/path/to/logdir
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G4L4-AD3-LAX7

sbatch -N $DGXNNODES -t $WALLTIME run.sub
