#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.0005
#export MINIBS=2
#export CP=8
#also can try
export MINIBS=1
export CP=4
export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
#export NUM_WORKERS=2 #default
export NUM_WORKERS=8 #better
#export NUM_WORKERS=16 #same as 8
export EVAL_CONTEXT_PARALLEL_SIZE=2

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=8
export DGXNGPU=4
export SEGMENT=8
export WALLTIME_RUNANDTIME=12
#export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
export WALLTIME=unlimited
