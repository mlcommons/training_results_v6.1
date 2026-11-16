#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.0005
export MINIBS=1
export CP=8
export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
export NUM_WORKERS=2
export EVAL_CONTEXT_PARALLEL_SIZE=2

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=16
export DGXNGPU=4
export SEGMENT=16
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
