#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters - borrowed from Inventec
export MAX_STEPS=500
export LR=0.00075
export MINIBS=1
export CP=1
export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
export NUM_WORKERS=4

export HEALING_ITER=350

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=35
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
