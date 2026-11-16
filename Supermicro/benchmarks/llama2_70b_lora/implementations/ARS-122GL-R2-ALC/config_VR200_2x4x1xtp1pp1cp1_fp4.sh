#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=550
export LR=0.0006
export MINIBS=1
export CP=1
export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
export NUM_WORKERS=4

export HEALING_ITER=350

# Keep the pinned FP8 healing weights on the NUMA node of each GPU.
# Without this, two GPUs per node copy them at about 19 GB/s at HEALING_ITER.
export BINDCMD="bindpcie --cpu=node"

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=2
export DGXNGPU=4
export WALLTIME_RUNANDTIME=35
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
