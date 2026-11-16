#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.00055
export MINIBS=4
export CP=2
export FP8_ACT=1
export NCCL_NVLS_ENABLE=0
export MCORE_CUDA_GRAPH=1

export HEALING_ITER=350

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=UNLIMITED
export WALLTIME=UNLIMITED

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu
