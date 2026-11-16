#!/bin/bash
# mlperf: XE9780 8x B200 config for the new v6.1 drop (2026-10-15); hyperparameters = v6.1 reference / best published v6.0 x86 B200

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=500
export LR=0.00075
export MINIBS=1
export CP=1
export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
export NUM_WORKERS=8

export HEALING_ITER=350

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=35
export WALLTIME=UNLIMITED
export NCCL_NVLS_ENABLE=0
