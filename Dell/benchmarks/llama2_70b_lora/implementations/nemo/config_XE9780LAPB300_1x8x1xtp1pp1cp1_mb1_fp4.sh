#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=500
export LR=0.00075
export MINIBS=1
export CP=1
export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
export NUM_WORKERS=4

export HEALING_ITER=350

# Keep the pinned FP8 healing weights on the NUMA node of each GPU (new drop VR200 intent).
# "bindpcie --cpu=node" maps ranks 6/7 to NUMA nodes 6/7, which do not exist on this
# 6-NUMA-node host (job 60), so use the explicit per-GPU map in bind_XE9780LAP.sh.
export EXTRA_MOUNTS="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/bind_XE9780LAP.sh:/workspace/ft-llm/bind_XE9780LAP.sh:ro"
export BINDCMD="bindpcie --cpu=/workspace/ft-llm/bind_XE9780LAP.sh --mem=/workspace/ft-llm/bind_XE9780LAP.sh"

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=35
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
