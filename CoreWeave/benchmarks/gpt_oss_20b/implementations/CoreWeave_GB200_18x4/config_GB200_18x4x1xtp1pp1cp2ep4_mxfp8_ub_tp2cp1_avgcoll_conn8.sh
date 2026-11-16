#!/bin/bash
source $(dirname ${BASH_SOURCE[0]})/config_GB300_18x4x1xtp1pp1cp2ep4_mxfp8.sh

export NCCL_UB_DP=True
export TENSOR_MODEL_PARALLEL=2
export CONTEXT_PARALLEL=1
export SEQ_PARALLEL=True
export VAL_CHECK_INTERVAL=342
export AVERAGE_IN_COLLECTIVE=True
export CUDA_DEVICE_MAX_CONNECTIONS=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
