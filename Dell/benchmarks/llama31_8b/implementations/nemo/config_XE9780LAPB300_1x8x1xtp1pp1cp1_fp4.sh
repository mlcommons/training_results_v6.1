#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

export MINIBS=2
export MICRO_BATCH_SIZE=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export LR=0.00045
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM="1xXE9780x8B300-SXM-288GB"

export EXTRA_MOUNTS="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/bind_XE9780LAP.sh:/workspace/llm/bind_XE9780LAP.sh:ro"
export BINDCMD="bindpcie --cpu=/workspace/llm/bind_XE9780LAP.sh --mem=/workspace/llm/bind_XE9780LAP.sh"
export PMIX_MCA_psec=native
export PMIX_MCA_gds=hash

export WALLTIME_RUNANDTIME=200
export WALLTIME=unlimited #$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
