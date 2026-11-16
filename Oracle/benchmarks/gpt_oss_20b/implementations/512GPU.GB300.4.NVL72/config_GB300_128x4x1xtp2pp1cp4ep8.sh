#!/bin/bash

# GPT-OSS 20B: 128 GB300 nodes x 4 GPUs = 512 GPUs.
# Carried forward from the validated 20261006 512-GPU recipe; the
# 20261012 common configuration files are identical.

source config_common.sh
source config_common_mxfp8.sh
source config_common_cg.sh

export MINIBS=1
export MICRO_BATCH_SIZE=1

export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=True
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=4
export EXPERT_PARALLEL=8
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${EXPERT_PARALLEL}

export USE_MNNVL=1
export MOE_EXPERT_RANK_CAPACITY_FACTOR=7
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export NCCL_UB_DP=False

export LR=0.00052
export VAL_CHECK_INTERVAL=192
export LR_WARMUP_STEPS=32

export DGXNNODES=128
export DGXNGPU=4
export SEGMENT=16

export WALLTIME_RUNANDTIME=25
export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))
