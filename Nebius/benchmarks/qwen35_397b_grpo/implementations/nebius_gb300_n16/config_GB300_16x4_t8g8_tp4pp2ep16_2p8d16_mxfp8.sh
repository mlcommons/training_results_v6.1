#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

export NEMO_GYM_REPLICAS=${NEMO_GYM_REPLICAS:-4}

source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_optimizer_cpu_offload.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_recompute_selective.sh"
export GBS="${GBS:-512}"
source "$(dirname "${BASH_SOURCE[0]}")/config_GBS${GBS}.sh"

export RECIPE="/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm_disagg_mxfp8.yaml"
export MLPERF_LINEAR_PRECISION="fp8"

export TRAIN_NODES=8
export GEN_NODES=8
export COLOCATED_GENERATION=0

export GYM_TRAIN_CONCURRENCY=${GYM_TRAIN_CONCURRENCY:-1024}
export GYM_EVAL_CONCURRENCY=${GYM_EVAL_CONCURRENCY:-256}

export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-16}
export PP=${PP:-2}
export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=31

export CTX_ENGINES=2
export GEN_ENGINES=1
export CTX_TP=8
export CTX_MTP=1
export CTX_MEP=8
export GEN_TP=16
export GEN_MTP=1
export GEN_MEP=16

export TRTLLM_TP=16
export TRTLLM_MEP=16
export VAL_BATCH_SIZE=126
export TRTLLM_GPU_UTIL="${TRTLLM_GPU_UTIL:-0.8}"

export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.01}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

export NEMORL_TRAIN_SEGMENT_SIZE="${NEMORL_TRAIN_SEGMENT_SIZE:-8}"
export SEGMENT="${SEGMENT:-8}"

export NRL_USE_FASTOKENS=1

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM="$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')"

export WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-120}
export WALLTIME=$((21 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))
add_grpo_eval_walltime
