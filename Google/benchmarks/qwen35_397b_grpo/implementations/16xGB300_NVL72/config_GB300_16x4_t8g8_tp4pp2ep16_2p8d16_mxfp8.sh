#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# 16 GB300 nodes: 8 training + 8 generation; MXFP8 forward and backward with BF16 Adam state.
# This profile defines its complete topology and runtime settings.

export NEMO_GYM_REPLICAS=${NEMO_GYM_REPLICAS:-4}

# Shared benchmark defaults and batch-size settings remain separate helpers.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_optimizer_cpu_offload.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_recompute_selective.sh"
export GBS="${GBS:-512}"
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/config_GBS${GBS}.sh"

export RECIPE="/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm_disagg_mxfp8.yaml"
export MLPERF_LINEAR_PRECISION="fp8"

export TRAIN_NODES=8
export GEN_NODES=8
export COLOCATED_GENERATION=0

export GYM_TRAIN_CONCURRENCY=${GYM_TRAIN_CONCURRENCY:-1024}
export GYM_EVAL_CONCURRENCY=${GYM_EVAL_CONCURRENCY:-256}

# Trainer parallelism; expert tensor parallelism is independent of dense TP.
export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-16}
export PP=${PP:-2}
export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=31

# Two TP8/EP8 context engines and one TP16/EP16 decode engine fill 32 GPUs.
export CTX_ENGINES=2
export GEN_ENGINES=1
export CTX_TP=8
export CTX_MTP=1
export CTX_MEP=8
export GEN_TP=16
export GEN_MTP=1
export GEN_MEP=16

# The top-level TRT-LLM configuration describes the widest engine.
export TRTLLM_TP=16
export TRTLLM_MEP=16
# Split the 251 validation prompts into waves of at most 126 prompts.
export VAL_BATCH_SIZE=126
export TRTLLM_GPU_UTIL="${TRTLLM_GPU_UTIL:-0.8}"

# Bound refit scratch space and preserve whole-node CPU scheduling.
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.01}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# Keep training EP groups within an NVLink domain. Training placement and
# Slurm allocation segments remain independently overridable.
export NEMORL_TRAIN_SEGMENT_SIZE="${NEMORL_TRAIN_SEGMENT_SIZE:-8}"
export SEGMENT="${SEGMENT:-8}"

export NRL_USE_FASTOKENS=1

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM="$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')"

export WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-120}
export WALLTIME=$((21 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))
add_grpo_eval_walltime
