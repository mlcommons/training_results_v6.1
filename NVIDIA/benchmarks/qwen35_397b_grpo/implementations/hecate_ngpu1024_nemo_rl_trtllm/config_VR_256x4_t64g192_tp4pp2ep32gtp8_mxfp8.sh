# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

export NEMO_GYM_REPLICAS=${NEMO_GYM_REPLICAS:-4}

source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_recompute_selective_offload.sh"
export GBS="${GBS:-512}"
source "$(dirname "${BASH_SOURCE[0]}")/config_GBS${GBS}.sh"

export RECIPE="/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm_mxfp8.yaml"
export MLPERF_LINEAR_PRECISION="fp8"

export TRAIN_NODES=64
export GEN_NODES=192
export COLOCATED_GENERATION=0

export GYM_TRAIN_CONCURRENCY=${GYM_TRAIN_CONCURRENCY:-$((2 * NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}
export GYM_EVAL_CONCURRENCY=${GYM_EVAL_CONCURRENCY:-1024}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-null}

export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-32}
export PP=${PP:-2}

export TRTLLM_TP="${TRTLLM_TP:-8}"
export TRTLLM_MEP="${TRTLLM_MEP:-8}"
export TRTLLM_GPU_UTIL="${TRTLLM_GPU_UTIL:-0.8}"

export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.01}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-352}"

export NEMORL_TRAIN_SEGMENT_SIZE="${NEMORL_TRAIN_SEGMENT_SIZE:-8}"
export SEGMENT="${SEGMENT:-8}"

export NRL_USE_FASTOKENS=1

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM="$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')"

export WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-90}
export WALLTIME=$((21 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))
add_grpo_eval_walltime
