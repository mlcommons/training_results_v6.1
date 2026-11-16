#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

export RECIPE=${RECIPE:-qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_mxfp8.yaml}
export MEGATRON_GDN_BACKEND=${MEGATRON_GDN_BACKEND:-transformer_engine}
_QWEN35_MLPERF_TARGET_ACCURACY=${MLPERF_TARGET_ACCURACY:-0.69}
source "$(dirname "${BASH_SOURCE[0]}")/config_gym_shards.sh" || return
export RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-10}
export WALLTIME_OFFLINE_EVAL=${WALLTIME_OFFLINE_EVAL:-40}
source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_recompute_selective_offload.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_GBS512.sh"

export VLLM_USE_FASTOKENS=${VLLM_USE_FASTOKENS:-1}

unset NRL_RUNTIME_PATCH NRL_RUNTIME_PATCH_CONTAINER REPO_LOCATION
export NRL_SOURCE_OVERLAY=0

if [[ "${RECIPE}" == *_mxfp8.yaml || "${RECIPE}" == *_routed_mxfp8_*.yaml ]]; then
    export MLPERF_LINEAR_PRECISION="fp8"
fi

export TRAIN_NODES=64
export GEN_NODES=192
export COLOCATED_GENERATION=0

export GYM_EVAL_CONCURRENCY=${GYM_EVAL_CONCURRENCY:-1024}

export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-32}
export PP=${PP:-2}
export GTP=${GTP:-8}
export GPP=${GPP:-1}
export GEP=${GEP:-8}

export VLLM_SCHEDULER_CLS=${VLLM_SCHEDULER_CLS:-vllm_extensions.prefill_scheduler.CadencedPrefillAsyncScheduler}
export VLLM_PREFILL_SCHEDULE_INTERVAL=${VLLM_PREFILL_SCHEDULE_INTERVAL:-8}
export VLLM_COMPILATION_MODE=${VLLM_COMPILATION_MODE:-VLLM_COMPILE}
export VLLM_COMPILATION_BACKEND=${VLLM_COMPILATION_BACKEND:-inductor}
export VLLM_CUDAGRAPH_MODE=${VLLM_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}
export VLLM_CUDAGRAPH_CAPTURE_SIZES=${VLLM_CUDAGRAPH_CAPTURE_SIZES:-'[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]'}
export VLLM_MOE_BACKEND=${VLLM_MOE_BACKEND:-flashinfer_trtllm}
export NRL_REFIT_BUFFER_MEMORY_RATIO=${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.01}

export MLPERF_TARGET_ACCURACY=${_QWEN35_MLPERF_TARGET_ACCURACY}
unset _QWEN35_MLPERF_TARGET_ACCURACY

export NEMORL_TRAIN_SEGMENT_SIZE="${NEMORL_TRAIN_SEGMENT_SIZE:-8}"
export SEGMENT="${SEGMENT:-8}"

export RAY_CGRAPH_get_timeout=${RAY_CGRAPH_get_timeout:-1810}
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM="$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')"

export WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-90}
export WALLTIME=$((21 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
add_grpo_eval_walltime
