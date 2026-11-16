# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

export DGXNGPU=4

export MEGATRON_GDN_BACKEND=${MEGATRON_GDN_BACKEND:-transformer_engine}
export MCORE_GDN_FUSION=${MCORE_GDN_FUSION:-1}
export NVTE_CPU_OFFLOAD_V1=${NVTE_CPU_OFFLOAD_V1:-1}

export OFFLINE_EVAL=${OFFLINE_EVAL:-1}
export WALLTIME_OFFLINE_EVAL=${WALLTIME_OFFLINE_EVAL:-60}
if [[ "${OFFLINE_EVAL}" == "1" ]]; then
    if [[ ! "${NEXP-1}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: deferred evaluation requires NEXP to be a positive integer without leading zeroes" >&2
        return 1
    fi
    MLPERF_REQUEUE_FLAG=--no-requeue
fi

source "$(dirname "${BASH_SOURCE[0]}")/grpo_budget.sh"

add_grpo_eval_walltime() {
    if [[ "${OFFLINE_EVAL}" == "1" ]]; then
        local sweeps=2 reserve
        [[ -z "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]] || sweeps=3
        if [[ ! "${DGXNNODES:-}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: DGXNNODES must be a positive integer before add_grpo_eval_walltime" >&2
            return 1
        fi
        grpo_validate_node_sweep_timeout || return 1
        reserve="$(grpo_teardown_reserve_minutes_for "${DGXNNODES}" "${sweeps}")"
        export WALLTIME=$((WALLTIME - 6 + ${NEXP:-1} * (WALLTIME_OFFLINE_EVAL + reserve)))
    fi
}

export RECIPE=${RECIPE:-/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm.yaml}

export MLPERF_TARGET_ACCURACY=${MLPERF_TARGET_ACCURACY:-0.69}
export MLPERF_BENCHMARK_NAME=qwen35_397b_grpo

export MAX_TRAJECTORY_AGE_STEPS=${MAX_TRAJECTORY_AGE_STEPS:-4}
export NEMO_GYM_REPLICAS=${NEMO_GYM_REPLICAS:-1}
export NEMO_GYM_NUM_WORKERS=${NEMO_GYM_NUM_WORKERS:-4}

export TRTLLM_HYBRID_DISAGG_REUSE="${TRTLLM_HYBRID_DISAGG_REUSE:-1}"
export TLLM_MAMBA_FOLD_SAVE_LAST="${TLLM_MAMBA_FOLD_SAVE_LAST:-1}"
export TLLM_ADP_PACKED_SYNC="${TLLM_ADP_PACKED_SYNC:-1}"

export RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-120}

export NCCL_TEST=${NCCL_TEST:-0}

export TORCH_DIST_USE_NCCL2=${TORCH_DIST_USE_NCCL2:-0}

export MLPERF_LINEAR_PRECISION="bfloat16"
export MLPERF_ATTN_PRECISION="bfloat16"
export MLPERF_COMM_PRECISION="bfloat16"

export NRL_OFFLOAD_BEFORE_REFIT_GC=${NRL_OFFLOAD_BEFORE_REFIT_GC:-1}
export NRL_OFFLOAD_BEFORE_REFIT_EMPTY_CACHE=${NRL_OFFLOAD_BEFORE_REFIT_EMPTY_CACHE:-1}
