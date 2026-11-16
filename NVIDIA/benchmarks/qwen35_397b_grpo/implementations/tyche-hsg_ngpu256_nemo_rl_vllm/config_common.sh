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
        local sweeps=1 reserve
        [[ -z "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]] || sweeps=2
        if [[ ! "${DGXNNODES:-}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: DGXNNODES must be a positive integer before add_grpo_eval_walltime" >&2
            return 1
        fi
        grpo_validate_node_sweep_timeout || return 1
        reserve="$(grpo_teardown_reserve_minutes_for "${DGXNNODES}" "${sweeps}")"
        export WALLTIME=$((WALLTIME - 6 + ${NEXP:-1} * (WALLTIME_OFFLINE_EVAL + reserve) + 15 * (${NEXP:-1} - 1)))
    fi
}

export RECIPE=${RECIPE:-qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml}

export MLPERF_TARGET_ACCURACY=${MLPERF_TARGET_ACCURACY:-0.69}
export MLPERF_BENCHMARK_NAME=qwen35_397b_grpo

export MAX_TRAJECTORY_AGE_STEPS=${MAX_TRAJECTORY_AGE_STEPS:-4}
export NEMO_GYM_NUM_WORKERS=${NEMO_GYM_NUM_WORKERS:-4}

export RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-120}

export NCCL_TEST=${NCCL_TEST:-0}

export TORCH_DIST_USE_NCCL2=${TORCH_DIST_USE_NCCL2:-0}

export MLPERF_LINEAR_PRECISION="bfloat16"
export MLPERF_ATTN_PRECISION="bfloat16"
export MLPERF_COMM_PRECISION="bfloat16"

export NRL_VLLM_ATTENTION_BACKEND=${NRL_VLLM_ATTENTION_BACKEND:-FLASHINFER}
export NRL_VLLM_FUSE_ALLREDUCE_RMS=true
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl
