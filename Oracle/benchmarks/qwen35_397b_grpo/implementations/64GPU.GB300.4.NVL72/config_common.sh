# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

export DGXNGPU=4

# Shared policy-training kernels; unsupported inputs keep their stock dispatch.
export MEGATRON_GDN_BACKEND=${MEGATRON_GDN_BACKEND:-transformer_engine}
export MCORE_GDN_FUSION=${MCORE_GDN_FUSION:-1}
export NVTE_CPU_OFFLOAD_V1=${NVTE_CPU_OFFLOAD_V1:-1}

# Development profiles opt out before sourcing this file.
export OFFLINE_EVAL=${OFFLINE_EVAL:-1}
export WALLTIME_OFFLINE_EVAL=${WALLTIME_OFFLINE_EVAL:-60}
if [[ "${OFFLINE_EVAL}" == "1" ]]; then
    # Validate before the system profile uses NEXP in shell arithmetic. Reject
    # leading zeroes as well: Bash interprets those as octal, not decimal.
    if [[ ! "${NEXP-1}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: deferred evaluation requires NEXP to be a positive integer without leading zeroes" >&2
        return 1
    fi
    # launch-mlperf-benchmark.sh appends this after SLURM_EXTRA.
    MLPERF_REQUEUE_FLAG=--no-requeue
fi

# Allocation sizing shares its teardown-budget calculation with the batch script.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/grpo_budget.sh"

# Call after the profile sets its training/allocation budgets.
add_grpo_eval_walltime() {
    if [[ "${OFFLINE_EVAL}" == "1" ]]; then
        # Profiles budget 21 minutes per allocation: 15 for setup and the former
        # fixed six-minute teardown tail. Replace that tail with the node-scaled
        # reserve the deferred-launch check enforces (grpo_budget.sh), once per
        # experiment: every experiment tears down its own containers, tmpfs and
        # scratch. Sweeps: `enroot remove`, the experiment tmpfs, and node-local
        # checkpoints when GRPO_LOCAL_CHECKPOINT_DIR is set.
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

# Recipe path inside the container (conf/ is baked to /opt/nemo-rl/qwen_35/configs/)
export RECIPE=${RECIPE:-/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm.yaml}

export MLPERF_TARGET_ACCURACY=${MLPERF_TARGET_ACCURACY:-0.69}
export MLPERF_BENCHMARK_NAME=qwen35_397b_grpo

# Allow up to four policy weight-version steps of rollout lookahead/staleness.
export MAX_TRAJECTORY_AGE_STEPS=${MAX_TRAJECTORY_AGE_STEPS:-4}
# Keep one Gym stack by default; scale profiles can opt into any replica count.
export NEMO_GYM_REPLICAS=${NEMO_GYM_REPLICAS:-1}
# Default NeMo-Gym FastAPI worker processes per Gym replica for the model
# proxy: four, i.e. sixteen with the four replicas of the 64- to 256-node
# profiles and 16N disaggregated profiles. The 14N disaggregated and 16N
# aggregated profiles pin sixteen instead. An override also applies per replica.
export NEMO_GYM_NUM_WORKERS=${NEMO_GYM_NUM_WORKERS:-4}

export TRTLLM_HYBRID_DISAGG_REUSE="${TRTLLM_HYBRID_DISAGG_REUSE:-1}"
export TLLM_MAMBA_FOLD_SAVE_LAST="${TLLM_MAMBA_FOLD_SAVE_LAST:-1}"
export TLLM_ADP_PACKED_SYNC="${TLLM_ADP_PACKED_SYNC:-1}"

# Sync node-local /tmp/ray logs to the shared log dir (crash tracebacks
# survive job teardown)
export RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-120}

export NCCL_TEST=${NCCL_TEST:-0}

# Use the legacy PyTorch NCCL backend to avoid Ray worker initialization failures.
export TORCH_DIST_USE_NCCL2=${TORCH_DIST_USE_NCCL2:-0}

# Lowest numerical precision used, logged for MLPerf compliance.
export MLPERF_LINEAR_PRECISION="bfloat16"
export MLPERF_ATTN_PRECISION="bfloat16"
export MLPERF_COMM_PRECISION="bfloat16"

# Force gc.collect() and torch.cuda.empty_cache() inside offload_before_refit.
export NRL_OFFLOAD_BEFORE_REFIT_GC=${NRL_OFFLOAD_BEFORE_REFIT_GC:-1}
export NRL_OFFLOAD_BEFORE_REFIT_EMPTY_CACHE=${NRL_OFFLOAD_BEFORE_REFIT_EMPTY_CACHE:-1}
