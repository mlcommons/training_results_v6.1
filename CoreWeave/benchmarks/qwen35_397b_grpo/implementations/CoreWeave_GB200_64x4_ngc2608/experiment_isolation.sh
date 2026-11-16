# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sourced by run.sub. No source-time mutations. Each call runs in a fresh
# subshell so environment, status, mounts and upload assets cannot leak to EXP+1.
_grpo_fresh_directory() {
    local base="$1" kind="$2" candidate input input_real
    [[ -n "${base}" && "${base}" != *[:,]* ]] || return 1
    candidate="$(realpath -m -- "${base}/qwen35-${SLURM_JOB_ID}/${kind}-${_experiment_index}")" || return
    # A configurable output base must never cause even mkdir under model/data
    # inputs. Resolve before creating anything, including missing ancestors.
    for input in "${HF_CKPT_PATH:-}" "${NRL_MEGATRON_CHECKPOINT_DIR:-}" \
                 "${NEMO_GYM_SWE_TRAIN_DATA_PATH:-}" "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:-}" \
                 "${NEMO_GYM_SWE_SIF_DIR:-}"; do
        [[ -n "${input}" ]] || continue
        if [[ "${input}" == "${HF_CKPT_PATH:-}" && "${input}" == */snapshots/* ]]; then
            # config_mounts exposes the whole model cache for blob symlinks.
            input="$(dirname "$(dirname "${input}")")"
        fi
        input_real="$(realpath -m -- "${input}")" || return
        if [[ "${candidate}" == "${input_real}" || "${candidate}" == "${input_real}/"* ||
              "${input_real}" == "${candidate}/"* ]]; then
            echo "ERROR: experiment output overlaps initial input: ${candidate}" >&2
            return 1
        fi
    done
    ( umask 0002; mkdir -p -- "${base}" ) || return
    base="$(realpath -e -- "${base}")" || return
    local job_root="${base}/qwen35-${SLURM_JOB_ID}"
    [[ ! -L "${job_root}" ]] || return 1
    ( umask 0002; mkdir -p -- "${job_root}" ) || return
    _grpo_validate_scratch_target "${base}" "${job_root}" || return
    [[ "${candidate}" == "${job_root}/${kind}-${_experiment_index}" ]] || return 1
    # Exclusive creation, including EXP1. Never reopen an old run or a symlink.
    ( umask 0002; mkdir -- "${candidate}" ) || return
    _grpo_validate_scratch_target "${job_root}" "${candidate}" || return
    printf '%s\n' "${candidate}"
}

prepare_grpo_experiment_paths() {
    if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ||
          ! "${_experiment_index:-}" =~ ^0*[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid experiment identity" >&2
        return 1
    fi
    # Arbitrary writable mounts/source overlays can bypass every namespace.
    # Do not silently promise isolation for an unreviewed development layout.
    if [[ -n "${EXTRA_MOUNTS:-}" || "${NRL_SOURCE_OVERLAY:-0}" != 0 ]]; then
        echo "ERROR: fresh experiments require EXTRA_MOUNTS empty and NRL_SOURCE_OVERLAY=0" >&2
        return 1
    fi
    local checkpoint_base="${CHECKPOINTS_DIR:-}" hf_base="${HF_CACHE_DIR:-}"
    LOGDIR="$(_grpo_fresh_directory "${LOGDIR}" experiment)" || return
    if [[ -n "${checkpoint_base}" ]]; then
        CHECKPOINTS_DIR="$(_grpo_fresh_directory "${checkpoint_base}" checkpoint)" || return
    else
        CHECKPOINTS_DIR="${LOGDIR}/checkpoint"
    fi
    if [[ -n "${hf_base}" ]]; then
        HF_CACHE_DIR="$(_grpo_fresh_directory "${hf_base}" hf-cache)" || return
    else
        HF_CACHE_DIR="${LOGDIR}/hf_cache"
    fi
    export LOGDIR CHECKPOINTS_DIR HF_CACHE_DIR
    export _GRPO_ISOLATED_EXPERIMENT=1
    echo "GRPO_EXPERIMENT_ISOLATION: experiment=${_experiment_index} results=${LOGDIR} checkpoints=${CHECKPOINTS_DIR} hf_cache=${HF_CACHE_DIR}" >&2
}

_grpo_check_container_inventory() {
    local expected="$1"
    [[ "${expected}" == 0 || "${expected}" == 1 ]] || return 1
    # Require absence before creation and exactly one recognized name after.
    # An unsupported site naming mode must fail before Ray/training, rather
    # than turn into a silently successful no-op during final cleanup.
    timeout --signal=TERM --kill-after=2s "$(_grpo_node_sweep_timeout)s" \
        srun --overlap --exact --nodes="${SLURM_JOB_NUM_NODES}" \
        --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpus-per-task=1 \
        --kill-on-bad-exit=1 /bin/bash -c '
            containers="$(enroot list)" || exit $?
            matches=()
            while IFS= read -r name; do
                if [[ "$name" == "$1" || "$name" == "pyxis_$1" ||
                      "$name" == "pyxis_${2}_$1" ]]; then
                    matches+=("$name")
                fi
            done <<< "$containers"
            if [[ "$3" == 0 && "${#matches[@]}" != 0 ]]; then
                echo "ERROR: refusing existing experiment container: ${matches[*]}" >&2
                exit 1
            fi
            if [[ "$3" == 1 ]]; then
                if [[ "${#matches[@]}" != 1 ]]; then
                    echo "ERROR: expected one identifiable experiment container on $(hostname), found ${#matches[@]}: $1" >&2
                    exit 1
                fi
                echo "GRPO_CONTAINER_CREATED: node=$(hostname) name=${matches[0]} logical_name=$1 status=0"
            fi
        ' -- "${_cont_name}" "${SLURM_JOB_ID}" "${expected}"
}

require_fresh_grpo_container() { _grpo_check_container_inventory 0; }
require_created_grpo_container() { _grpo_check_container_inventory 1; }

_grpo_runtime_node() {
    local action="$1" job_id="$2" index="$3" owner="$4"
    _grpo_local_checkpoint_node "${action}" /dev/shm "${job_id}" "${owner}" \
        "${index}" "-exp${index}-runtime" || return
    if [[ "${action}" == prepare ]]; then
        mkdir -m 1777 -- "/dev/shm/qwen35-${job_id}-exp${index}-runtime/tmp" \
            "/dev/shm/qwen35-${job_id}-exp${index}-runtime/shm" || return
    fi
}

_grpo_runtime_all_nodes() {
    local definitions
    definitions="$(declare -f _grpo_scratch_log _grpo_scratch_fail \
        _grpo_validate_no_mounts_at_or_below _grpo_validate_scratch_target \
        _grpo_local_checkpoint_node _grpo_runtime_node)"
    timeout --signal=TERM --kill-after=2s "$(_grpo_node_sweep_timeout)s" \
        srun --overlap --exact --nodes="${SLURM_JOB_NUM_NODES}" \
        --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpus-per-task=1 \
        --kill-on-bad-exit=0 /bin/bash -c \
        "${definitions}"$'\n''_grpo_runtime_node "$@"' -- \
        "$1" "${SLURM_JOB_ID}" "${_experiment_index}" "${LOGDIR}"
}

prepare_grpo_runtime_scratch() { _grpo_runtime_all_nodes prepare; }
cleanup_grpo_runtime_scratch() {
    [[ "${_GRPO_ISOLATED_EXPERIMENT:-0}" == 1 ]] || return 0
    [[ "${RETAIN_GRPO_SCRATCH:-0}" == 0 ]] || return 0
    _grpo_runtime_all_nodes cleanup
}
