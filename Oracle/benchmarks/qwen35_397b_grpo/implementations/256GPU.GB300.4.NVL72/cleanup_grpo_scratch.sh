#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is sourced by run.sub. Deletion targets are always derived here
# from a canonical experiment root and numeric Slurm identifiers; callers never
# supply a path to remove.

_grpo_scratch_log() {
    # Cleanup must not change the benchmark result, even if LOGDIR is full and
    # writing the batch stderr fails.
    echo "GRPO_SCRATCH_CLEANUP: $*" >&2 || true
}

_grpo_scratch_fail() {
    _grpo_scratch_log "REFUSED: $*"
    return 1
}

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/grpo_budget.sh"

# Deadline for one srun sweep over every node of this allocation; the
# calculation and its rationale live in grpo_budget.sh, shared with the
# allocation sizing in config_common.sh.
_grpo_node_sweep_timeout() {
    grpo_node_sweep_timeout_for "${SLURM_JOB_NUM_NODES:-1}"
}

_grpo_validate_no_mounts_at_or_below() {
    local target="$1"
    local mountinfo_file="${2:-/proc/self/mountinfo}"
    local mountinfo mount_id parent_id major_minor mount_root mountpoint remainder
    local encoded_without_known_escapes
    local -i saw_record=0

    # Device-number checks cannot distinguish a same-filesystem bind mount.
    # Read the current mount namespace directly and reject a mount at or below
    # the target. Linux mountinfo escapes space, tab, newline, and backslash as
    # \040, \011, \012, and \134 respectively.
    if [[ ! -r "${mountinfo_file}" ]]; then
        _grpo_scratch_fail "cannot read mount table: ${mountinfo_file}"
        return 1
    fi
    if ! mountinfo="$(< "${mountinfo_file}")" || [[ -z "${mountinfo}" ]]; then
        _grpo_scratch_fail "cannot read mount table: ${mountinfo_file}"
        return 1
    fi

    while IFS=' ' read -r mount_id parent_id major_minor mount_root mountpoint remainder; do
        saw_record=1
        if [[ -z "${mount_id}" || -z "${parent_id}" || -z "${major_minor}" ||
              -z "${mount_root}" || -z "${mountpoint}" ||
              "${remainder}" != *" - "* ]]; then
            _grpo_scratch_fail "malformed mount table: ${mountinfo_file}"
            return 1
        fi

        encoded_without_known_escapes="${mountpoint//\\040/}"
        encoded_without_known_escapes="${encoded_without_known_escapes//\\011/}"
        encoded_without_known_escapes="${encoded_without_known_escapes//\\012/}"
        encoded_without_known_escapes="${encoded_without_known_escapes//\\134/}"
        if [[ "${encoded_without_known_escapes}" == *\\* ]]; then
            _grpo_scratch_fail "unknown mount-table escape in: ${mountpoint}"
            return 1
        fi

        # Decode backslash last so a literal `\040` path component is not
        # mistaken for a space escape.
        mountpoint="${mountpoint//\\040/ }"
        mountpoint="${mountpoint//\\011/$'\t'}"
        mountpoint="${mountpoint//\\012/$'\n'}"
        mountpoint="${mountpoint//\\134/\\}"
        if [[ "${mountpoint}" == "${target}" ||
              "${mountpoint}" == "${target}/"* ]]; then
            _grpo_scratch_fail "target is or contains a mountpoint: ${target}"
            return 1
        fi
    done <<< "${mountinfo}"

    if [[ "${saw_record}" -eq 0 ]]; then
        _grpo_scratch_fail "empty mount table: ${mountinfo_file}"
        return 1
    fi
}

_grpo_validate_scratch_target() {
    local experiment_root="$1"
    local target="$2"
    local target_real

    if [[ -L "${target}" || ! -d "${target}" ]]; then
        _grpo_scratch_fail "target is not a real directory: ${target}"
        return 1
    fi
    if ! target_real="$(realpath -e -- "${target}")"; then
        _grpo_scratch_fail "cannot resolve target: ${target}"
        return 1
    fi

    # Equality also rejects a symlink in an intermediate component. The prefix
    # check is the final invariant: every resolved deletion target is a strict
    # descendant of the experiment root.
    if [[ "${target_real}" != "${target}" ||
          "${target_real}" != "${experiment_root}/"* ]]; then
        _grpo_scratch_fail "target escapes or is not canonical: ${target} -> ${target_real}"
        return 1
    fi
    _grpo_validate_no_mounts_at_or_below "${target_real}" || return
}

cleanup_grpo_scratch() {
    local experiment_root_input="${LOGDIR:-}"
    local job_id="${SLURM_JOB_ID:-}"
    local restart_count="${SLURM_RESTART_COUNT:-0}"
    local experiment_root gym_target ray_target target
    local -i attempt restart_count_decimal deleted_count=0
    local -a targets=()

    case "${RETAIN_GRPO_SCRATCH:-0}" in
        0) ;;
        1)
            _grpo_scratch_log "retaining NeMo-Gym and Ray scratch (RETAIN_GRPO_SCRATCH=1)"
            return 0
            ;;
        *)
            _grpo_scratch_fail "RETAIN_GRPO_SCRATCH must be 0 or 1"
            return 1
            ;;
    esac
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        _grpo_scratch_fail "SLURM_JOB_ID must be numeric"
        return 1
    fi
    if [[ ! "${restart_count}" =~ ^[0-9]+$ ]]; then
        _grpo_scratch_fail "SLURM_RESTART_COUNT must be numeric"
        return 1
    fi
    restart_count_decimal=$((10#${restart_count}))
    if [[ -z "${experiment_root_input}" ]]; then
        _grpo_scratch_fail "LOGDIR is unset"
        return 1
    fi
    if ! experiment_root="$(realpath -e -- "${experiment_root_input}")"; then
        _grpo_scratch_fail "cannot resolve experiment root: ${experiment_root_input}"
        return 1
    fi
    if [[ ! -d "${experiment_root_input}" || -L "${experiment_root_input}" ||
          "${experiment_root_input}" != "${experiment_root}" ||
          "${experiment_root}" == "/" ]]; then
        _grpo_scratch_fail "LOGDIR must be a canonical, real, non-root directory"
        return 1
    fi

    gym_target="${experiment_root}/nemo_logs_${job_id}/nemo_gym"
    if [[ -e "${gym_target}" || -L "${gym_target}" ]]; then
        targets+=("${gym_target}")
    fi
    for ((attempt = 0; attempt <= restart_count_decimal; attempt++)); do
        ray_target="${experiment_root}/ray-${job_id}.${attempt}"
        # The Lustre copy of every node's Ray session logs (rsynced by the
        # log-sync sidecar) is scratch like the Gym results: it follows
        # RETAIN_GRPO_SCRATCH. Live Ray state is in the private node tmpfs,
        # retired separately after process drain.
        if [[ -e "${ray_target}" || -L "${ray_target}" ]]; then
            targets+=("${ray_target}")
        fi
    done

    # Validate the complete, internally derived set once before unlinking any
    # target. This keeps path escape failures atomic without a metadata/state
    # machine around ordinary job-owned scratch.
    for target in "${targets[@]}"; do
        _grpo_validate_scratch_target "${experiment_root}" "${target}" || return
    done

    for target in "${targets[@]}"; do
        _grpo_scratch_log "deleting ${target}"
        if ! rm -rf --one-file-system --preserve-root=all -- "${target}"; then
            _grpo_scratch_fail "deletion failed: ${target}"
            return 1
        fi
        deleted_count=$((deleted_count + 1))
    done
    _grpo_scratch_log "deleted ${deleted_count} job-scoped target(s) under ${experiment_root}"
}

# Called only after the training/evaluation writers have stopped. Keep the
# checkpoint parent, logs and initial model inputs; remove completed and partial
# step directories and their latest-checkpoint status, including a final endpoint
# skipped after an earlier pass.
cleanup_grpo_checkpoints() {
    local checkpoint_root="${_checkpoint_dir:-}"
    local checkpoint_base directory target input input_real
    local -a targets=()

    case "${RETAIN_GRPO_CHECKPOINTS:-0}" in
        1) _grpo_scratch_log "retaining checkpoints (RETAIN_GRPO_CHECKPOINTS=1)"; return 0 ;;
        0) ;;
        *) _grpo_scratch_fail "RETAIN_GRPO_CHECKPOINTS must be 0 or 1"; return 1 ;;
    esac
    # No training trial was entered (for example a failed container prolog).
    [[ -n "${_experiment_index:-}" ]] || return 0
    if [[ ! "${_experiment_index}" =~ ^[0-9]+$ || -z "${LOGDIR:-}" ||
          ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
        _grpo_scratch_fail "invalid checkpoint experiment identity"
        return 1
    fi
    # The EXIT trap can run before config_mounts.sh has initialized the mount.
    # Never fall back to the shared base or accept another allocation's root.
    if [[ -z "${checkpoint_root}" ]] ||
       ! checkpoint_base="$(realpath -e -- "${CHECKPOINTS_DIR:-${LOGDIR}/checkpoint}")" ||
       [[ "${checkpoint_root}" != "${checkpoint_base}/qwen35-${SLURM_JOB_ID}" ]]; then
        _grpo_scratch_fail "checkpoint output root is not initialized for this Slurm job"
        return 1
    fi
    _grpo_validate_scratch_target "${checkpoint_base}" "${checkpoint_root}" || return

    if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
        # CHECKPOINT_DIR is container-side; translate only the /checkpoint mount.
        case "${CHECKPOINT_DIR}" in
            /checkpoint|/checkpoint/*) directory="${checkpoint_root}${CHECKPOINT_DIR#/checkpoint}" ;;
            *) _grpo_scratch_fail "CHECKPOINT_DIR is outside the /checkpoint mount"; return 1 ;;
        esac
    else
        directory="${checkpoint_root}/${_experiment_index}"
    fi
    while [[ "${directory}" == */ ]]; do directory="${directory%/}"; done
    for target in "${directory}"/step_* "${directory}"/tmp_step_*; do
        [[ "${target##*/}" =~ ^(tmp_)?step_[0-9]+$ ]] || continue
        [[ -e "${target}" || -L "${target}" ]] || continue
        _grpo_validate_scratch_target "${checkpoint_root}" "${target}" || return
        targets+=("${target}")
    done
    target="${directory}/latest_checkpoint_status.json"
    if [[ -e "${target}" || -L "${target}" ]]; then
        # A failed/previous cleanup may leave only the status file. Validate its
        # parent even when there are no step directories to trigger the guards.
        if [[ "${directory}" != "${checkpoint_root}" ]]; then
            _grpo_validate_scratch_target "${checkpoint_root}" "${directory}" || return
        fi
        if [[ -L "${target}" || ! -f "${target}" ]]; then
            _grpo_scratch_fail "checkpoint status is not a real file: ${target}"
            return 1
        fi
        _grpo_validate_no_mounts_at_or_below "${target}" || return
        targets+=("${target}")
    fi
    for input in "${HF_CKPT_PATH:-}" "${NRL_MEGATRON_CHECKPOINT_DIR:-}"; do
        [[ -n "${input}" ]] || continue
        input_real="$(realpath -e -- "${input}")" || return
        for target in "${targets[@]}"; do
            if [[ "${target}" == "${input_real}" || "${target}" == "${input_real}/"* ||
                  "${input_real}" == "${target}/"* ]]; then
                _grpo_scratch_fail "checkpoint target overlaps initial model input: ${target}"
                return 1
            fi
        done
    done
    for target in "${targets[@]}"; do
        _grpo_scratch_log "deleting checkpoint ${target}"
        rm -rf --one-file-system --preserve-root=all -- "${target}" || return
        if [[ -e "${target}" || -L "${target}" ]]; then
            _grpo_scratch_fail "checkpoint deletion incomplete: ${target}"
            return 1
        fi
    done
    _grpo_scratch_log "deleted ${#targets[@]} emitted checkpoint target(s) under ${checkpoint_root}"
}

# One private directory per allocation, outside the Enroot writable layer.
# The marker lets teardown clean a partially provisioned allocation without
# touching a pre-existing directory that made exclusive creation fail.
_grpo_local_checkpoint_node() {
    local action="$1" parent="$2" job_id="$3" experiment="$4"
    local experiment_index="${5:-}" namespace="${6:-}"
    local target node
    case "${action}" in
        prepare|cleanup) ;;
        cleanup-experiment)
            if [[ ! "${experiment_index}" =~ ^0*[1-9][0-9]*$ ]]; then
                _grpo_scratch_fail "node-local checkpoint experiment index must be a positive integer"
                return 1
            fi
            ;;
        *) _grpo_scratch_fail "invalid node-local checkpoint action"; return 1 ;;
    esac
    # Resolve the selected disk on this compute node, never the login node.
    if [[ ! "${job_id}" =~ ^[0-9]+$ || "${parent}" != /* ]] ||
       ! parent="$(realpath -e -- "${parent}")" ||
       [[ ! -d "${parent}" || "${parent}" == / ]]; then
        _grpo_scratch_fail "invalid node-local checkpoint parent or job ID"
        return 1
    fi
    target="${parent}/qwen35-${job_id}"
    if [[ -n "${namespace}" ]]; then
        [[ "${experiment_index}" =~ ^0*[1-9][0-9]*$ ]] || return 1
        case "${namespace}" in
            "-exp${experiment_index}"|"-exp${experiment_index}-runtime") ;;
            *) _grpo_scratch_fail "invalid experiment namespace"; return 1 ;;
        esac
        target+="${namespace}"
    fi
    if [[ "${action}" == prepare ]]; then
        ( umask 0002; mkdir -- "${target}" ) || return
        if ! printf '%s\n' "${experiment}" > "${target}/.grpo-experiment"; then
            rmdir -- "${target}" || true
            return 1
        fi
        _grpo_scratch_log "prepared node-local checkpoints: ${target}"
        return 0
    fi
    node="${SLURMD_NODENAME:-$(hostname)}" || return
    if [[ ! -e "${target}" && ! -L "${target}" ]]; then
        if [[ "${action}" == cleanup-experiment ]]; then
            _grpo_scratch_fail "node-local checkpoint job directory disappeared before the next experiment: ${target}"
            return 1
        fi
        _grpo_scratch_log "node-local checkpoints absent: node=${node} target=${target} status=already_absent"
        return 0
    fi
    if [[ ! -f "${target}/.grpo-experiment" || -L "${target}/.grpo-experiment" ||
          "$(< "${target}/.grpo-experiment")" != "${experiment}" ]]; then
        _grpo_scratch_fail "node-local checkpoint directory belongs to another experiment: ${target}"
        return 1
    fi
    _grpo_validate_scratch_target "${parent}" "${target}" || return
    if [[ "${action}" == cleanup-experiment ]]; then
        # The allocation-wide directory is mounted by live Ray containers.
        # Retire only this completed EXP; keep the mount and ownership marker
        # available to the next driver in the allocation.
        parent="${target}"
        target="${parent}/${experiment_index}"
        if [[ ! -e "${target}" && ! -L "${target}" ]]; then
            _grpo_scratch_log "node-local experiment checkpoints absent: node=${node} experiment=${experiment_index} target=${target} status=already_absent"
            return 0
        fi
        _grpo_validate_scratch_target "${parent}" "${target}" || return
        _grpo_scratch_log "deleting node-local experiment checkpoints: ${target}"
        rm -rf --one-file-system --preserve-root=all -- "${target}" || return
        [[ ! -e "${target}" && ! -L "${target}" ]] || return 1
        _grpo_scratch_log "node-local experiment checkpoints absent: node=${node} experiment=${experiment_index} target=${target} status=deleted"
        return 0
    fi
    _grpo_scratch_log "deleting node-local checkpoints: ${target}"
    rm -rf --one-file-system --preserve-root=all -- "${target}" || return
    [[ ! -e "${target}" && ! -L "${target}" ]] || return 1
    _grpo_scratch_log "node-local checkpoints absent: node=${node} target=${target} status=deleted"
}

_grpo_local_checkpoint_all_nodes() {
    local definitions namespace=""
    if [[ "${_GRPO_ISOLATED_EXPERIMENT:-0}" == 1 ]]; then
        namespace="-exp${_experiment_index}"
    fi
    # Send the small helper rather than assuming the batch script's directory
    # is mounted on compute nodes. Cleanup runs outside all containers. Like
    # every whole-allocation sweep, this one runs under the node-scaled
    # deadline so a wedged node cannot stall the remaining teardown phases.
    definitions="$(declare -f _grpo_scratch_log _grpo_scratch_fail \
        _grpo_validate_no_mounts_at_or_below _grpo_validate_scratch_target \
        _grpo_local_checkpoint_node)"
    timeout --signal=TERM --kill-after=2s "$(_grpo_node_sweep_timeout)s" \
        srun --overlap --kill-on-bad-exit=0 --ntasks-per-node=1 /bin/bash -c \
        "${definitions}"$'\n''_grpo_local_checkpoint_node "$@"' -- \
        "$1" "${GRPO_LOCAL_CHECKPOINT_DIR}" "${SLURM_JOB_ID}" "${LOGDIR}" "${_experiment_index:-}" "${namespace}"
}

prepare_grpo_local_checkpoints() {
    if [[ "${OFFLINE_EVAL:-0}" != 1 || ! "${NEXP:-1}" =~ ^[1-9][0-9]*$ ]]; then
        _grpo_scratch_fail "node-local checkpoints require OFFLINE_EVAL=1 and a positive integer NEXP"
        return 1
    fi
    # Colons and commas delimit Pyxis mounts; reject them before creating files.
    if [[ "${GRPO_LOCAL_CHECKPOINT_DIR}" == *[:,]* ]]; then
        _grpo_scratch_fail "node-local checkpoint parent contains a mount delimiter"
        return 1
    fi
    _grpo_local_checkpoint_all_nodes prepare
}

cleanup_grpo_local_checkpoints() {
    [[ -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]] || return 0
    case "${RETAIN_GRPO_CHECKPOINTS:-0}" in
        1) _grpo_scratch_log "retaining node-local checkpoints (RETAIN_GRPO_CHECKPOINTS=1)"; return 0 ;;
        0) ;;
        *) _grpo_scratch_fail "RETAIN_GRPO_CHECKPOINTS must be 0 or 1"; return 1 ;;
    esac
    _grpo_local_checkpoint_all_nodes cleanup
}

# The caller must first confirm that this EXP's training/evaluation actors and
# checkpoint writers exited. Ray daemons and job-wide scratch stay alive; only
# the guarded allocation EXIT path may retire those resources.
cleanup_grpo_experiment_checkpoints() {
    case "${RETAIN_GRPO_CHECKPOINTS:-0}" in
        1) _grpo_scratch_log "retaining experiment checkpoints (RETAIN_GRPO_CHECKPOINTS=1)"; return 0 ;;
        0) ;;
        *) _grpo_scratch_fail "RETAIN_GRPO_CHECKPOINTS must be 0 or 1"; return 1 ;;
    esac
    [[ -n "${_experiment_index:-}" ]] || return 0
    if [[ ! "${_experiment_index}" =~ ^0*[1-9][0-9]*$ ]]; then
        _grpo_scratch_fail "checkpoint experiment index must be a positive integer"
        return 1
    fi
    cleanup_grpo_checkpoints || return
    [[ -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]] || return 0
    _grpo_local_checkpoint_all_nodes cleanup-experiment
}
