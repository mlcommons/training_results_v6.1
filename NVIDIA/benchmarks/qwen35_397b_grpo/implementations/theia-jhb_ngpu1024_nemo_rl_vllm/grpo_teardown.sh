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

# Sourced by run.sub after cleanup_grpo_scratch.sh. Keep source-time behavior
# limited to function definitions.

initialize_grpo_teardown_settings() {
    : "${RETAIN_GRPO_SCRATCH:=0}"
    : "${RETAIN_GRPO_CHECKPOINTS:=0}"
    local flag
    for flag in RETAIN_GRPO_SCRATCH RETAIN_GRPO_CHECKPOINTS; do
        case "${!flag}" in
            0|1) ;;
            *) echo "ERROR: ${flag} must be 0 or 1" >&2; return 1 ;;
        esac
    done
    grpo_validate_node_sweep_timeout || return 1
}

# Seconds the deferred-launch check keeps free after the two phase budgets for
# the teardown that follows evaluation. The calculation is grpo_budget.sh's,
# the same one config_common.sh uses to size the allocation; only the sweep
# count is decided here from the runtime configuration.
_grpo_teardown_reserve_seconds() {
    # `enroot remove`; node-local checkpoints if configured.
    local sweeps=1
    if [[ -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]]; then
        sweeps=$((sweeps + 1))
    fi
    grpo_teardown_reserve_seconds_for "${SLURM_JOB_NUM_NODES:-1}" "${sweeps}"
}

validate_grpo_fresh_overrides() {
    local line argument key value normalized expect_layers=0
    local -a arguments=()
    if [[ -n "${EXTRA_MOUNTS:-}" || "${NRL_SOURCE_OVERLAY:-0}" != 0 ||
          -n "${NRL_RUNTIME_PATCH:-}" || -n "${NRL_RUNTIME_PATCH_CONTAINER:-}" ]]; then
        echo "ERROR: fresh experiments do not support extra mounts, source overlays, or runtime patches" >&2
        return 1
    fi
    # The driver parses --config and known proxy flags before applying Hydra
    # overrides. In fresh mode, permit ordinary scalar/list leaf tuning only:
    # path changes, replacing a parent dictionary, and interpolation can evade
    # the launcher's private mounts. Select immutable inputs via config_data.
    while IFS= read -r line; do
        [[ "${line}" =~ [^[:space:]] ]] || continue
        IFS=$' \t' read -r -a arguments <<< "${line}"
        for argument in "${arguments[@]}"; do
            if [[ "${expect_layers}" == 1 ]]; then
                [[ "${argument}" =~ ^[1-9][0-9]*$ ]] || {
                    echo "ERROR: invalid --proxy-num-layers in fresh EXTRA_ARGS" >&2; return 1;
                }
                expect_layers=0
                continue
            fi
            case "${argument}" in
                --random-init-policy) continue ;;
                --proxy-num-layers) expect_layers=1; continue ;;
                --proxy-num-layers=*)
                    [[ "${argument#*=}" =~ ^[1-9][0-9]*$ ]] && continue
                    ;;
                --*)
                    echo "ERROR: fresh EXTRA_ARGS cannot replace the driver configuration or add CLI options" >&2
                    return 1
                    ;;
            esac
            key="${argument%%=*}"
            value="${argument#*=}"
            if [[ "${argument}" != *=* ||
                  ! "${key}" =~ ^\+{0,2}[A-Za-z_][A-Za-z0-9_.]*$ ||
                  "${value}" == *'{'* || "${value}" == *'}'* ]]; then
                echo "ERROR: fresh EXTRA_ARGS supports only scalar/list leaf overrides without interpolation" >&2
                return 1
            fi
            key="${key#++}"
            key="${key#+}"
            normalized="$(printf '%s' "${key}" | tr '[:upper:]' '[:lower:]')"
            case "${normalized}" in
                logger|checkpointing|policy|policy.generation|policy.generation.vllm_cfg|policy.megatron_cfg|policy.megatron_cfg.checkpoint|data|data.train|data.validation|data.default|env|env.nemo_gym|grpo.deferred_evaluation)
                    echo "ERROR: fresh EXTRA_ARGS cannot replace protected configuration namespace ${key}" >&2
                    return 1
                    ;;
            esac
            case ".${normalized}." in
                *.path.*|*.paths.*|*.dir.*|*.dirs.*|*.directory.*|*.root.*|*.file.*|*.files.*|*_path.*|*_paths.*|*_dir.*|*_dirs.*|*_directory.*|*_root.*|*_file.*|*_files.*|*.resume.*|*.resume_*.*|*.restore.*|*.restore_*.*|*.checkpoint.load.*|*.checkpoint.save.*|*.checkpointing.load.*|*.checkpointing.save.*|*.pretrained_checkpoint.*|*.model.|*.model_name.|*.tokenizer.|*.tokenizer_name.|*.env_vars.*|*.env_variables.*|*.runtime_env.*)
                    echo "ERROR: fresh EXTRA_ARGS cannot redirect input/output/cache discovery via ${key}" >&2
                    return 1
                    ;;
            esac
        done
    done <<< "${EXTRA_ARGS:-}"
    if [[ "${expect_layers}" == 1 ]]; then
        echo "ERROR: missing --proxy-num-layers value in fresh EXTRA_ARGS" >&2
        return 1
    fi
}

# The batch shell owns no Ray steps or containers. It starts one complete copy
# of the frozen launcher per deferred experiment and waits for that child's
# EXIT cleanup before starting the next one. A child keeps the original NEXP
# for checkpoint/log names and budget checks, takes its SEED from this loop,
# and runs only its index.
run_grpo_experiments() {
    local job_info="$1" launcher="$2" index status expected
    if [[ "${OFFLINE_EVAL:-0}" != 1 ]]; then
        if [[ -n "${GRPO_SINGLE_EXPERIMENT_INDEX+x}" ||
              -n "${GRPO_EXPERIMENT_PARENT_PID+x}" ]]; then
            echo "ERROR: fresh experiment identity requires OFFLINE_EVAL=1" >&2
            return 1
        fi
        return 0
    fi
    validate_deferred_launch "${job_info}" || return "$?"
    validate_grpo_fresh_overrides || return "$?"
    case "${CHECKPOINT_DIR:-/checkpoint}" in
        /checkpoint|/checkpoint/) ;;
        *)
            echo "ERROR: fresh experiments require CHECKPOINT_DIR=/checkpoint; use CHECKPOINTS_DIR to select the host base" >&2
            return 1
            ;;
    esac
    if [[ -n "${GRPO_SINGLE_EXPERIMENT_INDEX+x}" ]]; then
        expected="${GRPO_SINGLE_EXPERIMENT_INDEX}"
        if [[ ! "${expected}" =~ ^0*[1-9][0-9]*$ || ${#expected} -gt ${#NEXP} ||
              ! "${GRPO_EXPERIMENT_PARENT_PID:-}" =~ ^[1-9][0-9]*$ ||
              "${GRPO_EXPERIMENT_PARENT_PID}" != "${PPID}" ]]; then
            echo "ERROR: invalid internal fresh experiment identity" >&2
            return 1
        fi
        if (( 10#${expected} > NEXP )) ||
           [[ "${expected}" != "$(printf "%0${#NEXP}d" "$((10#${expected}))")" ]]; then
            echo "ERROR: fresh experiment index must match the original NEXP numbering" >&2
            return 1
        fi
        export _experiment_index="${expected}"
        return 0
    fi
    if [[ -n "${GRPO_EXPERIMENT_PARENT_PID+x}" ]]; then
        echo "ERROR: unexpected internal fresh experiment parent identity" >&2
        return 1
    fi
    if [[ "${NEXP}" -gt 1 &&
          ( "${RETAIN_GRPO_SCRATCH}" != 0 || "${RETAIN_GRPO_CHECKPOINTS}" != 0 ) ]]; then
        echo "ERROR: fresh multi-experiment execution requires scratch and checkpoint retention disabled" >&2
        return 1
    fi
    # These defaults must be identical in every child rather than regenerated
    # from its new shell; CHECKPOINT_DIR deliberately remains the unindexed base.
    # SEED is the exception: this loop picks a new one for each child, since a
    # reseed inside a child would die with it.
    export DATESTAMP SEED NEXP
    for index in $(seq -w 1 "${NEXP}"); do
        echo "FRESH_EXPERIMENT_START index=${index} of=${NEXP} $(date +%s)"
        if GRPO_SINGLE_EXPERIMENT_INDEX="${index}" \
           GRPO_EXPERIMENT_PARENT_PID="$$" bash "${launcher}"; then
            echo "FRESH_EXPERIMENT_STOP index=${index} status=0 $(date +%s)"
            SEED=$RANDOM     # choose a new seed for the next experiment
        else
            status=$?
            echo "ERROR: experiment ${index} or its full cleanup failed with status ${status}; no later experiment will start" >&2
            exit "${status}"
        fi
    done
    exit 0
}

# Fresh deferred children never sweep a user's other named containers. Pyxis
# uses pyxis_NAME for global scope and pyxis_JOBID_NAME for job scope. Consider
# only those two exact aliases; ambiguity is a failure, never a reason to sweep.
grpo_experiment_container() {
    local operation="$1"
    if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ||
          ! "${GRPO_SINGLE_EXPERIMENT_INDEX:-}" =~ ^0*[1-9][0-9]*$ ||
          ! "${MODEL_NAME:-}" =~ ^[A-Za-z0-9_]+$ ||
          "${_cont_name:-}" != "${MODEL_NAME}_${SLURM_JOB_ID}_exp${GRPO_SINGLE_EXPERIMENT_INDEX}" ]]; then
        _grpo_scratch_log "REFUSED: invalid fresh experiment container identity"
        return 1
    fi
    timeout --signal=TERM --kill-after=2s "$(_grpo_node_sweep_timeout)s" \
        srun --overlap --exact --nodes="${SLURM_JOB_NUM_NODES}" \
             --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
             --cpus-per-task=1 --kill-on-bad-exit=0 /bin/bash -c '
        set -eu
        operation="$1" global_target="$2" job_target="$3"
        containers="$(enroot list)"
        found=0 target=""
        while IFS= read -r container; do
            if [[ "$container" == "$global_target" || "$container" == "$job_target" ]]; then
                found=$((found + 1))
                target="$container"
            fi
        done <<< "$containers"
        if [[ "$found" -gt 1 ]]; then
            echo "ERROR: ambiguous experiment container aliases on $(hostname): $global_target $job_target" >&2
            exit 1
        fi
        removed=none
        case "$operation" in
            absent)
                if [[ "$found" == 1 ]]; then
                    echo "ERROR: refusing reused experiment container $target on $(hostname)" >&2
                    exit 1
                fi
            ;;
            remove)
                if [[ "$found" == 1 ]]; then
                    enroot remove -f "$target"
                    removed="$target"
                fi
                containers="$(enroot list)"
                while IFS= read -r container; do
                    if [[ "$container" == "$global_target" || "$container" == "$job_target" ]]; then
                        echo "ERROR: experiment container remains after cleanup on $(hostname): $container" >&2
                        exit 1
                    fi
                done <<< "$containers"
                ;;
            *) exit 1 ;;
        esac
        printf "GRPO_EXPERIMENT_CONTAINER: node=%s global=%s job_scoped=%s removed=%s operation=%s status=absent\n" "$(hostname)" "$global_target" "$job_target" "$removed" "$operation"
    ' _ "${operation}" "pyxis_${_cont_name}" "pyxis_${SLURM_JOB_ID}_${_cont_name}"
}

validate_deferred_launch() {
    local job_info="$1" end_time reserve_seconds required_seconds
    if [[ ! "${NEXP:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: deferred evaluation requires NEXP to be a positive integer without leading zeroes" >&2
        return 1
    fi
    if [[ "${SLURM_RESTART_COUNT:-0}" != 0 || "${FORCE_SUCCESS_STATUS:-0}" != 0 ]]; then
        echo "ERROR: deferred evaluation requires a fresh job and FORCE_SUCCESS_STATUS=0" >&2
        return 1
    fi
    if [[ ! " ${job_info} " =~ [[:space:]]Requeue=0[[:space:]] ]]; then
        echo "ERROR: deferred evaluation requires sbatch --no-requeue" >&2
        return 1
    fi
    if [[ ! "${WALLTIME_RUNANDTIME}" =~ ^[1-9][0-9]*$ ||
          ! "${WALLTIME_OFFLINE_EVAL}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: deferred training and evaluation walltimes must be positive integer minutes" >&2
        return 1
    fi
    [[ ! " ${job_info} " =~ [[:space:]]TimeLimit=UNLIMITED[[:space:]] ]] || return 0
    end_time="$(sed -n 's/.*EndTime=\([^[:space:]]*\).*/\1/p' <<< "${job_info}")"
    # Check the next experiment, not NEXP again: this is rechecked before every
    # training phase, when earlier experiments have already used their budget.
    # Keep the node-scaled teardown reserve after both phase budgets.
    reserve_seconds="$(_grpo_teardown_reserve_seconds)"
    required_seconds=$((60 * (WALLTIME_RUNANDTIME + WALLTIME_OFFLINE_EVAL) + reserve_seconds))
    if [[ -z "${end_time}" || "${end_time}" == Unknown ]] ||
       ! end_time="$(date -d "${end_time}" +%s)" ||
       (( end_time - $(date +%s) < required_seconds )); then
        echo "ERROR: remaining Slurm time must cover training + deferred evaluation + ${reserve_seconds} s of teardown" >&2
        return 1
    fi
}

# Named Enroot containers are node-local and inaccessible after the allocation
# ends. Sweep every allocated node promptly and independently of shared-tree
# cleanup. The per-name loop avoids word splitting in container names.
cleanup_pyxis() {
    if [[ -n "${GRPO_SINGLE_EXPERIMENT_INDEX:-}" ]]; then
        grpo_experiment_container remove
        # Bash 5's bare return inside an EXIT trap uses the status from before
        # the trap, not the failing command. Forward cleanup status explicitly.
        return "$?"
    fi
    local cleanup_status
    local cleanup_started="${SECONDS}"
    local -a cleanup_command=(
        srun --overlap --kill-on-bad-exit=0 --ntasks-per-node=1 /bin/bash -c
        'containers="$(enroot list)"
        list_status=$?
        [[ "${list_status}" -eq 0 ]] || exit "${list_status}"
        cleanup_status=0
        while IFS= read -r container; do
            [[ -n "${container}" ]] || continue
            enroot remove -f "${container}" || cleanup_status=1
        done <<< "${containers}"
        exit "${cleanup_status}"'
    )

    if timeout --signal=TERM --kill-after=2s "$(_grpo_node_sweep_timeout)s" "${cleanup_command[@]}"; then
        cleanup_status=0
    else
        cleanup_status=$?
    fi
    _grpo_scratch_log "Enroot sweep finished in $((SECONDS - cleanup_started))s with status ${cleanup_status}"
    return "${cleanup_status}"
}

signal_ray_steps() {
    local pid

    # The ENDED marker asks Ray's sidecars to stop; TERM asks each local srun
    # launcher to tear down its remote step.
    if [[ -n "${RAY_CLUSTER_LOG_DIR:-}" ]]; then
        touch "${RAY_CLUSTER_LOG_DIR}/ENDED" || true
    fi
    for pid in "${SRUN_PIDS[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
}

wait_for_ray_steps() {
    local wait_started="${SECONDS}"
    local deadline=$((SECONDS + 15))
    local pid

    while [[ "${SECONDS}" -lt "${deadline}" ]]; do
        local any_running=0
        for pid in "${SRUN_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                any_running=1
                break
            fi
        done
        [[ "${any_running}" -eq 0 ]] && break
        sleep 1
    done
    for pid in "${SRUN_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            _grpo_scratch_log \
                "WARNING: Ray launcher drain finished in $((SECONDS - wait_started))s with status 1; srun pid ${pid} is still alive, retaining scratch and checkpoints"
            return 1
        fi
    done
    for pid in "${SRUN_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    _grpo_scratch_log "Ray launcher drain finished in $((SECONDS - wait_started))s with status 0"
}

_grpo_scontrol_output_is_quiescent() {
    local output="$1"
    local job_id="$2"
    local line step_id
    local -i saw_control_step=0

    while IFS= read -r line; do
        [[ -n "${line}" ]] || continue
        if [[ "${line}" != StepId=* ]]; then
            return 2
        fi
        step_id="${line#StepId=}"
        step_id="${step_id%% *}"
        case "${step_id}" in
            "${job_id}.batch"|"${job_id}.extern")
                saw_control_step=1
                ;;
            "${job_id}."*)
                return 1
                ;;
            *)
                return 2
                ;;
        esac
    done <<< "${output}"

    [[ "${saw_control_step}" -eq 1 ]] || return 2
}

scheduler_confirms_no_active_steps() {
    local job_id="${SLURM_JOB_ID:-}"
    local deadline="${1:-$((SECONDS + 60))}"
    local output remaining

    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        return 3
    fi
    remaining=$((deadline - SECONDS))
    [[ "${remaining}" -gt 0 ]] || return 2
    [[ "${remaining}" -le 30 ]] || remaining=30
    if ! output="$(timeout --signal=TERM --kill-after=2s "${remaining}s" \
        scontrol -o show step "${job_id}" 2>&1)"; then
        return 2
    fi
    # squeue may omit application steps; it cannot establish quiescence when
    # this exact-job query is unavailable or ambiguous.
    _grpo_scontrol_output_is_quiescent "${output}" "${job_id}"
}

wait_for_slurm_steps() {
    local wait_started="${SECONDS}"
    local deadline=$((SECONDS + 60))
    local status=2

    while [[ "${SECONDS}" -lt "${deadline}" ]]; do
        if scheduler_confirms_no_active_steps "${deadline}"; then
            _grpo_scratch_log "Slurm step drain finished in $((SECONDS - wait_started))s with status 0"
            return 0
        else
            status=$?
        fi
        if [[ "${status}" -eq 3 ]]; then
            _grpo_scratch_log \
                "WARNING: Slurm step drain finished in $((SECONDS - wait_started))s with status 3; invalid job ID, retaining scratch and checkpoints"
            return 1
        fi
        [[ "${SECONDS}" -lt "${deadline}" ]] || break
        sleep 1
    done
    if [[ "${status}" -eq 1 ]]; then
        _grpo_scratch_log \
            "WARNING: Slurm step drain finished in $((SECONDS - wait_started))s with status 1; scheduler still reports active job steps, retaining scratch and checkpoints"
    else
        _grpo_scratch_log \
            "WARNING: Slurm step drain finished in $((SECONDS - wait_started))s with status 2; scheduler did not confirm quiescence before the deadline, retaining scratch and checkpoints"
    fi
    return 1
}

_run_grpo_teardown() {
    local cleanup_checkpoints=0
    if [[ "${OFFLINE_EVAL:-0}" == "1" && "${RETAIN_GRPO_CHECKPOINTS:-0}" == "0" ]]; then
        cleanup_checkpoints=1
    fi
    signal_ray_steps

    # Do this first: unlike the shared trees, node-local containers cannot be
    # cleaned from a login node after the allocation disappears. Like every
    # cleanup phase it only warns on failure; leftovers are logged per node.
    if ! cleanup_pyxis; then
        _grpo_scratch_log "WARNING: Enroot cleanup was incomplete"
    fi

    if [[ "${RETAIN_GRPO_SCRATCH}" == "1" ]]; then
        _grpo_scratch_log "retaining shared scratch (RETAIN_GRPO_SCRATCH=1)"
        [[ "${cleanup_checkpoints}" == "1" ]] || return 0
    fi
    # The two drains are the liveness guard: nothing below may run while a Ray
    # or application step is alive, so their failure stays fatal and stops the
    # experiment sequence.
    if ! wait_for_ray_steps; then
        return 1
    fi
    # A local srun can exit before Slurm has retired its remote step. Require
    # the scheduler/step manager to report only batch/extern before rm.
    if ! wait_for_slurm_steps; then
        return 1
    fi
    # Everything below is cleanup of resources nothing uses any more. Each
    # phase runs regardless of the others and only warns on failure: a
    # timed-out 256-node `enroot remove` once aborted the teardown here and
    # left both deferred checkpoints (1.5 TB) and the Gym scratch on shared
    # storage. Leftovers are reported per node or path for manual removal.
    # One exception: a fresh child whose successor still has to start. The
    # successor requires an absent Ray job directory (run.sub) and an absent
    # node-local checkpoint job directory on every node (exclusive mkdir at
    # prepare), job-scoped names it shares with this experiment, so such a
    # leftover fails the boundary here instead of the next experiment's setup.
    local successor_pending=0 blocking_leftover=0
    if [[ "${GRPO_SINGLE_EXPERIMENT_INDEX:-}" =~ ^0*[1-9][0-9]*$ &&
          "${NEXP:-}" =~ ^[1-9][0-9]*$ ]] &&
       (( 10#${GRPO_SINGLE_EXPERIMENT_INDEX} < NEXP )); then
        successor_pending=1
    fi
    if [[ "${RETAIN_GRPO_SCRATCH}" == "0" ]] && ! cleanup_grpo_scratch; then
        _grpo_scratch_log "WARNING: shared scratch cleanup failed"
        if [[ "${successor_pending}" == 1 && -n "${RAY_CLUSTER_LOG_DIR:-}" ]] &&
           [[ -e "${RAY_CLUSTER_LOG_DIR}" || -L "${RAY_CLUSTER_LOG_DIR}" ]]; then
            _grpo_scratch_log "ERROR: the next fresh experiment requires an absent Ray job directory: ${RAY_CLUSTER_LOG_DIR}"
            blocking_leftover=1
        fi
    fi
    if [[ "${cleanup_checkpoints}" == "1" ]] && ! cleanup_grpo_checkpoints; then
        _grpo_scratch_log "WARNING: checkpoint cleanup failed"
    fi
    if [[ "${cleanup_checkpoints}" == "1" ]] && ! cleanup_grpo_local_checkpoints; then
        _grpo_scratch_log "WARNING: node-local checkpoint cleanup failed"
        if [[ "${successor_pending}" == 1 && -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]]; then
            _grpo_scratch_log "ERROR: the next fresh experiment requires an absent node-local checkpoint directory on every node: ${GRPO_LOCAL_CHECKPOINT_DIR}/qwen35-${SLURM_JOB_ID}"
            blocking_leftover=1
        fi
    fi
    return "${blocking_leftover}"
}

cleanup_containers() {
    local exit_status=$? cleanup_status

    # Preserve the original phase failure. Teardown returns non-zero only when
    # a drain cannot confirm that the Ray and application steps are gone, or
    # when a fresh child leaves a job-scoped directory its successor requires
    # absent; a fresh deferred child must then fail even after a successful
    # phase so the orchestration parent never continues into the next
    # experiment. The other cleanup phases only warn.
    set +e
    trap - EXIT
    trap '' HUP INT TERM
    _run_grpo_teardown
    cleanup_status=$?
    if [[ -n "${GRPO_SINGLE_EXPERIMENT_INDEX:-}" &&
          "${exit_status}" == 0 && "${cleanup_status}" != 0 ]]; then
        _grpo_scratch_log "ERROR: fresh experiment boundary cleanup failed with status ${cleanup_status}"
        exit "${cleanup_status}"
    fi
    return "${exit_status}"
}

cleanup_for_requeue() {
    local restore_errexit=0

    if [[ "$-" == *e* ]]; then
        restore_errexit=1
        set +e
    fi
    _run_grpo_teardown
    if [[ "${restore_errexit}" -eq 1 ]]; then
        set -e
    fi
}
