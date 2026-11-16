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
    # `enroot remove`; the experiment tmpfs; node-local checkpoints if configured.
    local sweeps=1
    if [[ "${_GRPO_ISOLATED_EXPERIMENT:-0}" == 1 ]]; then
        sweeps=$((sweeps + 1))
    fi
    if [[ -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]]; then
        sweeps=$((sweeps + 1))
    fi
    grpo_teardown_reserve_seconds_for "${SLURM_JOB_NUM_NODES:-1}" "${sweeps}"
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

# Remove only this experiment's named container on every allocated node.
# Never sweep `enroot list`: other allocations and retained evidence are not ours.
cleanup_pyxis() {
    local cleanup_status
    [[ -n "${_cont_name:-}" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ &&
       "${_experiment_index:-}" =~ ^0*[1-9][0-9]*$ &&
       "${_cont_name}" == "${MODEL_NAME}_${SLURM_JOB_ID}_exp${_experiment_index}" ]] || return 1
    local cleanup_started="${SECONDS}"
    local -a cleanup_command=(
        srun --overlap --exact --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}"
        --kill-on-bad-exit=0 --ntasks-per-node=1 --cpus-per-task=1 /bin/bash -c
        'containers="$(enroot list)"
        list_status=$?
        [[ "${list_status}" -eq 0 ]] || exit "${list_status}"
        cleanup_status=0
        while IFS= read -r container; do
            # Pyxis uses pyxis_JOB_NAME when container_scope=job, and
            # pyxis_NAME for global scope. Match only this exact experiment.
            [[ "${container}" == "$1" || "${container}" == "pyxis_$1" ||
               "${container}" == "pyxis_${2}_$1" ]] || continue
            remove_status=0
            enroot remove -f "${container}" || remove_status=$?
            echo "GRPO_CONTAINER_REMOVAL: node=$(hostname) name=${container} status=${remove_status}"
            [[ "${remove_status}" -eq 0 ]] || cleanup_status=1
        done <<< "${containers}"
        remaining="$(enroot list)" || exit $?
        while IFS= read -r container; do
            if [[ "${container}" == "$1" || "${container}" == "pyxis_$1" ||
                  "${container}" == "pyxis_${2}_$1" ]]; then
                echo "GRPO_CONTAINER_REMAINS: node=$(hostname) name=${container}" >&2
                cleanup_status=1
            fi
        done <<< "${remaining}"
        echo "GRPO_CONTAINER_CLEANUP: node=$(hostname) name=$1 status=${cleanup_status}"
        exit "${cleanup_status}"' -- "${_cont_name}" "${SLURM_JOB_ID}"
    )

    if timeout --signal=TERM --kill-after=2s "$(_grpo_node_sweep_timeout)s" "${cleanup_command[@]}"; then
        cleanup_status=0
    else
        cleanup_status=$?
    fi
    _grpo_scratch_log "experiment Enroot cleanup finished in $((SECONDS - cleanup_started))s with status ${cleanup_status}"
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
    if ! cleanup_pyxis; then
        _grpo_scratch_log "WARNING: Enroot cleanup was incomplete"
    fi
    if [[ "${RETAIN_GRPO_SCRATCH}" == "1" ]]; then
        _grpo_scratch_log "retaining shared scratch (RETAIN_GRPO_SCRATCH=1)"
    fi
    if [[ "${RETAIN_GRPO_SCRATCH}" == "0" ]] && ! cleanup_grpo_scratch; then
        _grpo_scratch_log "WARNING: shared scratch cleanup failed"
    fi
    if [[ "${cleanup_checkpoints}" == "1" ]] && ! cleanup_grpo_checkpoints; then
        _grpo_scratch_log "WARNING: checkpoint cleanup failed"
    fi
    if [[ "${cleanup_checkpoints}" == "1" ]] && ! cleanup_grpo_local_checkpoints; then
        _grpo_scratch_log "WARNING: node-local checkpoint cleanup failed"
    fi
    if [[ "${_GRPO_ISOLATED_EXPERIMENT:-0}" == 1 ]] && ! cleanup_grpo_runtime_scratch; then
        _grpo_scratch_log "WARNING: experiment tmpfs cleanup failed"
    fi
    return 0
}

cleanup_containers() {
    local exit_status=$?

    # Teardown is best effort and must never replace the benchmark status.
    set +e
    trap - EXIT
    trap '' HUP INT TERM
    _run_grpo_teardown
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
