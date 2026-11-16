#!/usr/bin/env bash
set -euo pipefail
(($# == 0)) || { echo 'No arguments expected' >&2; exit 2; }
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
RUN_ROOT=${RUN_ROOT:-${SCRIPT_DIR}/results/three_runs_$(date +%Y%m%d_%H%M%S)}
if [[ "${QWEN_DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$(dirname "${RUN_ROOT}")"
    mkdir "${RUN_ROOT}"
fi
previous=
for seed in 43 44 45; do
    export SEED=${seed} LOGDIR="${RUN_ROOT}/seed_${seed}"
    export SBATCH_DEPENDENCY="${previous:+afterok:${previous}}"
    bash "${SCRIPT_DIR}/run.sh"
    if [[ "${QWEN_DRY_RUN:-0}" == 1 ]]; then continue; fi
    job=$(awk '/^Submitted batch job [0-9]+$/ {print $4}' "${LOGDIR}/submission.txt")
    [[ "${job}" =~ ^[0-9]+$ ]]
    printf 'seed=%s job=%s logdir=%s\n' "${seed}" "${job}" "${LOGDIR}" | tee -a "${RUN_ROOT}/run_plan.txt"
    previous=${job}
done
