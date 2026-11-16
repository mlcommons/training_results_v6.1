#!/usr/bin/env bash
                                                                               
submit_without_allocation_env() {
    local key value
    local -a removals=()
    while IFS='=' read -r key value; do
        case "$key" in
            SLURM_CONF|SLURM_MPI_TYPE) ;;
            SLURM_*) removals+=(-u "$key");;
        esac
    done < <(env)
    env "${removals[@]}" "$@"
}

validate_submission_seeds() {
    [[ "${NEXP}" =~ ^([1-9]|10)$ ]] || { echo 'Expected one through ten full runs' >&2; return 1; }
    [[ "${DLRM_RUN_SEEDS:-}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Expected comma-separated seeds' >&2; return 1; }
    IFS=, read -r -a _submission_seeds <<< "${DLRM_RUN_SEEDS}"
    [[ ${#_submission_seeds[@]} -eq "$NEXP" ]] || { echo 'Seed count must match NEXP' >&2; return 1; }
    local seed
    local -A seen=()
    for seed in "${_submission_seeds[@]}"; do
        [[ "${seed}" =~ ^[1-9][0-9]{0,9}$ ]] && ((seed <= 2147483647)) || {
            echo 'Seed outside positive 31-bit range' >&2; return 1;
        }
        [[ -z "${seen[${seed}]:-}" ]] || { echo 'Duplicate submission seed' >&2; return 1; }
        seen[${seed}]=1
    done
}
