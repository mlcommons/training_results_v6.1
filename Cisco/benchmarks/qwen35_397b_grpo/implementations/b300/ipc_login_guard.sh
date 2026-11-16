#!/usr/bin/env bash
                                                                         
                                                                               
                                                                               

qwen_start_ipc_login_guards() {
    [[ "${QWEN_IPC_LOGIN_GUARD:-0}" == 1 ]] || return 0
    [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || return 1
    local node fd pid log deadline
    local -a nodes
    mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
    ((${#nodes[@]} > 0)) || return 1
    _QWEN_IPC_GUARD_PIDS=()
    _QWEN_IPC_GUARD_FDS=()
    _QWEN_IPC_GUARD_FILES=()
    _QWEN_IPC_GUARD_LOGS=()
    _QWEN_IPC_GUARD_DIR=$(mktemp -d "/tmp/qwen35-ipc-guard-${SLURM_JOB_ID}.XXXXXX")
    for node in "${nodes[@]}"; do
        [[ "${node}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || return 1
        log="${_QWEN_IPC_GUARD_DIR}/${node}.log"
        mkfifo "${_QWEN_IPC_GUARD_DIR}/${node}.stdin"
        _QWEN_IPC_GUARD_FILES+=("${_QWEN_IPC_GUARD_DIR}/${node}.stdin" "${log}")
                                                                            
        exec {fd}<>"${_QWEN_IPC_GUARD_DIR}/${node}.stdin"
        _QWEN_IPC_GUARD_FDS+=("${fd}")
        ssh -T -o BatchMode=yes -o ConnectTimeout=10 \
            -o ServerAliveInterval=15 -o ServerAliveCountMax=2 "${node}" \
            'test -n "${XDG_SESSION_ID:-}" || exit 1
             loginctl show-session "$XDG_SESSION_ID" -p User --value | grep -qx "$(id -u)" || exit 1
             printf "QWEN_IPC_LOGIN_GUARD_READY\n"
             exec cat > /dev/null' \
            < "${_QWEN_IPC_GUARD_DIR}/${node}.stdin" > "${log}" 2>&1 &
        pid=$!
        _QWEN_IPC_GUARD_PIDS+=("${pid}")
        _QWEN_IPC_GUARD_LOGS+=("${log}")
    done
    deadline=$((SECONDS + 45))
    for log in "${_QWEN_IPC_GUARD_LOGS[@]}"; do
        until grep -qx QWEN_IPC_LOGIN_GUARD_READY "${log}"; do
            qwen_check_ipc_login_guards || { tail -n 6 "${log}"; return 1; }
            ((SECONDS < deadline)) || { echo "ERROR: IPC login guard timed out: ${log}" >&2; return 1; }
            sleep 1
        done
    done
    echo "QWEN_IPC_LOGIN_GUARDS: ready nodes=${#nodes[@]} job=${SLURM_JOB_ID}"
}

qwen_check_ipc_login_guards() {
    local pid
    for pid in "${_QWEN_IPC_GUARD_PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null || { echo "ERROR: job-owned IPC login guard exited: ${pid}" >&2; return 1; }
    done
}

qwen_stop_ipc_login_guards() {
    [[ "${QWEN_IPC_LOGIN_GUARD:-0}" == 1 ]] || return 0
    local pid fd
    for pid in "${_QWEN_IPC_GUARD_PIDS[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for fd in "${_QWEN_IPC_GUARD_FDS[@]}"; do
        exec {fd}>&-
    done
    for pid in "${_QWEN_IPC_GUARD_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
                                                                  
    if [[ -n "${_QWEN_IPC_GUARD_DIR:-}" && "${_QWEN_IPC_GUARD_DIR}" == /tmp/qwen35-ipc-guard-* ]]; then
        ((${#_QWEN_IPC_GUARD_FILES[@]} == 0)) || rm -f -- "${_QWEN_IPC_GUARD_FILES[@]}"
        rmdir -- "${_QWEN_IPC_GUARD_DIR}" || true
    fi
    echo "QWEN_IPC_LOGIN_GUARDS: stopped job=${SLURM_JOB_ID:-unknown}"
}

qwen_cleanup_with_ipc_guards() {
    local workload_status=$?
    cleanup_containers
    qwen_stop_ipc_login_guards
    return "${workload_status}"
}
