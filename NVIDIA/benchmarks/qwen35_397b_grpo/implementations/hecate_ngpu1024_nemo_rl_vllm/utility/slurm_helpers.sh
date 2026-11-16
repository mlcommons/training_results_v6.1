#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Shared host/container helpers; sourcing does not change shell options.
# Keep identical copies so either workload can be distributed independently.
# Standalone topology detection:
#   bash path/to/slurm_helpers.sh detect-gpu-cpu-affinity 4 > gpu_cpu_affinity

detect_cpus_per_node() {
    local nodelist first_node raw cpus
    if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
        echo "ERROR: SLURM_JOB_NODELIST is unset; set CPUS_PER_WORKER explicitly" >&2
        return 1
    fi
    nodelist=$(scontrol show hostnames "${SLURM_JOB_NODELIST}") || return 1
    first_node=${nodelist%%$'\n'*}
    if [[ -z "${first_node}" ]]; then
        echo "ERROR: no allocated nodes found; set CPUS_PER_WORKER explicitly" >&2
        return 1
    fi
    raw=$(scontrol show node "${first_node}" -o) || return 1
    cpus=${raw#*CPUTot=}
    cpus=${cpus%% *}
    if ! [[ "${cpus}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: could not determine CPUTot for ${first_node}; set CPUS_PER_WORKER explicitly" >&2
        return 1
    fi
    printf '%s\n' "${cpus}"
}

configure_cpus_per_worker() {
    # Cluster configs or the caller take precedence. Detection assumes exclusive,
    # homogeneous whole-node allocations; it reads the first node's total CPUs.
    if [[ -z "${CPUS_PER_WORKER:-}" ]]; then
        CPUS_PER_WORKER=$(detect_cpus_per_node) || return 1
    fi
    if ! [[ "${CPUS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: CPUS_PER_WORKER must be a positive integer; got '${CPUS_PER_WORKER}'" >&2
        return 1
    fi
    export CPUS_PER_WORKER
    echo "CPUS_PER_WORKER=${CPUS_PER_WORKER}"
}

initialize_gpu_cpu_affinity_settings() {
    # Set to 0 to disable binding, including reuse of a stale default map.
    : "${NRL_ENABLE_GPU_CPU_AFFINITY:=1}"
    case "$NRL_ENABLE_GPU_CPU_AFFINITY" in
        0) export NRL_GPU_CPU_AFFINITY_FILE=/dev/null ;;
        1) ;;
        *) echo "ERROR: NRL_ENABLE_GPU_CPU_AFFINITY must be 0 or 1" >&2; return 1 ;;
    esac
    export NRL_ENABLE_GPU_CPU_AFFINITY
}

# Isolate pipefail and temporary variables from the sourcing shell.
detect_gpu_cpu_affinity() (
    set -euo pipefail

    expected_gpus=${1:-}
    if [[ $# -gt 1 || ( -n "$expected_gpus" && ! "$expected_gpus" =~ ^[1-9][0-9]*$ ) ]]; then
        echo "Usage: bash $0 detect-gpu-cpu-affinity [expected_gpu_count]" >&2
        return 1
    fi

    gpu_numa=$(nvidia-smi topo -m | awk '/^GPU[0-9]+[[:space:]]/ {
        gpu = $1
        sub(/^GPU/, "", gpu)
        numa = $(NF-1)
        # GB200/GB300 may report CPU NUMA followed by GPU/HBM nodes.
        sub(/[,-].*/, "", numa)
        print gpu, numa
    }') || return 1
    if [[ -z "$gpu_numa" ]]; then
        echo "ERROR: no GPU NUMA entries found in nvidia-smi topo -m" >&2
        return 1
    fi

    # Buffer the map so a failed probe never prints a partial result.
    affinity_map=$(
        while read -r gpu numa; do
            if ! [[ "$numa" =~ ^[0-9]+$ ]]; then
                echo "ERROR: GPU $gpu has no usable CPU NUMA node: $numa" >&2
                return 1
            fi
            cpulist=$(cat "/sys/devices/system/node/node${numa}/cpulist") || return 1
            if ! [[ "$cpulist" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
                echo "ERROR: invalid CPU list for GPU $gpu on NUMA node $numa" >&2
                return 1
            fi
            printf '%s:%s\n' "$gpu" "$cpulist"
        done <<< "$gpu_numa"
    ) || return 1
    entries=$(printf '%s\n' "$affinity_map" | wc -l | tr -d '[:space:]')
    if [[ -n "$expected_gpus" && "$entries" -ne "$expected_gpus" ]]; then
        echo "ERROR: expected $expected_gpus GPU NUMA entries, found $entries" >&2
        return 1
    fi
    printf '%s\n' "$affinity_map"
)

configure_gpu_cpu_affinity() {
    local affinity_map
    initialize_gpu_cpu_affinity_settings || return 1
    if [[ "$NRL_ENABLE_GPU_CPU_AFFINITY" == "0" ]]; then
        echo "GPU CPU affinity disabled"
        return 0
    fi
    if [[ -n "${NRL_GPU_CPU_AFFINITY_FILE:-}" ]]; then
        # Supplied paths must be readable inside every Ray container;
        # callers can use EXTRA_MOUNTS for files outside existing mounts.
        if [[ ! -r "$NRL_GPU_CPU_AFFINITY_FILE" ]]; then
            echo "ERROR: affinity file is not readable: $NRL_GPU_CPU_AFFINITY_FILE" >&2
            return 1
        fi
        echo "Using supplied NUMA affinity map: $NRL_GPU_CPU_AFFINITY_FILE"
    else
        affinity_map=$(detect_gpu_cpu_affinity "$@") || return 1
        export NRL_GPU_CPU_AFFINITY_FILE=/tmp/nrl_gpu_cpu_affinity
        printf '%s\n' "$affinity_map" > "$NRL_GPU_CPU_AFFINITY_FILE" || return 1
        echo "NUMA affinity map written to $NRL_GPU_CPU_AFFINITY_FILE:"
    fi
    export NRL_GPU_CPU_AFFINITY_FILE
    cat "$NRL_GPU_CPU_AFFINITY_FILE" || return 1
    if ! python3 -c 'import ctypes; ctypes.CDLL("libnuma.so.1")'; then
        echo "ERROR: libnuma.so.1 is not loadable" >&2
        return 1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        detect-gpu-cpu-affinity) shift; detect_gpu_cpu_affinity "$@" ;;
        *) echo "Usage: bash $0 detect-gpu-cpu-affinity [expected_gpu_count]" >&2; exit 1 ;;
    esac
fi
