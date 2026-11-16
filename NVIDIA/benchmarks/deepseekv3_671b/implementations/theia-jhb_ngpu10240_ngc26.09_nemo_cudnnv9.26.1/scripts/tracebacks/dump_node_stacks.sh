#!/bin/bash
# Collect host (and optionally device) state for every training process on this node.
# Must run INSIDE the training container, where the training PIDs are visible.
#
# All processes on the node are sampled concurrently, so wall time is roughly the
# time for one process rather than N.
#
# Env:
#   OUT_DIR            output directory (default /results/hang_artifacts/<timestamp>)
#   COLLECT_NATIVE     1 = also capture py-spy --native (shows NCCL/C++ frames). Default 1.
#   COLLECT_CUDA_GDB   1 = capture 'info cuda kernels'/'info cuda devices'. Default 0,
#                      because attaching cuda-gdb can abort the process. Runs last.
#   GDB_TIMEOUT        per-process gdb timeout in seconds (default 120)

set -u

: "${OUT_DIR:=/results/hang_artifacts/$(date +%Y%m%d_%H%M%S)}"
: "${COLLECT_NATIVE:=1}"
: "${COLLECT_CUDA_GDB:=0}"
: "${GDB_TIMEOUT:=120}"

HOST="$(hostname -s)"
mkdir -p "$OUT_DIR"

# Install only what is missing; apt/pip on every node every time is slow.
if ! command -v py-spy >/dev/null 2>&1; then
    pip install --quiet py-spy >/dev/null 2>&1
fi
if ! command -v gdb >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq gdb >/dev/null 2>&1
fi

CUDA_GDB="${CUDA_GDB_BIN:-$(command -v cuda-gdb || echo /usr/local/cuda/bin/cuda-gdb)}"

# Training processes = whatever holds a GPU, falling back to the entrypoint name.
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -un)"
if [ -z "$pids" ]; then
    pids="$(pgrep -f 'pretrain.py' 2>/dev/null | sort -un)"
fi

if [ -z "$pids" ]; then
    echo "[$HOST] no training processes found"
    exit 0
fi

# Global rank, so filenames match what the NCCL/torch logs call each rank.
rank_of() {
    local pid="$1" r=""
    if [ -r "/proc/$pid/environ" ]; then
        r="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^RANK=//p' | head -1)"
        [ -z "$r" ] && r="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^SLURM_PROCID=//p' | head -1)"
    fi
    echo "$r"
}

basename_for() {
    local pid="$1" rank tag
    rank="$(rank_of "$pid")"
    if [ -n "$rank" ] && [ -z "${rank//[0-9]/}" ]; then
        tag="$(printf 'rank%05d' "$rank")"
    else
        tag="rankUNKNOWN"
    fi
    echo "${OUT_DIR}/${tag}_${HOST}_pid${pid}"
}

collect_one() {
    local pid="$1" base
    base="$(basename_for "$pid")"

    {
        echo "host=$HOST pid=$pid rank=$(rank_of "$pid") date=$(date -Is)"
        echo "cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    } > "${base}.meta" 2>&1

    timeout 60 py-spy dump --pid "$pid" > "${base}.pyspy" 2>&1
    if [ "$COLLECT_NATIVE" = "1" ]; then
        timeout 180 py-spy dump --native --pid "$pid" > "${base}.pyspy-native" 2>&1
    fi
    timeout "$GDB_TIMEOUT" gdb -p "$pid" -batch -ex 'thread apply all bt' > "${base}.gdb" 2>&1
}

echo "[$HOST] collecting from PIDs: $(echo $pids | tr '\n' ' ')-> $OUT_DIR"

for pid in $pids; do
    ps -p "$pid" >/dev/null 2>&1 || continue
    collect_one "$pid" &
done
wait

# cuda-gdb last and serialized: it can kill the process, so everything else is
# already on disk by the time it runs.
if [ "$COLLECT_CUDA_GDB" = "1" ]; then
    for pid in $pids; do
        ps -p "$pid" >/dev/null 2>&1 || continue
        base="$(basename_for "$pid")"
        timeout 240 "$CUDA_GDB" -batch \
            -ex 'info cuda devices' \
            -ex 'info cuda kernels' \
            -p "$pid" > "${base}.cuda-gdb" 2>&1
    done
fi

echo "[$HOST] done -> $OUT_DIR"
