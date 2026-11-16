#!/bin/bash
                                                                                     
                                                                              
 
                                                                                 
                                     
 
      
                                                                                     
                                                                                           
                                                                                      
                                                                                   
                                                                       

set -u

: "${OUT_DIR:=/results/hang_artifacts/$(date +%Y%m%d_%H%M%S)}"
: "${COLLECT_NATIVE:=1}"
: "${COLLECT_CUDA_GDB:=0}"
: "${GDB_TIMEOUT:=120}"

HOST="$(hostname -s)"
mkdir -p "$OUT_DIR"

                                                                         
if ! command -v py-spy >/dev/null 2>&1; then
    pip install --quiet py-spy >/dev/null 2>&1
fi
if ! command -v gdb >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq gdb >/dev/null 2>&1
fi

CUDA_GDB="${CUDA_GDB_BIN:-$(command -v cuda-gdb || echo /usr/local/cuda/bin/cuda-gdb)}"

                                                                                 
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -un)"
if [ -z "$pids" ]; then
    pids="$(pgrep -f 'pretrain.py' 2>/dev/null | sort -un)"
fi

if [ -z "$pids" ]; then
    echo "[$HOST] no training processes found"
    exit 0
fi

                                                                          
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
