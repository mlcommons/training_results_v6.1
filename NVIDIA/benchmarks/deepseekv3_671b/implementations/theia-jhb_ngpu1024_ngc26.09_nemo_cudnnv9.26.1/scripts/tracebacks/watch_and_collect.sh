#!/bin/bash
# Watch a running job's log and automatically collect hang artifacts when it stalls.
# Removes the need to sit and watch for the log to stop advancing.
#
# Stall detection is deliberately dumb: if the log file stops growing for
# STALL_MINUTES, treat the job as hung. That is reliable here because a healthy run
# emits progress continuously, and it is far more robust than parsing iteration
# numbers.
#
# Usage:
#   JOBID=1234567 LOGFILE=/path/to/slurm-1234567.out ./watch_and_collect.sh
#
# Env:
#   JOBID          (required) running job id
#   LOGFILE        (required) the log to watch for growth
#   STALL_MINUTES  minutes of no growth before collecting (default 12)
#   POLL_SECONDS   how often to check (default 60)
#   CUDA_AFTER     1 = after the host-side pass, do a second cuda-gdb pass that can
#                  kill the job (default 1; the host pass is already safely on disk)

set -euo pipefail

: "${JOBID:?set JOBID}"
: "${LOGFILE:?set LOGFILE to the job log to watch}"
: "${STALL_MINUTES:=12}"
: "${POLL_SECONDS:=60}"
: "${CUDA_AFTER:=1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT="${SCRIPT_DIR}/collect_hang.sh"
[ -x "$COLLECT" ] || { echo "ERROR: missing or non-executable ${COLLECT}" >&2; exit 1; }

stall_limit=$(( STALL_MINUTES * 60 ))
last_size=-1
unchanged_for=0

echo "watching ${LOGFILE} for job ${JOBID}; stall threshold ${STALL_MINUTES}m"

while true; do
    state="$(squeue -j "$JOBID" -h -o %T 2>/dev/null | tr -d ' ' || true)"
    if [ "$state" != "RUNNING" ]; then
        echo "job ${JOBID} is no longer RUNNING (state='${state:-gone}'); nothing to attach to."
        exit 0
    fi

    size="$(stat -c %s "$LOGFILE" 2>/dev/null || echo 0)"
    if [ "$size" = "$last_size" ]; then
        unchanged_for=$(( unchanged_for + POLL_SECONDS ))
    else
        unchanged_for=0
        last_size="$size"
    fi

    if [ "$unchanged_for" -ge "$stall_limit" ]; then
        echo "log has not grown for ${unchanged_for}s -> collecting host-side artifacts"
        OUT_DIR="/results/hang_artifacts/$(date +%Y%m%d_%H%M%S)"
        JOBID="$JOBID" OUT_DIR="$OUT_DIR" COLLECT_CUDA_GDB=0 bash "$COLLECT" || true

        if [ "$CUDA_AFTER" = "1" ]; then
            echo "host-side pass complete; now collecting device state (may abort the job)"
            JOBID="$JOBID" OUT_DIR="$OUT_DIR" COLLECT_NATIVE=0 COLLECT_CUDA_GDB=1 \
                bash "$COLLECT" || true
        fi

        echo "collection finished into ${OUT_DIR}"
        exit 0
    fi

    sleep "$POLL_SECONDS"
done
