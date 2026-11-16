#!/bin/bash
# Container-side entry point invoked by run.sub's hang monitor, once per node.
# run.sub calls this by exact name from ${WORK_DIR}, so keep the filename.
#
# Delegates to dump_node_stacks.sh, which samples every training process on the node
# concurrently (py-spy, py-spy --native, gdb) and optionally cuda-gdb.
#
# Env (set by run.sub / hang_monitor):
#   TRACEBACKS_ID                     dump session id (DATESTAMP)
#   RESULTS_DIR                       default /results/tracebacks
#   ATTEMPT_CUDA_GDB_TRACEBACKS_DUMP  1 = also capture 'info cuda kernels'. Defaults
#                                     to 0, matching llama31_8b: attaching cuda-gdb can
#                                     abort the process, so it must not happen because
#                                     an opt-out failed to propagate into the container.
#                                     Prefer ATTEMPT_CUDA_GDB_CORE_DUMP, which yields the
#                                     same 'info cuda kernels' offline without attaching.

set -u

: "${TRACEBACKS_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${RESULTS_DIR:=/results/tracebacks}"
: "${ATTEMPT_CUDA_GDB_TRACEBACKS_DUMP:=0}"
: "${COLLECT_NATIVE:=1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${SCRIPT_DIR}/dump_node_stacks.sh"

if [ ! -f "$COLLECTOR" ]; then
    echo "ERROR: ${COLLECTOR} not found. Is scripts/tracebacks mounted into the container?" >&2
    exit 1
fi

export OUT_DIR="${RESULTS_DIR}/${TRACEBACKS_ID}"
export COLLECT_NATIVE
export COLLECT_CUDA_GDB="${ATTEMPT_CUDA_GDB_TRACEBACKS_DUMP}"

echo "$0: host=$(hostname -s) -> ${OUT_DIR} (cuda_gdb=${COLLECT_CUDA_GDB})"
exec bash "$COLLECTOR"
