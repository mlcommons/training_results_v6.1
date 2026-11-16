#!/bin/bash
# Container-side entry point invoked by run.sub's hang monitor when
# ATTEMPT_CUDA_GDB_CORE_DUMP=1. Triggers a CUDA lightweight core dump in every
# training process on this node by writing to the per-process coredump pipes.
#
# Assumes the application was launched with CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
# and CUDA_COREDUMP_PIPE set, both of which run.sub handles.
#
# The resulting .nvcudmp files are readable offline with:
#   cuda-gdb -c core_<host>_<pid>.nvcudmp
#   (gdb) info cuda kernels
#
# Env:
#   CUDA_COREDUMP_PIPE  pipe name pattern created for user-triggered core dumps

set -u

if [ -z "${CUDA_COREDUMP_PIPE:-}" ]; then
    echo "ERROR: CUDA_COREDUMP_PIPE is not set; nothing to trigger." >&2
    exit 1
fi

PIPE_DIR="$(dirname "${CUDA_COREDUMP_PIPE}")"
echo "$0: host=$(hostname -s) triggering GPU core dumps via pipes in ${PIPE_DIR}"
echo "$0: dumps will land at ${CUDA_COREDUMP_FILE:-<CUDA_COREDUMP_FILE unset>}"

if [ ! -d "$PIPE_DIR" ]; then
    echo "ERROR: ${PIPE_DIR} does not exist; no pipes were created." >&2
    exit 1
fi

shopt -s nullglob
pipes=("${PIPE_DIR}"/*)
if [ "${#pipes[@]}" -eq 0 ]; then
    echo "ERROR: no coredump pipes in ${PIPE_DIR}." >&2
    exit 1
fi

# Writing to each pipe blocks until the CUDA runtime reads it, so fan out and wait.
for corepipe in "${pipes[@]}"; do
    echo 1 >> "$corepipe" &
done
wait

echo "$0: core dump trigger complete on $(hostname -s)"
