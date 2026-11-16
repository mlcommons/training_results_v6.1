#!/bin/bash
                                                                   
                                                                              
                                                                             
 
                                                                                 
                                                            
 
                                                         
                                         
                           
 
      
                                                                               

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

                                                                                   
for corepipe in "${pipes[@]}"; do
    echo 1 >> "$corepipe" &
done
wait

echo "$0: core dump trigger complete on $(hostname -s)"
