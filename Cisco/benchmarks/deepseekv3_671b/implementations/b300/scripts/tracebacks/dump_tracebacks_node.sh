#!/bin/bash
                                                                              
                                                                          
 
                                                                                    
                                                                      
 
                                      
                                                                 
                                                                 
                                                                                    
                                                                                       
                                                                                      
                                                                                        
                                                                                         
                                                                                         

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
