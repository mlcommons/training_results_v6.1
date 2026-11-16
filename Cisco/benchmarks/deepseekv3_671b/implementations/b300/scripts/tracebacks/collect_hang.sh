#!/bin/bash
                                                                             
                                                                     
 
                                                                                  
                                                              
 
        
                                   
                                                                                      
 
      
                                                
                                                                                  
                                                                       
                                                                                      
                                                         
                                                                                             

set -euo pipefail

: "${JOBID:?set JOBID to the running job id}"
: "${MODEL_NAME:=deepseekv3_671b}"
: "${CONTAINER:=${MODEL_NAME}_${JOBID}}"
: "${COLLECT_NATIVE:=1}"
: "${COLLECT_CUDA_GDB:=0}"
: "${OUT_DIR:=/results/hang_artifacts/$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${SCRIPT_DIR}/dump_node_stacks.sh"
[ -f "$COLLECTOR" ] || { echo "ERROR: missing ${COLLECTOR}" >&2; exit 1; }

NNODES="$(squeue -j "$JOBID" -h -o %D | tr -d ' ')"
STATE="$(squeue -j "$JOBID" -h -o %T | tr -d ' ')"
if [ "$STATE" != "RUNNING" ]; then
    echo "ERROR: job ${JOBID} is in state '${STATE}', not RUNNING." >&2
    echo "The processes must still be alive to attach. Raise the NCCL timeout so the" >&2
    echo "watchdog does not abort the job before you get here." >&2
    exit 1
fi

echo "job=${JOBID} nodes=${NNODES} container=${CONTAINER}"
echo "out=${OUT_DIR}  native=${COLLECT_NATIVE}  cuda_gdb=${COLLECT_CUDA_GDB}"

B64="$(base64 -w0 "$COLLECTOR")"

srun --overlap --jobid="$JOBID" -N "$NNODES" --ntasks-per-node=1 -l \
    --container-name="$CONTAINER" \
    bash -c "echo '${B64}' | base64 -d > /tmp/dump_node_stacks.sh && \
        OUT_DIR='${OUT_DIR}' \
        COLLECT_NATIVE='${COLLECT_NATIVE}' \
        COLLECT_CUDA_GDB='${COLLECT_CUDA_GDB}' \
        bash /tmp/dump_node_stacks.sh"

echo
echo "Artifacts are in ${OUT_DIR} (inside the container = \$LOGDIR/hang_artifacts/... on the host)."
echo "Summarize with:"
echo "  python3 ${SCRIPT_DIR}/summarize_stacks.py <host_path_to_that_dir> --kind pyspy"
echo "  python3 ${SCRIPT_DIR}/summarize_stacks.py <host_path_to_that_dir> --kind gdb"
