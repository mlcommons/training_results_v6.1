#!/bin/bash

                                                                                            
                                              

set -euo pipefail
export SUBMISSION_ORG=Cisco
export SUBMISSION_PLATFORM=Zeus_8xB300_SXM_288GB_multi_planar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TRAINING_DIR="${SCRIPT_DIR}"

                         
export NEXP="${NEXP:-10}"

export NCCL_SOCKET_IFNAME="mgmt_eth"
export NCCL_IB_GID_INDEX=3
export NCCL_TEST="${NCCL_TEST:-0}"
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export GLOO_SOCKET_IFNAME="mgmt_eth"
source "${SCRIPT_DIR}/network.sh"

                                                                            
export DATADIR="${DATADIR:?Set DATADIR}"
export MODEL="${MODEL:?Set MODEL}"
export VERIFY_MOUNTS=0
                                                                            
export MLPERF_RULESET="${MLPERF_RULESET:-6.0.0}"
export LOGDIR="${SCRIPT_DIR}/results"

B300_LLAMA2_CONFIG="${B300_LLAMA2_CONFIG:-config_Zeus_8xB300_SXM_288GB_multi_planar.sh}"
source "${SCRIPT_DIR}/${B300_LLAMA2_CONFIG}"

                                                                                      
                                                                                         
                                                                
export BINDCMD="bindpcie --cpu=node"
                                                                                                 
export SKIP_PACKED_ATTENTION_MASK=1
export NUM_WORKERS=2
                                                                                   
                                                                                        
                                   
unset TORCH_SHARING_STRATEGY

export EXTRA_MOUNTS="${SCRIPT_DIR}:/workspace/ft-llm:rw"
export CONT="${CONT:?Set CONT}"

export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))

mkdir -p "${LOGDIR}"

sbatch --job-name="llama2-70b-b300-quadplane" \
    -D "${SCRIPT_DIR}" \
    --output="${LOGDIR}/slurm-%j.out" \
    -p "${SBATCH_PARTITION:?Set SBATCH_PARTITION}" \
    -w "${SBATCH_NODELIST:?Set SBATCH_NODELIST}" -N "${DGXNNODES}" --time="${WALLTIME}" \
    "${SCRIPT_DIR}/run.sub"
