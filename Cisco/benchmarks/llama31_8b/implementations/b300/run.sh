#!/bin/bash

                                                                        
                                                                             

set -euo pipefail
export SUBMISSION_ORG=Cisco
export SUBMISSION_PLATFORM=Zeus_8xB300_SXM_288GB_multi_planar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TRAINING_DIR="${SCRIPT_DIR}"

                                               
export NCCL_SOCKET_IFNAME="mgmt_eth"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_CFG_PATH="/workspace/llm/conf/nccl/custom_communicator_cta.yaml"
export NCCL_SHARP_GROUP_SIZE_THRESH=2
export NCCL_WORK_FIFO_DEPTH=1048576
export SLURM_MPI_TYPE="pmi2"
export GLOO_SOCKET_IFNAME="mgmt_eth"
export NVTE_UB_SOCKET_IFNAME="mgmt_eth"

export DATADIR="${DATADIR:?Set DATADIR}"

source "${SCRIPT_DIR}/config_Zeus_8xB300_SXM_288GB_multi_planar.sh"
source "${SCRIPT_DIR}/network.sh"

export REMOUNT_WORKDIR="${SCRIPT_DIR}"
export CONT="${CONT:?Set CONT}"
export LOGDIR="${SCRIPT_DIR}/results"
mkdir -p "${LOGDIR}"

export NEXP="${NEXP:-2}"
export NCCL_TEST="${NCCL_TEST:-0}"
export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))

sbatch --job-name="llama8b-8N-b300-quadplane" \
    -D "${SCRIPT_DIR}" \
    --output="${LOGDIR}/slurm-%j.out" \
    -p "${SBATCH_PARTITION:?Set SBATCH_PARTITION}" \
    -w "${SBATCH_NODELIST:?Set SBATCH_NODELIST}" -N "${DGXNNODES}" --time="${WALLTIME}" \
    "${SCRIPT_DIR}/run.sub"
