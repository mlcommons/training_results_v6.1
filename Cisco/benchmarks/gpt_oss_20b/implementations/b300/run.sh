#!/bin/bash

                                                                       
                                                                                 

set -euo pipefail
export SUBMISSION_ORG=Cisco
export SUBMISSION_PLATFORM=Zeus_8xB300_SXM_288GB_multi_planar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TRAINING_DIR="${SCRIPT_DIR}"
BENCH_DIR="${SCRIPT_DIR}"

export LOGDIR="${SCRIPT_DIR}/results"
mkdir -p "${LOGDIR}"

export DATADIR="${DATADIR:?Set DATADIR}"
export REMOUNT_WORKDIR="${SCRIPT_DIR}"

source "${SCRIPT_DIR}/config_Zeus_8xB300_SXM_288GB_multi_planar.sh"
source "${SCRIPT_DIR}/network.sh"

export CONT="${CONT:?Set CONT}"
export NEXP="${NEXP:-2}"

export NCCL_SOCKET_IFNAME="mgmt_eth"
export GLOO_SOCKET_IFNAME="mgmt_eth"
export NVTE_UB_SOCKET_IFNAME="mgmt_eth"
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_TEST="${NCCL_TEST:-0}"

export PMIX_GDS_MODULE=^ds12
export PMIX_MCA_gds=^ds12
export SLURM_MPI_TYPE="pmi2"

                                          
unset NCCL_CFG_PATH
unset NCCL_SHARP_GROUP_SIZE_THRESH

export NVTX_FLAG=0
export PROFILE=False
export TRANSFORMERS_CACHE="/tmp/"
export TRITON_HOME="/tmp/"
export HF_HOME="/tmp/"
export WALLTIME=$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))

sbatch \
    --job-name="gpt_oss_20b-b300-quadplane" \
    -D "${SCRIPT_DIR}" \
    --output="${LOGDIR}/slurm-%j.out" \
    -p "${SBATCH_PARTITION:?Set SBATCH_PARTITION}" \
    -w "${SBATCH_NODELIST:?Set SBATCH_NODELIST}" \
    -N "${DGXNNODES}" \
    --time="${WALLTIME}" \
    "${SCRIPT_DIR}/run.sub"
