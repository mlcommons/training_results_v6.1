#!/bin/bash
set -euo pipefail
export SUBMISSION_ORG=Cisco
export SUBMISSION_PLATFORM=Zeus_8xB300_SXM_288GB_multi_planar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"

export DATADIR="${DATADIR:?Set DATADIR}"
export LOAD_CHECKPOINTS_DIR="${LOAD_CHECKPOINTS_DIR:?Set LOAD_CHECKPOINTS_DIR}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/checkpoints}"
export CONT="${CONT:?Set CONT}"
export LOGDIR="${LOGDIR:-${SCRIPT_DIR}/results/b300_v61_$(date +%Y%m%d_%H%M%S)}"
export REMOUNT_WORKDIR="${SCRIPT_DIR}"

export CHECK_COMPLIANCE="${CHECK_COMPLIANCE:-0}"
export VERIFY_MOUNTS="${VERIFY_MOUNTS:-0}"
export CLEAR_CACHES="${CLEAR_CACHES:-0}"
export NCCL_TEST="${NCCL_TEST:-0}"
export PRE_RUN_CLEANUP_PYXIS="${PRE_RUN_CLEANUP_PYXIS:-0}"
export STARTUP_TIMING="${STARTUP_TIMING:-1}"
export HANG_MONITOR_TIMEOUT="${HANG_MONITOR_TIMEOUT:-12}"

                                                                            
                                                                              
                                                                             
                                                                            
                                                                         
                                                    
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-gpu0.0_eth,gpu0.1_eth,gpu0.2_eth,gpu0.3_eth,gpu1.0_eth,gpu1.1_eth,gpu1.2_eth,gpu1.3_eth,gpu2.0_eth,gpu2.1_eth,gpu2.2_eth,gpu2.3_eth,gpu3.0_eth,gpu3.1_eth,gpu3.2_eth,gpu3.3_eth,gpu4.0_eth,gpu4.1_eth,gpu4.2_eth,gpu4.3_eth,gpu5.0_eth,gpu5.1_eth,gpu5.2_eth,gpu5.3_eth,gpu6.0_eth,gpu6.1_eth,gpu6.2_eth,gpu6.3_eth,gpu7.0_eth,gpu7.1_eth,gpu7.2_eth,gpu7.3_eth}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-gpu0.0_rdma,gpu0.1_rdma,gpu0.2_rdma,gpu0.3_rdma,gpu1.0_rdma,gpu1.1_rdma,gpu1.2_rdma,gpu1.3_rdma,gpu2.0_rdma,gpu2.1_rdma,gpu2.2_rdma,gpu2.3_rdma,gpu3.0_rdma,gpu3.1_rdma,gpu3.2_rdma,gpu3.3_rdma,gpu4.0_rdma,gpu4.1_rdma,gpu4.2_rdma,gpu4.3_rdma,gpu5.0_rdma,gpu5.1_rdma,gpu5.2_rdma,gpu5.3_rdma,gpu6.0_rdma,gpu6.1_rdma,gpu6.2_rdma,gpu6.3_rdma,gpu7.0_rdma,gpu7.1_rdma,gpu7.2_rdma,gpu7.3_rdma}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-mgmt_eth}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-mgmt_eth}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export B300_QUADRAIL_PER_LOCAL_RANK="${B300_QUADRAIL_PER_LOCAL_RANK:-1}"

                                                                       
                                                                              
                                         
export NVTE_UB_SOCKET_IFNAME="${NVTE_UB_SOCKET_IFNAME:-mgmt_eth}"
export TE_UB_DISABLE="${TE_UB_DISABLE:-1}"
export TE_UB_FT_COMM_ID="${TE_UB_FT_COMM_ID:-nccl}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmi2}"

export EXTRA_CONTAINER_ENV="${EXTRA_CONTAINER_ENV:-NCCL_IB_GID_INDEX,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,UCX_NET_DEVICES,NVTE_UB_SOCKET_IFNAME,TE_UB_DISABLE,TE_UB_FT_COMM_ID,HYDRA_FULL_ERROR,STRICT_FSDP_DTENSOR_LOAD,EXIT_ON_MISSING_CHECKPOINT,MEGATRON_FSDP_VERSION,B300_QUADRAIL_PER_LOCAL_RANK}"

CONFIG_FILE="${B300_CONFIG_FILE:-${SCRIPT_DIR}/config_Zeus_8xB300_SXM_288GB_multi_planar.sh}"
[[ -f "${CONFIG_FILE}" ]] || { echo "Missing config: ${CONFIG_FILE}" >&2; exit 1; }
source "${CONFIG_FILE}"

                                                                                      
[[ "${LOGDIR}" != / && "${LOGDIR}" != "${SCRIPT_DIR}" ]] || { echo "Choose a dedicated LOGDIR" >&2; exit 1; }

                                                                                 
if [[ "${HYBRID_EP_USE_PREBUILT_CACHE:-0}" == "1" ]]; then
    export HYBRID_EP_CACHE_TAG="${HYBRID_EP_CACHE_TAG:-b300_ep8_cp2_t2048_h7168_e32_top8_bf16_pad128_sm32_deepep58397eac}"
    export HYBRID_EP_CACHE_DIR="${HYBRID_EP_CACHE_DIR:-/tmp/hybridep-jit}"
    export HYBRID_EP_PREBUILT_CACHE="${HYBRID_EP_PREBUILT_CACHE:-/workspace/llm/.hybridep_jit/${HYBRID_EP_CACHE_TAG}}"
    EXTRA_CONTAINER_ENV="${EXTRA_CONTAINER_ENV},HYBRID_EP_CACHE_DIR,HYBRID_EP_PREBUILT_CACHE"
    export EXTRA_CONTAINER_ENV
fi

PARTITION="${PARTITION:?Set PARTITION}"
NODELIST="${NODELIST:?Set NODELIST}"
ACCOUNT="${ACCOUNT:-}"

[[ -f "${CONT}" ]] || { echo "Missing container: ${CONT}" >&2; exit 1; }
if [[ -n "${SLOW_DATADIR:-}" ]]; then
                                                                             
                                                                            
                                                                            
    [[ -d "${SLOW_DATADIR}/8b" ]] || { echo "Missing staging dataset: ${SLOW_DATADIR}/8b" >&2; exit 1; }
    echo "Dataset staging: ${SLOW_DATADIR}/8b -> ${DATADIR}/8b on each node"
else
    [[ -d "${DATADIR}/8b" ]] || { echo "Missing dataset: ${DATADIR}/8b" >&2; exit 1; }
fi
[[ -d "${LOAD_CHECKPOINTS_DIR}/iter_0000000" ]] || { echo "Missing checkpoint: ${LOAD_CHECKPOINTS_DIR}" >&2; exit 1; }
if [[ "${HYBRID_EP_USE_PREBUILT_CACHE:-0}" == "1" ]]; then
    HOST_HYBRID_EP_PREBUILT_CACHE="${SCRIPT_DIR}/.hybridep_jit/${HYBRID_EP_CACHE_TAG}"
    if ! compgen -G "${HOST_HYBRID_EP_PREBUILT_CACHE}/rank-*/*.so" >/dev/null; then
        echo "Missing HybridEP prebuilt kernels: ${HOST_HYBRID_EP_PREBUILT_CACHE}" >&2
        echo "Run: sbatch ${SCRIPT_DIR}/precompile_hybridep_b300.sub" >&2
        exit 1
    fi
fi

SBATCH_ARGS=(
    -N "${DGXNNODES}"
    --time="${WALLTIME}"
    --gres="gpu:${DGXNGPU}"
    -p "${PARTITION}"
    -w "${NODELIST}"
)
[[ -z "${ACCOUNT}" ]] || SBATCH_ARGS+=( -A "${ACCOUNT}" )
[[ -z "${B300_SLURM_DEPENDENCY:-}" ]] || SBATCH_ARGS+=( --dependency="${B300_SLURM_DEPENDENCY}" )
[[ -z "${B300_JOB_NAME:-}" ]] || SBATCH_ARGS+=( --job-name="${B300_JOB_NAME}" )
[[ -z "${B300_SLURM_OUTPUT:-}" ]] || SBATCH_ARGS+=( --output="${B300_SLURM_OUTPUT}" --error="${B300_SLURM_OUTPUT}" )

WORLD_SIZE=$((DGXNNODES * DGXNGPU))
DATA_PARALLEL=$((WORLD_SIZE / (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL)))
echo "Submitting v6.1 B300 qualification: PP=${PIPELINE_MODEL_PARALLEL} VP=${INTERLEAVED_PIPELINE:-none} EP=${EXPERT_PARALLEL} DP=${DATA_PARALLEL} MBS=${MICRO_BATCH_SIZE} GBS=$((MINIBS * DATA_PARALLEL))"
echo "Logs: ${LOGDIR}"
exec sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run.sub"
