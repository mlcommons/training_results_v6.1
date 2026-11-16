#!/bin/bash
# Fixed Azure production configuration with externally staged runtime dependencies.

_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export IMPLEMENTATION_ROOT="$(cd -- "${_config_dir}/.." && pwd -P)"

source "${_config_dir}/config_common.sh"
source "${_config_dir}/config_common_mxfp8.sh"
source "${_config_dir}/config_common_full_cg.sh"

export MINIBS=12
export TENSOR_MODEL_PARALLEL=2
export PIPELINE_MODEL_PARALLEL=4
export INTERLEAVED_PIPELINE=4
export MICROBATCH_GROUP_SIZE_PER_VP_STAGE=6
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=32
export PIPELINE_LAYOUT="Et*4|(t*4|)*14tmL"
export MOE_ROUTER_FORCE_LOAD_BALANCING=False
source "${_config_dir}/config_common_moe_flex_hybridep.sh"

export OVERLAP_MOE_EXPERT_PARALLEL_COMM=True
export USE_DYNAMIC_COMP_STREAM=True
export DELAY_WGRAD_COMPUTE=True
export BUCKET_SIZE=536870912
export SKIP_TRAIN_CG_PARAM_AG=False
export DGXNNODES=2560
export DGXNGPU=4
export SEGMENT=16
export DGXSYSTEM=GB200_2560x4x12xtp2pp4ep32cp1_mxfp8_full_cg
export CONFIG_FILENAME="${BASH_SOURCE[0]##*/}"
export WALLTIME_RUNANDTIME=50
export WALLTIME=60
export EVAL_USE_MINIMAL_SCHEDULE_WITH_PADDING=True
export MAX_STEPS=100
unset EVAL_BATCH_SIZE TARGET_LOG_PPL
export MOCK_DATASET=False
export FORCE_SUCCESS_STATUS=0
export USE_SYNTHETIC_DATA=0
export NCCL_TEST=0
# Run official v6.1 compliance checking separately on the collected rank-zero log.
export CHECK_COMPLIANCE=0
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-Azure}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-Azure_GB200}"
export LOAD_CHECKPOINT=/checkpoints
export LOAD_CHECKPOINT_STEP=0

export LOGROOT="${LOGROOT-${IMPLEMENTATION_ROOT}/results}"
export MRC_MOUNTS_FILE="${MRC_MOUNTS_FILE-${_config_dir}/mrc_mounts.sh}"
for _key in CONT DATADIR LOAD_CHECKPOINTS_DIR MLPERF_TRAIN_DIR NODE_LOGDIR NODE_TMPDIR LOGROOT MRC_MOUNTS_FILE; do
    _path="${!_key:-}"
    [[ "$_path" = /* && "$_path" != *[:,]* && "$_path" != *$'\n'* && "$_path" != *$'\r'* ]] || {
        echo "$_key is required, must be absolute, and cannot contain mount delimiters (comma/colon/newline)" >&2
        return 2
    }
    export "$_key"
done
unset _key _path

export VMRC_LIBMRC_SO="/opt/mellanox/doca/lib/aarch64-linux-gnu/libnv_mrc.so"
export VMRC_LIBIBVERBS_SO="/lib/aarch64-linux-gnu/libibverbs.so.1"
export NCCL_VMRC_SO="/opt/microsoft/mrc/mrc-verbs-shim-lib/build/lib/libibverbs.so"
export NCCL_IBVERBS_LIB="/opt/vmrc_shim/lib/libibverbs.so"
export UCX_TLS=tcp
export UCX_NET_DEVICES=enP22p1s0f1
export NCCL_SOCKET_IFNAME=enP22p1s0f1
export GLOO_SOCKET_IFNAME=enP22p1s0f1
export NCCL_NET_PLUGIN=none
export NCCL_IB_HCA=mlx5_1,mlx5_0,mlx5_3,mlx5_2
export NCCL_IB_ECE_ENABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IB_TC=4
export NCCL_IB_FIFO_TC=12
export NV_MRC_POST_SEND_PREFER_BF=1
unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export NCCL_MNNVL_ENABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_SHARP_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG=WARN
export SLURM_TREE_WIDTH=64
export SLURM_TCP_TIMEOUT=10
export SLURM_EIO_TIMEOUT=120
export SRUN_WAIT=120
export SLURM_MPI_TYPE=pmi2
export MRC_LD_LIBRARY_PATH="/opt/nccl_mrc/build/lib:/opt/mellanox/doca/lib/aarch64-linux-gnu:/opt/mellanox/dpdk/lib/aarch64-linux-gnu:/opt/mellanox/flexio/lib:/opt/hpcx-v2.23.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/ompi/lib:/usr/local/cuda/lib64:/lib/aarch64-linux-gnu/:/usr/lib/aarch64-linux-gnu/"
export MRC_PATH="/opt/hpcx-v2.23.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64/ompi/bin"

# Mount construction belongs on the batch host, where driver libraries are present.
EXTRA_MOUNTS="/dev:/dev"
unset MRC_MOUNTS
# A separate strict shell propagates manifest failures and cannot change training flags.
MRC_MOUNTS="$(bash -euo pipefail -c '
    unset MRC_MOUNTS
    source "$1" >&2
    : "${MRC_MOUNTS:?DOCA mount manifest must define MRC_MOUNTS}"
    printf "%s" "$MRC_MOUNTS"
' _ "${MRC_MOUNTS_FILE}")" || {
    echo "Failed to load MRC_MOUNTS_FILE: ${MRC_MOUNTS_FILE}" >&2
    return 2
}
[[ -n "$MRC_MOUNTS" ]] || { echo "Empty MRC_MOUNTS_FILE" >&2; return 2; }
EXTRA_MOUNTS+=",${MRC_MOUNTS}"
unset MRC_MOUNTS
# Apply package submounts after the site manifest's parent-directory mounts.
EXTRA_MOUNTS+=",${MLPERF_TRAIN_DIR}/nccl/lib:/opt/nccl_mrc/build/lib:ro"
EXTRA_MOUNTS+=",${MLPERF_TRAIN_DIR}/hpcx:/opt/hpcx-v2.23.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64:ro"
EXTRA_MOUNTS+=",${MLPERF_TRAIN_DIR}/verbs/lib:/opt/vmrc_shim/lib:ro"
EXTRA_MOUNTS+=",${MLPERF_TRAIN_DIR}/host-verbs/lib:/opt/microsoft/mrc/mrc-verbs-shim-lib/build/lib:ro"
EXTRA_MOUNTS+=",/dev/nvidia-caps-imex-channels/channel0:/dev/nvidia-caps-imex-channels/channel0"
for _driver in /usr/lib/aarch64-linux-gnu/libnvidia* /usr/lib/aarch64-linux-gnu/libcuda*; do
    [[ -f "${_driver}" ]] || continue
    case "${_driver}" in *libcudart.so*) continue ;; esac
    EXTRA_MOUNTS+=",${_driver}:${_driver}:ro"
done
EXTRA_MOUNTS+=",${IMPLEMENTATION_ROOT}/run_and_time.sh:/workspace/llm/run_and_time.sh:ro"
# Only this entrypoint overlay is new: it adds the requested config_filename event.
EXTRA_MOUNTS+=",${IMPLEMENTATION_ROOT}/pretrain.py:/workspace/llm/pretrain.py:ro"
export EXTRA_MOUNTS
unset _driver _config_dir
