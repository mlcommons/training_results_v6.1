#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/dsv3-671b}"
CONFIG_FILE="${CONFIG_FILE:-config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh}"
NEXP="${NEXP:-1}"

[[ "$(id -u)" -eq 0 ]] && { echo "LAUNCH BLOCKED: uid 0" >&2; exit 2; }
cd "${WORKDIR}"
[[ -f site.env && -f "${CONFIG_FILE}" && -f run.sub.use2a ]] || {
  echo "LAUNCH BLOCKED: need site.env, ${CONFIG_FILE}, run.sub.use2a in ${WORKDIR}" >&2
  exit 2
}
grep -q 'recompute_granularity = "selective"' pretrain.py || {
  echo "LAUNCH BLOCKED: ${WORKDIR}/pretrain.py does not set selective recompute" >&2
  exit 2
}

set -a
source "${WORKDIR}/site.env"
source "${WORKDIR}/${CONFIG_FILE}"
set +a

export EXTRA_MOUNTS="${WORKDIR}/pretrain.py:/workspace/llm/pretrain.py,${WORKDIR}/numabind.sh:/workspace/llm/numabind.sh"
export PYTHONFAULTHANDLER=1

unset SLURM_ACCOUNT
export NEXP
export CHECK_COMPLIANCE="${CHECK_COMPLIANCE:-0}"
export NCCL_TEST=0
export BINDCMD="${BINDCMD_OVERRIDE:-}"
export DROPCACHE_CMD="${DROPCACHE_OVERRIDE:-true}"
export PMIX_MCA_gds='^ds12'
export NCCL_PROFILER_PLUGIN_OVERRIDE="${NCCL_PROFILER_PLUGIN_OVERRIDE:-none}"
export MLPERF_CLUSTER_NAME="${MLPERF_CLUSTER_NAME:-ml-dev-use2a}"
export MLPERF_SUBMITTER="${MLPERF_SUBMITTER:-CoreWeave}"
export MLPERF_SYSTEM_NAME="${MLPERF_SYSTEM_NAME:-CoreWeave_GB200_64x4}"
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-CoreWeave}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-${MLPERF_SYSTEM_NAME}}"
export LOAD_CHECKPOINTS_DIR="${LOAD_CHECKPOINTS_DIR:-/mnt/data/mlperftv61/dsv3-671b/checkpoint-megatron}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/checkpoints}"
export LOAD_CHECKPOINT_STEP="${LOAD_CHECKPOINT_STEP:-0}"
export SLOW_DATADIR=/mnt/data/mlperftv61/dsv3-671b/dataset
export DATADIR=/tmp/mlperf-c4
export WORKDIR="${HOME}/dsv3-671b"
export LOGDIR="${HOME}/dsv3-671b/logs"
mkdir -p "${LOGDIR}"

NODELIST="${NODELIST:?set NODELIST to the 64-node allocation}"
COUNT="$(scontrol show hostnames "${NODELIST}" | wc -l)"
[[ "${COUNT}" -eq 64 ]] || { echo "LAUNCH BLOCKED: nodelist count ${COUNT}" >&2; exit 2; }

echo "config=${CONFIG_FILE} system=${DGXSYSTEM} tp=${TENSOR_MODEL_PARALLEL} pp=${PIPELINE_MODEL_PARALLEL} vpp=${INTERLEAVED_PIPELINE} ep=${EXPERT_PARALLEL} nvlink_domain=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN} minibs=${MINIBS} recompute=${RECOMPUTE_MODULES:-} cg=${CUDA_GRAPH_IMPLEMENTATION} cg_warmup=${CUDA_GRAPH_WARMUP_STEPS} cap=${MOE_EXPERT_RANK_CAPACITY_FACTOR} stash_cuda=${MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CUDA} mla_down_fusion=${MLA_DOWN_PROJ_FUSION} fp8_output_proj=${FP8_OUTPUT_PROJ} shared_grouped_gemm=${USE_GROUPED_GEMM_FOR_SHARED_EXPERT} glu_interleave=${MOE_SHARED_EXPERT_GLU_INTERLEAVE_SIZE:-unset} bind=${BINDCMD} nvls=${NCCL_NVLS_ENABLE:-unset} cudnn_norm=${NVTE_NORM_FWD_USE_CUDNN}/${NVTE_NORM_BWD_USE_CUDNN} faulthandler=${PYTHONFAULTHANDLER} skip_cg_param_ag=${SKIP_TRAIN_CG_PARAM_AG} nccl_graph_register=${NCCL_GRAPH_REGISTER} alloc=${PYTORCH_CUDA_ALLOC_CONF} layout=${PIPELINE_LAYOUT} mounts=${EXTRA_MOUNTS} step=${LOAD_CHECKPOINT_STEP} nodes=${COUNT} nexp=${NEXP} compliance=${CHECK_COMPLIANCE} mlperf_system=${MLPERF_SYSTEM_NAME} org=${MLPERF_SUBMISSION_ORG} profiler_override=${NCCL_PROFILER_PLUGIN_OVERRIDE:-unset} wall=${WALLTIME} wall_runandtime=${WALLTIME_RUNANDTIME}"

SBATCH_ARGS=(
  --partition=gb200
  --job-name=dsv3-64n-tp2
  --nodes=64
  --gres="gpu:${DGXNGPU}"
  --nodelist="${NODELIST}"
  --exclusive
  --mem=0
  --no-requeue
  --time="${WALLTIME}"
  --output="${LOGDIR}/slurm-%j.out"
  --error="${LOGDIR}/slurm-%j.out"
  --export=ALL
)
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN: sbatch ${SBATCH_ARGS[*]} ${WORKDIR}/run.sub.use2a"; exit 0; }
sbatch "${SBATCH_ARGS[@]}" "${WORKDIR}/run.sub.use2a"
