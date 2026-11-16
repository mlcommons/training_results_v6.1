#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/dsv3-671b}"
BATCH="${BATCH:-run.sub.vr}"
ARM="${ARM:-baseline}"
NEXP="${NEXP:-1}"
SHORT="${SHORT:-0}"
PROFILE_ARM="${PROFILE_ARM:-0}"
SYSTEM_NAME=CoreWeave_VR200_64x4

SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --set) SETS+=("$2"); shift 2 ;;
    --set=*) SETS+=("${1#--set=}"); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

case "${ARM}" in
  baseline) CONFIG_FILE="${CONFIG_FILE:-config_VR200_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh}" ;;
  *) echo "unknown ARM ${ARM} (baseline)" >&2; exit 2 ;;
esac

[[ "$(id -u)" -eq 0 ]] && { echo "LAUNCH BLOCKED: uid 0" >&2; exit 2; }
cd "${WORKDIR}"
[[ -f site.env && -f "${CONFIG_FILE}" && -f "${BATCH}" ]] || {
  echo "LAUNCH BLOCKED: need site.env, ${CONFIG_FILE}, ${BATCH} in ${WORKDIR}" >&2; exit 2
}
grep -q 'resolved MASTER_ADDR' "${BATCH}" || { echo "LAUNCH BLOCKED: ${BATCH} lacks the IPv4 MASTER_ADDR resolver" >&2; exit 2; }
grep -q 'NCCL_PROFILER_PLUGIN_OVERRIDE' "${BATCH}" || { echo "LAUNCH BLOCKED: ${BATCH} lacks the NCCL profiler override" >&2; exit 2; }

_PRE_MAX_STEPS="${MAX_STEPS-}"
_PRE_WALLTIME_RUNANDTIME="${WALLTIME_RUNANDTIME-}"
set -a
source site.env
source "${CONFIG_FILE}"
set +a
mkdir -p "${LOGDIR}"

if [[ -n "${SETS_FILE:-}" ]]; then
  [[ -f "${SETS_FILE}" ]] || { echo "LAUNCH BLOCKED: SETS_FILE ${SETS_FILE} missing" >&2; exit 2; }
  while IFS= read -r line; do
    line="${line%%#*}"; line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [[ -n "${line}" ]] && SETS=("${line}" "${SETS[@]+"${SETS[@]}"}")
  done < "${SETS_FILE}"
fi
for kv in "${SETS[@]+"${SETS[@]}"}"; do
  [[ "${kv}" == *=* ]] || { echo "bad --set ${kv}" >&2; exit 2; }
  export "${kv?}"
done

[[ "${CONT}" == *-20261006.sqsh ]] || { echo "LAUNCH BLOCKED: CONT is ${CONT}, expected the 20261006 image" >&2; exit 2; }
[[ -f "${CONT}" ]] || { echo "LAUNCH BLOCKED: missing image ${CONT}" >&2; exit 2; }
[[ -f "${DATADIR}/8b/tokenizer/tokenizer.json" ]] || { echo "LAUNCH BLOCKED: missing tokenizer under ${DATADIR}/8b" >&2; exit 2; }
[[ -f "${DATADIR}/8b/c4-train.en_6_text_document.bin" ]] || { echo "LAUNCH BLOCKED: missing C4 train shard under ${DATADIR}/8b" >&2; exit 2; }
[[ -d "${LOAD_CHECKPOINTS_DIR}/iter_0000000" && -f "${LOAD_CHECKPOINTS_DIR}/latest_train_state.pt" ]] || {
  echo "LAUNCH BLOCKED: Megatron checkpoint incomplete at ${LOAD_CHECKPOINTS_DIR}" >&2; exit 2
}
CKPT_FILES=$(find "${LOAD_CHECKPOINTS_DIR}/iter_0000000" -maxdepth 1 -type f | wc -l)
[[ "${CKPT_FILES}" -ge 500 ]] || { echo "LAUNCH BLOCKED: iter_0000000 has ${CKPT_FILES} files (expected 518)" >&2; exit 2; }

LOCAL_COPY="${LOCAL_COPY:-0}"
if [[ "${LOCAL_COPY}" == "1" ]]; then
  export SLOW_DATADIR="${DATADIR}"
  export DATADIR="${LOCAL_DATADIR:-/tmp/mlperf-c4}"
fi

DP=$(( DGXNNODES * DGXNGPU / (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) ))
GBS=$(( MINIBS * DP ))
[[ "${GBS}" -eq 15360 ]] || { echo "LAUNCH BLOCKED: MINIBS ${MINIBS} x DP ${DP} = ${GBS}, not 15360" >&2; exit 2; }
[[ $(( DP % EXPERT_PARALLEL )) -eq 0 ]] || { echo "LAUNCH BLOCKED: DP ${DP} not a multiple of EP ${EXPERT_PARALLEL}" >&2; exit 2; }
[[ $(( MINIBS % MICRO_BATCH_SIZE )) -eq 0 ]] || { echo "LAUNCH BLOCKED: MINIBS ${MINIBS} not a multiple of MICRO_BATCH_SIZE ${MICRO_BATCH_SIZE}" >&2; exit 2; }
[[ $(( EXPERT_PARALLEL * TENSOR_MODEL_PARALLEL )) -le 72 ]] || { echo "LAUNCH BLOCKED: EP x TP ${EXPERT_PARALLEL}x${TENSOR_MODEL_PARALLEL} exceeds one NVL72 rack" >&2; exit 2; }

if [[ "${SHORT}" == "1" ]]; then
  export MAX_STEPS="${_PRE_MAX_STEPS:-20}"
  export CHECK_COMPLIANCE=0
  export WALLTIME_RUNANDTIME="${_PRE_WALLTIME_RUNANDTIME:-60}"
  JOB_SUFFIX="-short"
else
  [[ -n "${_PRE_MAX_STEPS}" ]] && export MAX_STEPS="${_PRE_MAX_STEPS}"
  export CHECK_COMPLIANCE="${CHECK_COMPLIANCE:-1}"
  export WALLTIME_RUNANDTIME="${_PRE_WALLTIME_RUNANDTIME:-90}"
  JOB_SUFFIX=""
fi
if [[ "${PROFILE_ARM}" == "1" ]]; then
  export NVTX_FLAG=1 PROFILE=True PROFILE_RANKS="${PROFILE_RANKS:-0}"
  export PROFILE_START_STEP="${PROFILE_START_STEP:-10}" PROFILE_END_STEP="${PROFILE_END_STEP:-12}"
  JOB_SUFFIX="${JOB_SUFFIX}-prof"
fi

export NCCL_TEST="${NCCL_TEST:-1}"
export BINDCMD=
export DROPCACHE_CMD=true
export PMIX_MCA_gds='^ds12' SLURM_MPI_TYPE=pmix
export ATTEMPT_CUDA_GDB_CORE_DUMP=0 HANG_MONITOR_TIMEOUT=0
export PYTHONFAULTHANDLER=1
export NCCL_PROFILER_PLUGIN_OVERRIDE="${NCCL_PROFILER_PLUGIN_OVERRIDE:-none}"
export EXTRA_MOUNTS="${WORKDIR}/pretrain.py:/workspace/llm/pretrain.py"
export NEXP
export MLPERF_SUBMITTER="${MLPERF_SUBMITTER:-CoreWeave}"
export MLPERF_SYSTEM_NAME="${MLPERF_SYSTEM_NAME:-${SYSTEM_NAME}}"
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-CoreWeave}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-${SYSTEM_NAME}}"
export WALLTIME="${WALLTIME_OVERRIDE:-$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))}"
unset SLURM_ACCOUNT SBATCH_SEGMENT_SIZE

if [[ -z "${NODELIST:-}" ]]; then
  NODELIST="$(scontrol show topology | sed -n 's/^BlockName=[^ ]*.*Nodes=\([^ ]*\).*/\1/p' \
    | while read -r nodes; do scontrol show hostnames "${nodes}" | sort | head -16; done | paste -sd, -)"
  NODELIST="$(scontrol show hostlistsorted "${NODELIST}")"
fi
COUNT="$(scontrol show hostnames "${NODELIST}" | wc -l)"
[[ "${COUNT}" -eq "${DGXNNODES}" ]] || { echo "LAUNCH BLOCKED: nodelist has ${COUNT} nodes, config wants ${DGXNNODES}" >&2; exit 2; }
while read -r blk nodes; do
  n=$(comm -12 <(scontrol show hostnames "${NODELIST}" | sort) <(scontrol show hostnames "${nodes}" | sort) | wc -l)
  [[ "${n}" -eq 16 ]] || { echo "LAUNCH BLOCKED: ${blk} contributes ${n} nodes, want 16" >&2; exit 2; }
done < <(scontrol show topology | sed -n 's/^BlockName=\([^ ]*\).*Nodes=\([^ ]*\).*/\1 \2/p')
IDLE="$(sinfo -h -N -n "${NODELIST}" -t idle -o '%N' | sort -u | wc -l)"
[[ "${IDLE}" -eq "${COUNT}" || "${ALLOW_BUSY_NODES:-0}" == "1" ]] || { echo "LAUNCH BLOCKED: ${IDLE} of ${COUNT} nodes idle (ALLOW_BUSY_NODES=1 to queue anyway)" >&2; exit 2; }

JOB_NAME="dsv3-64n-${ARM}${JOB_SUFFIX}"
echo "arm=${ARM} config=${CONFIG_FILE} system=${DGXSYSTEM} tp=${TENSOR_MODEL_PARALLEL} pp=${PIPELINE_MODEL_PARALLEL} vpp=${INTERLEAVED_PIPELINE} ep=${EXPERT_PARALLEL} dp=${DP} nvlink_domain=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN} lr=${LR:-config} dispatcher=${MOE_FLEX_DISPATCHER_BACKEND} minibs=${MINIBS} mbs=${MICRO_BATCH_SIZE} gbs=${GBS} layout=${PIPELINE_LAYOUT} recompute=${RECOMPUTE_MODULES:-none} wgrad_ovl=${OVERLAP_DISPATCH_BACKWARD_WITH_EXPERTS_WGRAD:-} te_act=${USE_TE_ACTIVATION_FUNC:-} margins=${NVTE_EXT_MARGIN_SM:-}/${CUDNNFE_CLUSTER_OVERLAP_MARGIN:-} ub_dp=${NCCL_UB_DP:-} offload=${FINE_GRAINED_ACTIVATION_OFFLOADING:-}:${OFFLOAD_MODULES:-} mb_group=${MICROBATCH_GROUP_SIZE_PER_VP_STAGE:-} cg=${CUDA_GRAPH_IMPLEMENTATION} stash=${MOE_PAGED_STASH:-} conn=${CUDA_DEVICE_MAX_CONNECTIONS:-} ep_sms=${NCCL_EP_SHUFFLE_SMS:-}/${NCCL_EP_PREPROCESS_NUM_SMS:-}/${NCCL_EP_PROLOG_EPILOG_SMS:-} sgw=${MOE_SINGLE_GROUPED_WEIGHT:-} skipag=${SKIP_TRAIN_CG_PARAM_AG:-} cap=${MOE_EXPERT_RANK_CAPACITY_FACTOR} stash_cuda=${MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CUDA} mla_down_fusion=${MLA_DOWN_PROJ_FUSION} fp8_output_proj=${FP8_OUTPUT_PROJ} fp8_dpa=${FP8_DPA} attn=${MLPERF_ATTN_PRECISION} max_steps=${MAX_STEPS:-config} compliance=${CHECK_COMPLIANCE} nexp=${NEXP} profile=${PROFILE:-False} profile_ranks=${PROFILE_RANKS:-} local_copy=${LOCAL_COPY} datadir=${DATADIR} slow_datadir=${SLOW_DATADIR:-} sets=${SETS[*]+"${SETS[*]}"} hca=${NCCL_IB_HCA} profiler_override=${NCCL_PROFILER_PLUGIN_OVERRIDE} nodes=${COUNT} idle=${IDLE} wall=${WALLTIME} wall_runandtime=${WALLTIME_RUNANDTIME} cont=${CONT} job=${JOB_NAME}"

SBATCH_ARGS=(
  --job-name="${JOB_NAME}"
  --nodes="${DGXNNODES}" --gpus-per-node="${DGXNGPU}" --nodelist="${NODELIST}"
  --exclusive --mem=0 --no-requeue --time="${WALLTIME}"
  --output="${LOGDIR}/slurm-%j.out" --error="${LOGDIR}/slurm-%j.out" --export=ALL
)
[[ -n "${DEPENDENCY:-}" ]] && SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
if [[ "${REAL:-0}" != "1" ]]; then
  echo "DRY_RUN (set REAL=1 to submit): sbatch ${SBATCH_ARGS[*]} ${WORKDIR}/${BATCH}"
  exit 0
fi
sbatch "${SBATCH_ARGS[@]}" "${WORKDIR}/${BATCH}"
