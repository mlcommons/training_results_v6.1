#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${HOME}/gpt-oss-20b"
CONFIG_FILE="${CONFIG_FILE:-config_GB300_18x4x1xtp1pp1cp2ep4_mxfp8.sh}"
# System name and job name follow the config node count (16x4 since the 2026-10-14 NVIDIA 16-node file).
SEED_BASE="${SEED_BASE:-21530}"
NEXP="${NEXP:-1}"

[[ "$(id -u)" -eq 0 ]] && { echo "LAUNCH BLOCKED: uid 0" >&2; exit 2; }
cd "${WORKDIR}"
[[ -f site.env && -f "${CONFIG_FILE}" && -f run.sub ]] || {
  echo "LAUNCH BLOCKED: need site.env, ${CONFIG_FILE}, run.sub in ${WORKDIR}" >&2; exit 2
}

_PRE_LR="${LR-}"
_PRE_LR_WARMUP_STEPS="${LR_WARMUP_STEPS-}"
_PRE_WARMUP_STEPS="${WARMUP_STEPS-}"
_PRE_CUDA_GRAPH_WARMUP_STEPS="${CUDA_GRAPH_WARMUP_STEPS-}"
_PRE_AVERAGE_IN_COLLECTIVE="${AVERAGE_IN_COLLECTIVE-}"

set -a
# shellcheck disable=SC1091
source site.env
# shellcheck disable=SC1091
source "${CONFIG_FILE}"
set +a
SYSTEM_NAME="CoreWeave_GB200_${DGXNNODES}x${DGXNGPU}"
# shellcheck disable=SC1091
source "${WORKDIR}/gb200-opts.sh"
# Per-run srun --time budget override (the config value is sized for VR200).
[[ -n "${WALLTIME_RUNANDTIME_OVERRIDE:-}" ]] && export WALLTIME_RUNANDTIME="${WALLTIME_RUNANDTIME_OVERRIDE}"
mkdir -p "${LOGDIR}"

TOK="${TOK:-${SLOW_DATADIR:-${DATADIR}}/8b/tokenizer/tokenizer.json}"
[[ -f "${TOK}" ]] || { echo "LAUNCH BLOCKED: missing ${TOK}" >&2; exit 2; }

export BINDCMD="" DROPCACHE_CMD="true"
export ATTEMPT_CUDA_GDB_CORE_DUMP=0 HANG_MONITOR_TIMEOUT=0
export PMIX_MCA_gds='^ds12' SLURM_MPI_TYPE=pmix
export NCCL_PROFILER_PLUGIN_OVERRIDE="${NCCL_PROFILER_PLUGIN_OVERRIDE:-none}"
export USE_LIBUV=1 TORCH_USE_LIBUV=1
export NEXP SEED_BASE
export MLPERF_CLUSTER_NAME="${MLPERF_CLUSTER_NAME:-ml-dev-use2a}"
export MLPERF_SUBMITTER="${MLPERF_SUBMITTER:-CoreWeave}"
export MLPERF_SYSTEM_NAME="${MLPERF_SYSTEM_NAME:-${SYSTEM_NAME}}"
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-CoreWeave}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-${SYSTEM_NAME}}"
export WALLTIME="${WALLTIME_OVERRIDE:-$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))}"

WORKDIR="${WORKDIR}" bash "${WORKDIR}/patch-run-sub-use2a.sh"

echo "CONFIG=${CONFIG_FILE} NODES=${DGXNNODES} NEXP=${NEXP} WALLTIME=${WALLTIME}"
echo "DATADIR=${DATADIR} CONT=${CONT} LOGDIR=${LOGDIR}"

SBATCH_ARGS=(
  --job-name="gptoss-${DGXNNODES}n-n${NEXP}" --partition=gb200 --nodes="${DGXNNODES}" --gpus-per-node="${DGXNGPU}"
  --segment="${SEGMENT}" --exclusive --mem=0 --time="${WALLTIME}"
  --output="${LOGDIR}/slurm-%j.out" --error="${LOGDIR}/slurm-%j.out" --export=ALL
)
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=(--nodelist="${NODELIST}")
[[ -n "${EXCLUDE:-}" ]] && SBATCH_ARGS+=(--exclude="${EXCLUDE}")
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN: sbatch ${SBATCH_ARGS[*]} run.sub.use2a"; exit 0; }
sbatch "${SBATCH_ARGS[@]}" "${WORKDIR}/run.sub.use2a"
