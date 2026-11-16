#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-config_VR200_18x4x1xtp1pp1cp1_8b.sh}"
SYSTEM_NAME=CoreWeave_VR200_18x4
SEED_BASE="${SEED_BASE:-31900}"
NEXP="${NEXP:-10}"

[[ "$(id -u)" -eq 0 ]] && { echo "LAUNCH BLOCKED: uid 0" >&2; exit 2; }
[[ "${NEXP}" == "1" || ( "${NEXP}" =~ ^[0-9]+$ && "${NEXP}" -ge 10 ) ]] || {
  echo "LAUNCH BLOCKED: NEXP=${NEXP}; use NEXP>=10 (one job per scored set) or NEXP=1 (single run)" >&2; exit 2
}
cd "${WORKDIR}"
[[ -f site.env && -f "${CONFIG_FILE}" && -f run.sub ]] || {
  echo "LAUNCH BLOCKED: need site.env, ${CONFIG_FILE}, run.sub in ${WORKDIR}" >&2; exit 2
}

set -a
source site.env
source "${CONFIG_FILE}"
set +a
mkdir -p "${LOGDIR}"

if [[ -n "${SLOW_DATADIR_SRC:-}" ]]; then
  export SLOW_DATADIR="${SLOW_DATADIR_SRC}"
  export DATADIR="${LOCAL_DATADIR:-/tmp/mlperf-c4}"
fi
[[ -n "${WARMUP_STEPS_OVERRIDE:-}" ]] && export WARMUP_STEPS="${WARMUP_STEPS_OVERRIDE}"
[[ -n "${LR_WARMUP_STEPS_OVERRIDE:-}" ]] && export LR_WARMUP_STEPS="${LR_WARMUP_STEPS_OVERRIDE}"
[[ -n "${CUDA_GRAPH_WARMUP_STEPS_OVERRIDE:-}" ]] && export CUDA_GRAPH_WARMUP_STEPS="${CUDA_GRAPH_WARMUP_STEPS_OVERRIDE}"

export BINDCMD="" DROPCACHE_CMD="true"
export ATTEMPT_CUDA_GDB_CORE_DUMP=0 HANG_MONITOR_TIMEOUT=0
export PMIX_MCA_gds='^ds12' SLURM_MPI_TYPE=pmix
export USE_LIBUV=1 TORCH_USE_LIBUV=1
export NEXP SEED_BASE
export MLPERF_CLUSTER_NAME="${MLPERF_CLUSTER_NAME:-vr-testing}"
export MLPERF_SUBMITTER="${MLPERF_SUBMITTER:-CoreWeave}"
export MLPERF_SYSTEM_NAME="${MLPERF_SYSTEM_NAME:-${SYSTEM_NAME}}"
export MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-CoreWeave}"
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-${SYSTEM_NAME}}"
export WALLTIME="${WALLTIME_OVERRIDE:-$((5 + NEXP * (WALLTIME_RUNANDTIME + 5)))}"

WORKDIR="${WORKDIR}" bash "${WORKDIR}/patch-run-sub-vr.sh"

echo "CONFIG=${CONFIG_FILE} NODES=${DGXNNODES} NEXP=${NEXP} WALLTIME=${WALLTIME} WARMUP_STEPS=${WARMUP_STEPS:-} CUDA_GRAPH_WARMUP_STEPS=${CUDA_GRAPH_WARMUP_STEPS:-}"
echo "SLOW_DATADIR=${SLOW_DATADIR:-} DATADIR=${DATADIR} CONT=${CONT} LOGDIR=${LOGDIR}"

SBATCH_ARGS=(
  --job-name="llama31-18n-n${NEXP}" --nodes="${DGXNNODES}" --gpus-per-node="${DGXNGPU}"
  --exclusive --mem=0 --time="${WALLTIME}"
  --output="${LOGDIR}/slurm-%j.out" --error="${LOGDIR}/slurm-%j.out" --export=ALL
)
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=(--nodelist="${NODELIST}")
[[ -n "${EXCLUDE:-}" ]] && SBATCH_ARGS+=(--exclude="${EXCLUDE}")
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN: sbatch ${SBATCH_ARGS[*]} run.sub.vr"; exit 0; }
sbatch "${SBATCH_ARGS[@]}" "${WORKDIR}/run.sub.vr"
