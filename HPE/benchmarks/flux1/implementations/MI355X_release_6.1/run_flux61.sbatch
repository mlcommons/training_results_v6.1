#!/bin/bash
#SBATCH --partition=MDF-High
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --job-name=monnetb-flux61
#SBATCH --output=%x-%j.out
#
# HPE: AMD 6.1 FLUX.1 drop (3b590b6), one node at global batch 512
# (config_MI355X_1x8x64.sh). Host VAR=VAL arguments override the config.
# lodin4 (user monnetb): data on local /lvol0 (stage_flux_lodin4.sbatch), logs in $HOME.
# lodin1/2 (hpebench): data read from /nas over InfiniBand, logs in /nas results.
#
# Usage: lodin4: sbatch -w lodin4 --export=ALL,TAG=<tag>[,NEXP=n] run_flux61.sbatch [VAR=VAL ...]
#        lodin1: sbatch -p def -w lodin1 --comment=docker --export=ALL,TAG=<tag> run_flux61.sbatch [VAR=VAL ...]
set -uo pipefail
TAG=${TAG:?set TAG}
export NEXP=${NEXP:-1}
if [[ -d /nas/data/SHARED/MLPERF/containers ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
if [[ $(id -un) == hpebench ]]; then
    IMPL=${IMPL:-/home/hpebench/mlcommons_mlperf_training_61a/HPE/benchmarks/flux1/implementations/MI355X_release_6.1}
    DATADIR=${DATADIR:-${NAS}/sd-flux}
    LOGDIR=${LOGDIR:-${NAS}/results/flux1/${TAG}}
fi
IMPL=${IMPL:-${HOME}/mlperf61/flux1}
export CONT=${CONT:-rocm/amd-mlperf:flux1_training_6.1}
export DGXSYSTEM=${DGXSYSTEM:-MI355X_1x8x64}
export DATADIR=${DATADIR:-/lvol0/mlperf/flux1_mlc}
export LOGDIR=${LOGDIR:-${HOME}/mlperf61/results/flux1/${TAG}}
export CONT_NAME=mlperf_flux1_61_${SLURM_JOB_ID}
export RUN_RUNTIME_TUNABLES=0
export MLLOG_SUBMISSION_ORG=${HPE_ORG:-HPE} MLLOG_SUBMISSION_PLATFORM=${HPE_PLATFORM:-XD685_MI355X}
unset PYTHONPATH
for kv in "$@"; do export "${kv?}"; done

echo "=== flux1 6.1 $(date -u) on $(hostname): TAG=${TAG} NEXP=${NEXP} CONT=${CONT} overrides: $*"
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
[[ -f "${DATADIR}/.stage_complete" || "${DATADIR}" == "${NAS}/sd-flux" ]] || { echo "FATAL: ${DATADIR} not staged/verified"; exit 1; }
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    tar=${NAS}/containers/amd-mlperf_${CONT##*:}.tar; [[ -r "${tar}" ]] || tar=${HOME}/containers/amd-mlperf_${CONT##*:}.tar
    docker load -q -i "${tar}" || { echo "FATAL: no image ${CONT}"; exit 1; }
fi
mkdir -p "${LOGDIR}"
{ date -Is; echo "job=${SLURM_JOB_ID} node=$(hostname -s) image=$(docker image inspect --format '{{.Id}}' "${CONT}") config=${DGXSYSTEM}"
  echo "overrides: $*"; } >> "${LOGDIR}/launch.txt"
# Run from a per-job copy kept with the results (exact code, checkout may change).
CODE=${LOGDIR}/code-${SLURM_JOB_ID}-$(date +%y%m%d%H%M%S)
cp -a "${IMPL}" "${CODE}" && cd "${CODE}" || exit 1
bash run_with_docker.sh
rc=$?
echo "=== done rc=${rc} $(date -u)"
exit ${rc}
