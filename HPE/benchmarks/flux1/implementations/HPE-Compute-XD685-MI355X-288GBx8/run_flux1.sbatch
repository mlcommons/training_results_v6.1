#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=10:00:00
#SBATCH --job-name=bruno-flux1
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Flux.1-schnell, 1 node x 8 MI355X, AMD's own config_MI355X_01x08x64.sh:
# GBS 512 (MBS 64), lr 2e-4, warmup 1600 = the BS 512 RCP hyperparameters.
# Needs a prepared energon tree at ${DATADIR}/energon (see stage_flux_from_nas.sbatch).
#
# Paths: lodin1/2 mount the NAS root at /nas (striker's /nas/SHARED/MLPERF is
# /nas/data/SHARED/MLPERF there); striker and lodin3 mount /nas/data at /nas.
# Detected below. Slurm logs go to /home, identical on all hosts.
#
# Usage: sbatch [-p dmoise -w lodin3] --export=ALL,TAG=x run_flux1.sbatch [NEXP]
set -uo pipefail

if [[ -d /nas/data/SHARED/MLPERF/results ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
IMPL=${IMPL:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}   # submit from this directory
export NEXP=${1:-${NEXP:-1}}
export CONT=${CONT:-rocm/amd-mlperf:flux1_training_6.0}
export DATADIR=${DATADIR:-/lvol0/mlperf/flux1}
export LOGDIR=${LOGDIR:-${NAS}/results/flux1/${TAG:-single}}
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-1}

echo "=== flux1 $(date -u) on $(hostname), NEXP=${NEXP}, LOGDIR=${LOGDIR}"

# A CG job's epilog runs disable-docker.sh and would kill our container.
for _ in $(seq 1 60); do
    [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
[[ -d ${DATADIR}/energon/.nv-meta ]] || { echo "FATAL: no prepared energon tree in ${DATADIR}"; exit 1; }
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    tar="${NAS}/containers/$(echo "${CONT#rocm/}" | tr '/:' '__').tar"
    if [[ -f "${tar}" ]]; then echo "loading ${CONT} from ${tar} ($(date -u +%H:%M))"; docker load -q -i "${tar}"; fi
    docker image inspect "${CONT}" >/dev/null 2>&1 || docker pull -q "${CONT}"
fi
docker image inspect "${CONT}" >/dev/null 2>&1 || { echo "FATAL: no image ${CONT}"; exit 1; }

mkdir -p "${LOGDIR}"
cd "${IMPL}"
export PYTHONPATH=""
source config_MI355X_01x08x64.sh
bash run_with_docker.sh
echo "=== done $(date -u)"
