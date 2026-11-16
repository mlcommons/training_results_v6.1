#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=5:00:00
#SBATCH --job-name=bruno-primus-llama
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Run one of AMD's Primus MLPerf Llama recipes (examples/mlperf/<BENCH>) with
# AMD's own run_with_docker.sh, from a Primus checkout on the host.
#   BENCH       llama2_70b | llama3.1_8b
#   PRIMUS_HOST Primus checkout (default ~/primus/v26.7.0)
#   CONT        image (default rocm/primus:v26.7)
# Never runs AMD's runtime_tunables.sh (it writes kernel settings, which we
# must not change); AMD's launcher still drops the page cache per trial.
# HF_TOKEN comes from ~/.env (needed by the Llama 2 hooks).
#
# Prefer NEXP=1 per job (chain jobs instead). AMD's launcher runs all trials in
# one container; on lodin (strict overcommit) Llama 3.1 trial 2 died with
# 'unable to allocate shared memory ... No space left on device' in a
# DataLoader worker, although /dev/shm had 2 TB free.
#
# Usage: sbatch -w <node> --export=ALL,BENCH=llama2_70b,TAG=v26.7 run_primus_llama.sbatch [NEXP]
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/results ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
BENCH=${BENCH:?set BENCH=llama2_70b or llama3.1_8b}
export NEXP=${1:-${NEXP:-1}}
export PRIMUS_HOST=${PRIMUS_HOST:-/home/hpebench/primus/v26.7.0}
export CONT=${CONT:-rocm/primus:v26.7}
export RUN_RUNTIME_TUNABLES=0 INTERACTIVE=0
case ${BENCH} in
  llama2_70b)  export DATADIR=${DATADIR:-/lvol0/mlperf/llama2/data}; R=llama2_70b ;;
  llama3.1_8b) export DATADIR=${DATADIR:-/lvol0/mlperf/llama31_8b/data}
               export MODELDIR=${MODELDIR:-/lvol0/mlperf/llama31_8b/model}; R=llama31_8b ;;
  *) echo "FATAL: unknown BENCH ${BENCH}"; exit 1 ;;
esac
export LOGDIR=${LOGDIR:-${NAS}/results/${R}/${TAG:-primus-v26.7}}
set -a; . /home/hpebench/.env; set +a
[[ -n "${HF_TOKEN:-}" ]] || { echo "FATAL: no HF_TOKEN in ~/.env"; exit 1; }

echo "=== ${BENCH} $(date -u) on $(hostname): Primus $(cd ${PRIMUS_HOST} && git log -1 --format='%h %cs') + local changes: $(cd ${PRIMUS_HOST} && git status --short | wc -l) files; CONT=${CONT}; NEXP=${NEXP}"
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    tar="${NAS}/containers/$(echo "${CONT#rocm/}" | tr '/:' '__').tar"
    [[ -f "${tar}" ]] && docker load -q -i "${tar}"
    docker image inspect "${CONT}" >/dev/null 2>&1 || { echo "FATAL: image ${CONT} not on node and no tar at ${tar}"; exit 1; }
fi
mkdir -p "${LOGDIR}"
bash "${PRIMUS_HOST}/examples/mlperf/${BENCH}/run_with_docker.sh" </dev/null
echo "=== done $(date -u)"
