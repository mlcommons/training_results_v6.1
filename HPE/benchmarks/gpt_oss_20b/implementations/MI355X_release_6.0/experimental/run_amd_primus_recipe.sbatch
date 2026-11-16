#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=8:00:00
#SBATCH --job-name=bruno-gptoss-amd
#SBATCH --output=/nas/SHARED/MLPERF/results/gpt_oss_20b/amd-%j.out
#
# AMD's in-progress MLPerf GPT-OSS-20B recipe from Primus main
# (examples/mlperf/gpt_oss_20b): FP8 + Turbo attention, GBS 32, MBS 4,
# public image rocm/primus:v26.5. Built for 1 node x 8 MI355X.
#
# Why: the MLCommons reference (BF16, GBS 16) runs at ~10 samples/s on
# MI355X -- both here and in AMD's own rcp_logs -- i.e. 500-700 min per run.
#
# Only change vs AMD: PRIMUS_NUM_WORKERS=0. lodin runs
# vm.overcommit_memory=2, and forked loader workers fail with ENOMEM.
#
# Usage: sbatch run_amd_primus_recipe.sbatch [NEXP]
set -uo pipefail

PRIMUS_COMMIT=${PRIMUS_COMMIT:-a48595c8}
PRIMUS_DIR=${PRIMUS_DIR:-/lvol0/mlperf/primus_${PRIMUS_COMMIT}}
export NEXP=${1:-${NEXP:-1}}
export CONT=${CONT:-rocm/primus:v26.5}
export DATADIR=${DATADIR:-/lvol0/mlperf/llama31_8b/data}
export MODELDIR=${MODELDIR:-/lvol0/mlperf/llama31_8b/model}
export LOGDIR=${LOGDIR:-/nas/SHARED/MLPERF/results/gpt_oss_20b/${TAG:-amd-recipe}}
export DGXSYSTEM=MI355X_1x8x1
export PRIMUS_NUM_WORKERS=${PRIMUS_NUM_WORKERS:-0}
export RUN_RUNTIME_TUNABLES=${RUN_RUNTIME_TUNABLES:-1}

echo "=== gpt_oss_20b AMD recipe $(date -u) on $(hostname), Primus ${PRIMUS_COMMIT}, NEXP=${NEXP}"
for _ in $(seq 1 60); do
    [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }

if [[ ! -d ${PRIMUS_DIR}/.git ]]; then
    git clone -q https://github.com/AMD-AGI/Primus.git "${PRIMUS_DIR}" \
        || git clone -q https://github.com/AMD-AIG-AIMA/Primus.git "${PRIMUS_DIR}" \
        || { echo "FATAL: cannot clone Primus"; exit 1; }
fi
cd "${PRIMUS_DIR}" && git fetch -q origin && git checkout -q "${PRIMUS_COMMIT}" \
    && git submodule update -q --init --recursive || { echo "FATAL: checkout ${PRIMUS_COMMIT}"; exit 1; }
echo "Primus at $(git log -1 --format='%h %cs')"
docker image inspect "${CONT}" >/dev/null 2>&1 || docker pull -q "${CONT}"

mkdir -p "${LOGDIR}"
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
echo "=== done $(date -u)"
