#!/usr/bin/env bash
set -euo pipefail

export PATH="$PATH:/opt/slurm/bin"

cd /home/mlperf/flux1/implementation

export CONT=/data/mlperf/images/flux1-arm.sqsh
export DATAROOT=/data/mlperf/flux1/data/energon
export LOGDIR=/home/mlperf/logs/flux1/config_GB300_04x04x32
export NEXP="${NEXP:-1}"

# 1. SOURCE THE CONFIG FIRST
source config_GB300_04x04x32.sh

# 2. THEN export overrides
export MLPERF_CLUSTER_NAME="gb300-icat"
export SRUN_KILL_ON_BAD_EXIT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${LOGDIR}/nccl_trace_rank_"
export NVIDIA_VISIBLE_DEVICES=all

NODELIST="${NODELIST:?NODELIST must be set explicitly for this benchmark}"

SBATCH_ARGS=(
    -N "${DGXNNODES}"
    --time="${WALLTIME}"
    -A crusoe -p r204 --exclusive
    -w "${NODELIST}"
    --job-name=flux1-16gpu
    --export=ALL
)

echo "+ sbatch ${SBATCH_ARGS[*]} ... run.sub  (NEXP=${NEXP})"
if [[ "${TEST_ONLY:-0}" -eq 1 ]]; then
    sbatch "${SBATCH_ARGS[@]}" --test-only run.sub
elif [[ "${RUN:-0}" -eq 1 ]]; then
    sbatch "${SBATCH_ARGS[@]}" run.sub
else
    echo "(dry run: not submitting. Set RUN=1 to submit, TEST_ONLY=1 for sbatch --test-only.)"
fi
