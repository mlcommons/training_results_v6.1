#!/usr/bin/env bash
set -euo pipefail
export PATH="$PATH:/opt/slurm/bin"
cd /home/mlperf/flux1/implementation-20260929
export CONT=/data/mlperf/images/flux1-arm-20260929.sqsh
export DATAROOT=/data/mlperf/flux1/data/energon
export LOGDIR=/home/mlperf/logs/flux1/config_GB300_18x04x16-20260929
export NEXP="${NEXP:-1}"
source config_GB300_18x04x16.sh
export MLPERF_CLUSTER_NAME="gb300-icat"
export SRUN_KILL_ON_BAD_EXIT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${LOGDIR}/nccl_trace_rank_"
export NVIDIA_VISIBLE_DEVICES=all
export MLPERF_SUBMISSION_ORG="Crusoe"
export MLPERF_SUBMISSION_PLATFORM="crusoe_gb300"
NODELIST="${NODELIST:-r204-compute-node-gb300-icat-[000-017]}"
SBATCH_ARGS=(
    -N "${DGXNNODES}"
    --time="${WALLTIME}"
    -A crusoe -p r204 --exclusive
    -w "${NODELIST}"
    --job-name=flux1-72gpu
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
