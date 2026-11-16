#!/usr/bin/env bash
set -euo pipefail
export PATH="$PATH:/opt/slurm/bin"
cd /home/mlperf/gpt_oss_20b/implementation
export CONT=/data/mlperf/images/gpt_oss_20b-arm.sqsh
export DATADIR=/data/mlperf/llama31_8b/data
export DATA_FOLDER=8b
export LOGDIR=/home/mlperf/logs/gpt_oss_20b/config_GB300_18x4x1xtp1pp1cp2ep4_mxfp8
export NEXP="${NEXP:-1}"
source config_GB300_18x4x1xtp1pp1cp2ep4_mxfp8.sh
export MLPERF_CLUSTER_NAME="gb300-icat"
export SRUN_KILL_ON_BAD_EXIT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${LOGDIR}/nccl_trace_rank_"
export NVIDIA_VISIBLE_DEVICES=all
NODELIST="${NODELIST:-r204-compute-node-gb300-icat-[000-017]}"
SBATCH_ARGS=(
    -N "${DGXNNODES}"
    --time="${WALLTIME}"
    -A crusoe -p r204 --exclusive
    -w "${NODELIST}"
    --job-name=gpt_oss_20b-72gpu
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
