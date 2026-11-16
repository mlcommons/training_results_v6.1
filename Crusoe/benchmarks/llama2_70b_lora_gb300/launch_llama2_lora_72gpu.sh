#!/usr/bin/env bash
set -euo pipefail
export PATH="$PATH:/opt/slurm/bin"
cd /home/mlperf/llama2_70b_lora/implementation
export CONT=/data/mlperf/images/llama2_70b_lora-arm.sqsh
export DATADIR=/data/mlperf/llama2_70b_lora/data/gov_report
export MODEL=/data/mlperf/llama2_70b_lora/model
export LOGDIR=/home/mlperf/logs/llama2_70b_lora/config_GB300_18x4x1xtp1pp1cp8
export NEXP="${NEXP:-1}"
source config_GB300_18x4x1xtp1pp1cp8.sh
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
    --job-name=llama2_70b_lora-72gpu
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
