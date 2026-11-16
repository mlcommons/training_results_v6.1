#!/usr/bin/env bash
set -euo pipefail
export PATH="$PATH:/opt/slurm/bin"
cd /home/mlperf/dlrmv4/implementation-20260929
export CONT=/data/mlperf/images/dlrmv4-arm-20260929.sqsh
# Stage dataset (incl. hstu_cache) from NFS to node-local NVMe once per job;
# run.sub rsyncs SLOW_DATADIR -> DATADIR on every node before the experiments.
export SLOW_DATADIR=/data/mlperf/dlrmv4/data
export DATADIR=/scratch/local/mlperf/dlrmv4/data
export LOGDIR=/home/mlperf/logs/dlrmv4/config_GB300_16x4x256
export NEXP="${NEXP:-1}"
source config_GB300_16x4x256.sh
export MLPERF_CLUSTER_NAME="gb300-icat"
export SRUN_KILL_ON_BAD_EXIT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${LOGDIR}/nccl_trace_rank_"
export NVIDIA_VISIBLE_DEVICES=all
export MLPERF_SUBMISSION_ORG="Crusoe"
export MLPERF_SUBMISSION_PLATFORM="crusoe_gb300"
NODELIST="${NODELIST:-r204-compute-node-gb300-icat-[000-015]}"
SBATCH_ARGS=(
    -N "${DGXNNODES}"
    --time="${WALLTIME}"
    -A crusoe -p r204 --exclusive
    -w "${NODELIST}"
    --job-name=dlrmv4-64gpu
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
