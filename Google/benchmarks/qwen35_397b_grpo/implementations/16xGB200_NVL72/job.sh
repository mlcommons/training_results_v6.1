#!/usr/bin/env bash
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail
source "${GB200_RUN_DIR:?}/config.sh"
echo "GB200_TRAIN_START job=$SLURM_JOB_ID GBS=512 train_GPUs=32 gen_GPUs=32 seed=$SEED"
echo "PROFILE=$DGXSYSTEM chunked_offload=$CHUNKED_OPTIMIZER_STATE_OFFLOAD recompute=$RECOMPUTE_GRANULARITY"
cd "$GB200_LLMDIR"
exec bash "$GB200_RUN_DIR/run.gb200.sub"
