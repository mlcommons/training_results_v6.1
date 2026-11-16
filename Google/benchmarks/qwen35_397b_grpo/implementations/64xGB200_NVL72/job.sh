#!/usr/bin/env bash
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail
source "${GB200_RUN_DIR:?}/config.sh"
echo "GB200_TRAIN_START job=$SLURM_JOB_ID GBS=512 train_GPUs=128 gen_GPUs=128"
cd "$GB200_LLMDIR"
exec bash "$GB200_RUN_DIR/run.gb200.sub"
