#!/usr/bin/env bash
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail
source "${GB200_RUN_DIR:?}/config.sh"
echo "GB200_TRAIN_START job=$SLURM_JOB_ID GBS=512 train_GPUs=32 gen_GPUs=24 seed=$SEED ctx=${CTX_ENGINES}xTP${CTX_TP}/EP${CTX_MEP} gen=${GEN_ENGINES}xTP${GEN_TP} gym_train_concurrency=${GYM_TRAIN_CONCURRENCY:-unset} ctx_max_batch=${CTX_MAX_BATCH:-unset}"
echo "PROFILE=$DGXSYSTEM chunked_offload=$CHUNKED_OPTIMIZER_STATE_OFFLOAD recompute=$RECOMPUTE_GRANULARITY"
cd "$GB200_LLMDIR"
exec bash "$GB200_RUN_DIR/run.gb200.sub"
