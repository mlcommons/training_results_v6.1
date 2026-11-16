source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params
# 128 per rank x 8 nodes x 8 GPUs = global batch 8192. Same GBS and training
# hyperparameters as config_DGXB200_1x8x1024.sh. Alternative to config_DGXB200_8x8x256.sh
# (GBS 16384): fewer samples to converge (pruned RCP mean 69.1 M, not 83.7 M) but half
# the work per GPU in each step. Faster only if a step takes <= ~0.046 s.
export BATCH_SIZE=128

# RCP hyperparameters for GBS 8192
export DENSE_LR=0.000001
export SPARSE_LR=0.000001
export LR_WARMUP_STEPS=24000
export LR_WARMUP_START_LR=0.0
export GRAD_CLIP_NORM=1.0

export START_TS=0
export NUM_TRAIN_TS=299
export EVAL_EVERY_DATA_PCT=0.001
# GBS-DEPENDENT: the GBS-8192 value. The checker requires the first eval at block 25.
export SKIP_EVAL_EPOCH_PCT=0.025
export AUC_THRESHOLD=0.75

# Embedding planner HBM cap per GPU. The gin default (260 GiB) is for GB300.
# B200 has 179 GiB.
export HBM_CAP_GB=${HBM_CAP_GB:-170}

# Eval loader: 2 batches per worker preloaded, not 8 (the config_common.sh default).
# After each eval the code restarts the eval loader at once (utils.py _do_eval_db), so
# its workers load the next eval's first batches while training runs again. With 8,
# this took CPU from the train workers: ~16 s lost per eval on 8 nodes. With 2 (trial
# job 45250108, 8 nodes): ~4.5 s lost, each eval 5.2 s instead of 4.3 s. Not a
# hyperparameter: the eval set (125 batches per rank on 8 nodes) and the AUC do not change.
export EVAL_PREFETCH_FACTOR=2

## System run parms
export DGXNNODES=8
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=30  # no full trial yet; 8 nodes at GBS 16384 need 7-13 min
# 30 min prolog headroom: each node copies the dataset to local disk
export WALLTIME=$((30 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
