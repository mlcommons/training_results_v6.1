source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params
# 128 per rank x 16 nodes x 8 GPUs = global batch 16384. Same GBS and training
# hyperparameters as config_DGXB200_8x8x256.sh (RCP GBS 16384). GBS 16384, not 8192 or
# 32768: at this node count GBS 8192 leaves too little work per GPU in each step, and
# GBS 32768 needs a first-eval block (41) that no config here has used yet.
export BATCH_SIZE=128

# RCP hyperparameters for GBS 16384
export DENSE_LR=0.000002
export SPARSE_LR=0.000002
export LR_WARMUP_STEPS=24000
export LR_WARMUP_START_LR=0.0
export GRAD_CLIP_NORM=1.0

export START_TS=0
export NUM_TRAIN_TS=299
export EVAL_EVERY_DATA_PCT=0.001
# GBS-DEPENDENT: the GBS-16384 value. The checker requires the first eval at block 31.
export SKIP_EVAL_EPOCH_PCT=0.0303
export AUC_THRESHOLD=0.75

# Embedding planner HBM cap per GPU. The gin default (260 GiB) is for GB300.
# B200 has 179 GiB.
export HBM_CAP_GB=${HBM_CAP_GB:-170}

# Eval loader: 2 batches per worker preloaded, not 8, as in config_DGXB200_8x8x128.sh
# (see the comment there). Not a hyperparameter.
export EVAL_PREFETCH_FACTOR=2

## System run parms
export DGXNNODES=16
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=25  # no full trial yet; 8 nodes at GBS 16384 need 7-13 min
# 30 min prolog headroom: each node copies the dataset to local disk
export WALLTIME=$((30 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
