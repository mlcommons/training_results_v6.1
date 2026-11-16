export NEXP=10

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export FP8_DPA=True
export NVTE_DPA_FP8_RECIPE="Float8CurrentScaling"
export NVTE_DPA_FP8_FORMAT="HYBRID"
export NVTE_DPA_FP8CS_O_in_F16=1
export NVTE_DPA_FP8DS_AMAX_ALGO=most_recent
export NVTE_DPA_FP8DS_AMAX_HISTLEN=1
export NVTE_DPA_FP8DS_REDUCE_AMAX=0
export MLPERF_ATTN_PRECISION="fp8"

# Borrowed from University of Florida - batch size 16
export MINIBS=2
export MICRO_BATCH_SIZE=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=1

# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=0
export AVERAGE_IN_COLLECTIVE=True

# Override capacity factor for CG
export MOE_EXPERT_RANK_CAPACITY_FACTOR=1.2

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

# Borrowed hyperparameters from University of Florida
export LR=0.0004
export MIN_LR=4e-05
export VAL_CHECK_INTERVAL=512
export LR_WARMUP_STEPS=128
export OPT_LR_DECAY_STEPS=1199872

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=200
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
