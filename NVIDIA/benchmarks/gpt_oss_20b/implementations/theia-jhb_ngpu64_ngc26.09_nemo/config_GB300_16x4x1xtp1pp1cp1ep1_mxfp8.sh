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


export MINIBS=1
export MICRO_BATCH_SIZE=1
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=1

# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=1
export AVERAGE_IN_COLLECTIVE=True

# Keep the CUDA-graph expert capacity factor at 2 (the common default).
export MOE_EXPERT_RANK_CAPACITY_FACTOR=2

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

export LR=0.001
export VAL_CHECK_INTERVAL=192
export LR_WARMUP_STEPS=192

export DGXNNODES=16
export DGXNGPU=4
export SEGMENT=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=40
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
