source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_full_cg.sh
# Performance profile configs
# source  $(dirname ${BASH_SOURCE[0]})/config_performance_profile.sh

export MINIBS=240  # GBS=15360
export TENSOR_MODEL_PARALLEL=1
export PIPELINE_MODEL_PARALLEL=4
export INTERLEAVED_PIPELINE=4
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=32
export PIPELINE_LAYOUT="Et*4|(t*4|)*14tmL"

# Run with real dataset.
export MOE_ROUTER_FORCE_LOAD_BALANCING=False

export RECOMPUTE_MODULES=""
# disable single-param GEMM for now
# export MOE_SINGLE_GROUPED_WEIGHT=True

# Source after EXPERT_PARALLEL; this common derives NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN from it.
source $(dirname ${BASH_SOURCE[0]})/config_common_moe_flex_hybridep.sh
export MOE_HYBRIDEP_NUM_SMS=64
export MOE_HYBRIDEP_NUM_SMS_PREPROCESSING=108

export MOE_FLEX_DISPATCHER_BACKEND=ncclep
export NCCL_EP_SHUFFLE_SMS=32
export NCCL_EP_PROLOG_EPILOG_SMS=148
export NCCL_EP_PREPROCESS_NUM_SMS=80
export NCCL_EP_HT_EM_PULL_PUSH=1
export NVTE_EP_FUSED_PREPARE_DISPATCH=1
export NCCL_EP_HT_UNFUSED_SYNC=1
export DATA_CREATE_ATTENTION_MASK=False
export DATA_NUM_WORKERS=8
export DATA_PIN_MEMORY=1
export DATA_PREFETCH_FACTOR=32
export MOE_PAGED_STASH_COPY_BLOCK_SIZE=4096
export MOE_PAGED_STASH_COPY_MAX_BLOCKS=8192
# 1F1B overlap
export OVERLAP_MOE_EXPERT_PARALLEL_COMM=True
export USE_DYNAMIC_COMP_STREAM=True
export DELAY_WGRAD_COMPUTE=True

# MXFP8 LM-head output projection
export FP8_OUTPUT_PROJ=True

export DGXNNODES=64
export DGXNGPU=4
export SEGMENT=16

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=30
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
