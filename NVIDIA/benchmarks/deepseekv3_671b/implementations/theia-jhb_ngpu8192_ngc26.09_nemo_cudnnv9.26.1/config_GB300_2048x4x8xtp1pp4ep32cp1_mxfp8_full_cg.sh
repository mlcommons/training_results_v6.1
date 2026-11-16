source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_full_cg.sh
# Performance profile configs
# source  $(dirname ${BASH_SOURCE[0]})/config_performance_profile.sh

export MINIBS=8  # GBS=16384
export TENSOR_MODEL_PARALLEL=1
export PIPELINE_MODEL_PARALLEL=4
export INTERLEAVED_PIPELINE=4
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=32
export PIPELINE_LAYOUT="Et*4|(t*4|)*14tmL"
export EVAL_CONTEXT_PARALLEL_SIZE=4
export EVAL_USE_MINIMAL_SCHEDULE_WITH_PADDING=True
export WARMUP_MOCK_PIPELINE_COMPILE_MODE=schedule

# Run with real dataset.
export MOE_ROUTER_FORCE_LOAD_BALANCING=False

export RECOMPUTE_MODULES=""
# disable single-param GEMM for now
# export MOE_SINGLE_GROUPED_WEIGHT=True

# Source after EXPERT_PARALLEL; this common derives NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN from it.
source $(dirname ${BASH_SOURCE[0]})/config_common_moe_flex_hybridep.sh

# 1F1B overlap
export OVERLAP_MOE_EXPERT_PARALLEL_COMM=True
export USE_DYNAMIC_COMP_STREAM=True

export DELAY_WGRAD_COMPUTE=True

# MXFP8 LM-head output projection
export FP8_OUTPUT_PROJ=True

export DGXNNODES=2048
export DGXNGPU=4
export SEGMENT=16

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

