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


export MINIBS=3
export MICRO_BATCH_SIZE=3
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=1

# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
# Single HGX node: no multi-node NVLink domain (GB300 config uses 1)
export USE_MNNVL=0
export AVERAGE_IN_COLLECTIVE=True

# 5x oversized activation for CG. Override the env. var set in config_common_cg.sh
export MOE_EXPERT_RANK_CAPACITY_FACTOR=1.2

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

export LR=0.0005
export VAL_CHECK_INTERVAL=512
export LR_WARMUP_STEPS=256

# 1 node x 8 GPUs (GB300 config: 2 nodes x 4 GPUs)
export DGXNNODES=1
export DGXNGPU=8

# NUMA binding for this node (6 NUMA nodes, see bind_XE9780LAP.sh)
export EXTRA_MOUNTS="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/bind_XE9780LAP.sh:/workspace/llm/bind_XE9780LAP.sh:ro"
export BINDCMD="bindpcie --cpu=/workspace/llm/bind_XE9780LAP.sh --mem=/workspace/llm/bind_XE9780LAP.sh"
export PMIX_MCA_psec=native
export PMIX_MCA_gds=hash

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=200
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
