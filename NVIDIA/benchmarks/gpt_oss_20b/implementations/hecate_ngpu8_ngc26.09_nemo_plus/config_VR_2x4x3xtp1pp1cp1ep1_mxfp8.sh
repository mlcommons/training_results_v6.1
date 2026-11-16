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
export USE_MNNVL=1
export AVERAGE_IN_COLLECTIVE=True

# 5x oversized activation for CG. Override the env. var set in config_common_cg.sh
export MOE_EXPERT_RANK_CAPACITY_FACTOR=1.2

# Fused router: TE's fused_topk_with_score_function computes scores, top-4, post-top-k softmax, probs and
# routing_map in one kernel (fused backward) instead of the eager topk / softmax / scatter / cast chain.
export MOE_ROUTER_FUSION=True

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

# Single grouped expert weights: one stacked weight and bias parameter per GroupedLinear instead of 32, so the
# fused grouped MLP runs cuDNN's dense (stacked) mode: one grouped quantize per call, no per-expert pointer tables,
# zero-copy bias, wgrad written straight into main_grad. Same math; custom_kernels/sgw_hook.py pads TE's scale buffers.
export MOE_SINGLE_GROUPED_WEIGHT_BIAS=True
# TE enables single_grouped_weight / single_grouped_bias only when this is set; left unset, it silently falls
# back to per-expert weight<i> / bias<i> parameters and the setting above does nothing.
export NVTE_GROUPED_LINEAR_SINGLE_PARAM=1

export MOE_DGLU_STRIP=0

export LR=0.0005
export VAL_CHECK_INTERVAL=512
export LR_WARMUP_STEPS=256

export DGXNNODES=2
export DGXNGPU=4
export SEGMENT=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# 50 min of run_and_time: WALLTIME = 5 + NEXP * 55 = 60 min for one experiment (was 200 -> 210 min)
export WALLTIME_RUNANDTIME=50
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
