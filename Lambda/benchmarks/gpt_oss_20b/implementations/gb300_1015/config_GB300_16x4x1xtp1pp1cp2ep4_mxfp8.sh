source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh


export MINIBS=1
export MICRO_BATCH_SIZE=1
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=2
export EXPERT_PARALLEL=4
# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=1

# 5x oversized activation for CG. Override the env. var set in config_common_cg.sh
export MOE_EXPERT_RANK_CAPACITY_FACTOR=5

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

# Disable NCCL_UB
export NCCL_UB_DP=False

export LR=0.0004
export VAL_CHECK_INTERVAL=384
export LR_WARMUP_STEPS=256

# yqb01: on driver 580 (+CUDA forward compat) HybridEP intermittently deadlocked at the first step (~31% of 16-node runs);
# a 64-SM HybridEP / 4-8 NCCL CTA budget reduced it to ~10%. After upgrading driver+IMEX to 615.71.09 (what this
# container is built for), NVIDIA's defaults (128 SMs, 16-32 CTAs) ran 12/12 clean at 0.157 s/step, so the override is removed.

export DGXNNODES=16
export DGXNGPU=4
export SEGMENT=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=40
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
