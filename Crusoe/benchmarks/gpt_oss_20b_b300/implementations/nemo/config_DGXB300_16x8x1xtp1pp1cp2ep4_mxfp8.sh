# Derived from config_DGXB300_8x8x1xtp1pp1cp2ep4_mxfp8.sh
# (NVIDIA shipped, 8 nodes x 8 GPU = 64 GPU) by doubling DGXNNODES only,
# with TP/PP/CP/EP/MINIBS held fixed. GBS=64 (RCP-backed: yes -- exact
# match to rcps_gpt_oss_20b.json "gpt_oss_20b_ref_64", BS=64).
# HP-table values used: LR / LR_WARMUP_STEPS taken from NVIDIA's own
# GBS=64 shipped config (config_GB300_128x4x1xtp2pp1cp4ep8.sh, the only
# shipped config that lands on GBS=64), NOT from the RCP json's AMD
# reference Hyperparams block (opt_base_learning_rate=1e-3 there is
# AMD's own reference tuning on 1xMI355X, not what NVIDIA's Megatron
# implementation uses -- the RCP json's Hyperparams are provided for
# reference/interpolation only; per training_rules.adoc, gpt_oss_20b's
# opt_base_learning_rate and opt_learning_rate_warmup_steps are both
# "unconstrained", so a submitter's own validated values are compliant).
# VAL_CHECK_INTERVAL is rule-derived (not tuned): training_rules.adoc /
# implementations/nemo/README.md section 5 fixes evaluation frequency at
# every 12,288 sequences, i.e. VAL_CHECK_INTERVAL = 12288 / GBS = 192.
#
# Scale: 16 nodes x 8x B300 GPUs = 128 GPUs. NVLink domain = 8 GPUs
# (whole node); EP=4 fits entirely inside one node -> USE_MNNVL=0.

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
export USE_MNNVL=0

# 5x oversized activation for CG (same EP=4 value as the shipped 8-node
# / 18-node EP=4 configs). Override the env. var set in config_common_cg.sh
export MOE_EXPERT_RANK_CAPACITY_FACTOR=5

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

export LR=0.00052
export VAL_CHECK_INTERVAL=192
export LR_WARMUP_STEPS=32

export DGXNNODES=16
export DGXNGPU=8

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Walltime budget: init alone is 5-6 min, projected train ~15.6 min.
export WALLTIME_RUNANDTIME=40
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
