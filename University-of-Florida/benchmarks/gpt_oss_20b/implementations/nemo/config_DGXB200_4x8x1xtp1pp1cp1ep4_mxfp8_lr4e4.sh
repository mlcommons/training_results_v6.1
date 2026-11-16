# GPT-OSS 20B, 4 nodes x 8 GPUs (DGXB200), GBS 32 (TP 1, PP 1, CP 1, DP 32, micro-batch 1), EP 4 inside one node.
# Base: NVIDIA v6.1 config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh (FP8 attention, NCCL user buffers on).
# EP 4 and expert rank capacity factor 5 as in NVIDIA's EP 4 configs; FP8 parameter gather.
# LR 4e-4, warmup 256 steps (8192 samples). LR and warmup are unconstrained for gpt_oss_20b (see README_UF.md).
# Eval every 12288 samples: VAL_CHECK_INTERVAL = 12288 / GBS = 384 steps.
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh"
export DGXNNODES=4 DGXNGPU=8
unset SEGMENT                      # NVL72 segment size; not used on HGX B200
export USE_MNNVL=0                 # NVLink only inside a node
export MCORE_BUILD_BIND=1          # writable dir for the NCCL user-buffer allocator build (NCCL_UB_DP=True)
export EXPERT_PARALLEL=4 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4
export MOE_EXPERT_RANK_CAPACITY_FACTOR=5
export FP8_PARAM_GATHER=True REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=True
export MINIBS=1 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32
export LR=0.0004 LR_WARMUP_STEPS=256 VAL_CHECK_INTERVAL=384
export FORCE_SUCCESS_STATUS=0 TARGET_LOG_PPL=3.34
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60
export WALLTIME_RUNANDTIME=80
export WALLTIME=$((5 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))
export DGXSYSTEM=DGXB200_4x8x1xtp1pp1cp1ep4_mxfp8_lr4e4
