# GPT-OSS 20B, 2 nodes x 8 GPUs (DGXB200), GBS 16 (TP 1, PP 1, CP 1, DP 16, micro-batch 1), EP 4 inside one node.
# Base: NVIDIA v6.1 config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh (FP8 attention, NCCL user buffers on).
# EP 4 and expert rank capacity factor 5 as in NVIDIA's EP 4 configs; FP8 parameter gather.
# LR 4e-4, warmup 128 steps: RCP gpt_oss_20b_ref_16 (warmup 2048 samples).
# Eval every 12288 samples: VAL_CHECK_INTERVAL = 12288 / GBS = 768 steps.
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh"
export DGXNNODES=2 DGXNGPU=8
unset SEGMENT                      # NVL72 segment size; not used on HGX B200
export USE_MNNVL=0                 # NVLink only inside a node
export MCORE_BUILD_BIND=1          # writable dir for the NCCL user-buffer allocator build (NCCL_UB_DP=True)
export EXPERT_PARALLEL=4 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4
export MOE_EXPERT_RANK_CAPACITY_FACTOR=5
export FP8_PARAM_GATHER=True REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=True
export MINIBS=1 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=16
export LR=0.0004 LR_WARMUP_STEPS=128 VAL_CHECK_INTERVAL=768
export FORCE_SUCCESS_STATUS=0 TARGET_LOG_PPL=3.34
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60
export WALLTIME_RUNANDTIME=120
export WALLTIME=$((5 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))
export DGXSYSTEM=DGXB200_2x8x1xtp1pp1cp1ep4_mxfp8
