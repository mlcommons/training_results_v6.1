# GPT-OSS 20B, 16 nodes x 8 GPUs (DGXB200), GBS 64 (TP 1, PP 1, CP 2, DP 64, micro-batch 1), EP 4 inside one node.
# Base: NVIDIA v6.1 config_VR200_18x4x1xtp1pp1cp2ep4_mxfp8.sh (BF16 attention under CP, EP 4, capacity factor 5).
# CP keeps GBS at 64, the largest batch in the RCP table. FP8 parameter gather; NCCL user buffers on.
# LR 4e-4, warmup 32 steps (2048 samples). LR and warmup are unconstrained for gpt_oss_20b (see README_UF.md).
# Eval every 12288 samples: VAL_CHECK_INTERVAL = 12288 / GBS = 192 steps.
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_18x4x1xtp1pp1cp2ep4_mxfp8.sh"
export DGXNNODES=16 DGXNGPU=8
unset SEGMENT                      # NVL72 segment size; not used on HGX B200
export USE_MNNVL=0                 # NVLink only inside a node
export MCORE_BUILD_BIND=1          # writable dir for the NCCL user-buffer allocator build
export MINIBS=1 MICRO_BATCH_SIZE=1
export NCCL_UB_DP=True             # NVIDIA's CP 2 config turns it off
export FP8_PARAM_GATHER=True REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=True
export FP8_DPA=False MLPERF_ATTN_PRECISION=bfloat16
export LR=0.0004 LR_WARMUP_STEPS=32 VAL_CHECK_INTERVAL=192
export FORCE_SUCCESS_STATUS=0 TARGET_LOG_PPL=3.34
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60
export WALLTIME_RUNANDTIME=80
export WALLTIME=$((5 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))
export DGXSYSTEM=DGXB200_16x8x1xtp1pp1cp2ep4_mxfp8_ub_lr4e4
