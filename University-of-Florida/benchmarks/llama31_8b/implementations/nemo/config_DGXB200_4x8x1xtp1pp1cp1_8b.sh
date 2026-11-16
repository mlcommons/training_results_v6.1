# Llama 3.1 8B, 4 nodes x 8 GPUs (DGXB200), GBS 32 (MINIBS 1, micro-batch 1).
# Base: NVIDIA v6.1 config_VR200_2x4x2xtp1pp1cp1_8b.sh. Per-GPU batch 1 at scale, as in NVIDIA's 72-GPU config.
# LR 8e-4, warmup 128 steps: RCP llama31_8b_ref_32 (warmup 4096 samples).
# Eval every 12288 samples: VAL_CHECK_INTERVAL = 12288 / GBS = 384 steps.
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x2xtp1pp1cp1_8b.sh"
export DGXNNODES=4 DGXNGPU=8
export MINIBS=1 MICRO_BATCH_SIZE=1
export LR=0.0008 WARMUP_STEPS=128 VAL_CHECK_INTERVAL=384
export WALLTIME_RUNANDTIME=60
export DGXSYSTEM=DGXB200_4x8x1xtp1pp1cp1_8b
