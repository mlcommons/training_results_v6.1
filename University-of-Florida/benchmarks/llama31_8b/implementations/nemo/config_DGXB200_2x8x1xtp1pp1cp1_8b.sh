# Llama 3.1 8B, 2 nodes x 8 GPUs (DGXB200), GBS 16 (MINIBS 1, micro-batch 1).
# Base: NVIDIA v6.1 config_VR200_2x4x2xtp1pp1cp1_8b.sh. Per-GPU batch 1 at scale, as in NVIDIA's 72-GPU config.
# LR 4e-4, warmup 16 steps: NVIDIA's GBS-16 values = RCP llama31_8b_ref_16 (warmup 256 samples).
# Eval every 12288 samples: VAL_CHECK_INTERVAL = 12288 / GBS = 768 steps.
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x2xtp1pp1cp1_8b.sh"
export DGXNNODES=2 DGXNGPU=8
export MINIBS=1 MICRO_BATCH_SIZE=1
export LR=0.0004 WARMUP_STEPS=16 VAL_CHECK_INTERVAL=768
export WALLTIME_RUNANDTIME=100
export DGXSYSTEM=DGXB200_2x8x1xtp1pp1cp1_8b
