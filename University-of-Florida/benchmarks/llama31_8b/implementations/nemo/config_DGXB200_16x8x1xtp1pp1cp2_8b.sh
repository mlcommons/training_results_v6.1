# Llama 3.1 8B, 16 nodes x 8 GPUs (DGXB200), GBS 64 (MINIBS 1, micro-batch 1, CP 2: GBS = 128 GPUs / 2).
# Base: NVIDIA v6.1 config_VR200_2x4x2xtp1pp1cp1_8b.sh. Per-GPU batch 1 at scale, as in NVIDIA's 72-GPU config.
# LR 8e-4, warmup 96 steps: RCP llama31_8b_ref_64 (warmup 6144 samples).
# Eval every 12288 samples: VAL_CHECK_INTERVAL = 12288 / GBS = 192 steps.
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x2xtp1pp1cp1_8b.sh"
export DGXNNODES=16 DGXNGPU=8
export MINIBS=1 MICRO_BATCH_SIZE=1 CONTEXT_PARALLEL=2
export LR=0.0008 WARMUP_STEPS=96 VAL_CHECK_INTERVAL=192
export WALLTIME_RUNANDTIME=35
export DGXSYSTEM=DGXB200_16x8x1xtp1pp1cp2_8b
