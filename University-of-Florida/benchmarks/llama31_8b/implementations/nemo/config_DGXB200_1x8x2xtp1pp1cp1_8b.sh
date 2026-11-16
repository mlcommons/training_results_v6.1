# Llama 3.1 8B, 1 node x 8 GPUs (DGXB200).
# From NVIDIA v6.1 config_VR200_2x4x2xtp1pp1cp1_8b.sh: 2 nodes x 4 GPUs = the same 8 GPUs, so the
# per-GPU batch, parallelism and hyperparameters stay the same (GBS 16, LR 0.0004, FP4 + FP8 attention).
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x2xtp1pp1cp1_8b.sh"
export DGXNNODES=1 DGXNGPU=8
export DGXSYSTEM=DGXB200_1x8x2xtp1pp1cp1_8b
