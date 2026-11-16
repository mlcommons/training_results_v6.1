# Llama 2 70B LoRA, 16 nodes x 8 GPUs (DGXB200), GBS 64 (MINIBS 1, CP 2: GBS = GPUs / 2), NVFP4 + FP8 healing.
# Base: NVIDIA v6.1 config_VR200_2x4x1xtp1pp1cp1_fp4.sh. B200 (178 GiB) needs CP 2: CP 1 runs out of memory.
# Hyperparameters: LR 5e-4 (the LR of NVIDIA's 72-GPU LoRA config; the LR is unconstrained), max steps 1024.
# Healing step 88, as in the 8- and 32-node configs (5632 samples at GBS 64).
# Eval every 384 samples (VAL_CHECK_INTERVAL counts samples; NVIDIA's value at every scale).
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x1xtp1pp1cp1_fp4.sh"
export DGXNNODES=16 DGXNGPU=8
export CP=2 MINIBS=1
export LR=0.0005 MAX_STEPS=1024 HEALING_ITER=88
export WALLTIME_RUNANDTIME=25
export DGXSYSTEM=DGXB200_16x8x1xtp1pp1cp2_fp4_lr5e4
