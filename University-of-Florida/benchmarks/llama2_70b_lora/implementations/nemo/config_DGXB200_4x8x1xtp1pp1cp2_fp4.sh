# Llama 2 70B LoRA, 4 nodes x 8 GPUs (DGXB200), GBS 16 (MINIBS 1, CP 2: GBS = GPUs / 2), NVFP4 + FP8 healing.
# Base: NVIDIA v6.1 config_VR200_2x4x1xtp1pp1cp1_fp4.sh. B200 (178 GiB) needs CP 2: CP 1 runs out of memory.
# Hyperparameters: RCP llama2_70b_lora_ref_16 (LR 4e-4, max steps 1024).
# Healing step 175: NVIDIA heals at step 350 at GBS 8 (2800 samples); the same sample point at GBS 16.
# Eval every 384 samples (VAL_CHECK_INTERVAL counts samples; NVIDIA's value at every scale).
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x1xtp1pp1cp1_fp4.sh"
export DGXNNODES=4 DGXNGPU=8
export CP=2 MINIBS=1
export LR=0.0004 MAX_STEPS=1024 HEALING_ITER=175
export WALLTIME_RUNANDTIME=45
export DGXSYSTEM=DGXB200_4x8x1xtp1pp1cp2_fp4
