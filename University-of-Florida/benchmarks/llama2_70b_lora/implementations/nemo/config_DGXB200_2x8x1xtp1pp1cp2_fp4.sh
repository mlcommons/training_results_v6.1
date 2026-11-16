# Llama 2 70B LoRA, 2 nodes x 8 GPUs (DGXB200), GBS 8 (MINIBS 1, CP 2: GBS = GPUs / 2), NVFP4 + FP8 healing.
# Base: NVIDIA v6.1 config_VR200_2x4x1xtp1pp1cp1_fp4.sh. B200 (178 GiB) needs CP 2: CP 1 runs out of memory.
# Hyperparameters: NVIDIA's GBS-8 values (LR 6e-4, max steps 550, healing at step 350).
# Eval every 384 samples (VAL_CHECK_INTERVAL counts samples; NVIDIA's value at every scale).
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x1xtp1pp1cp1_fp4.sh"
export DGXNNODES=2 DGXNGPU=8
export CP=2 MINIBS=1
export WALLTIME_RUNANDTIME=45
export DGXSYSTEM=DGXB200_2x8x1xtp1pp1cp2_fp4
