# Llama 2 70B LoRA, 1 node x 8 GPUs (DGXB200).
# From NVIDIA v6.1 config_VR200_2x4x1xtp1pp1cp1_fp4.sh: 2 nodes x 4 GPUs = the same 8 GPUs, so the
# per-GPU batch, parallelism and hyperparameters stay the same (GBS 8, LR 0.0006, NVFP4, healing at step 350).
# Needs the v6.1 dataset and checkpoint (see README: the v6.0 copies on HPG do not match).
source "$(dirname "${BASH_SOURCE[0]}")/config_VR200_2x4x1xtp1pp1cp1_fp4.sh"
export DGXNNODES=1 DGXNGPU=8
# B200 (178 GiB) runs out of memory with CP 1 in a forward GEMM in warm-up (kit test job 45106767).
# CP 2 splits each 8192-token sequence over 2 GPUs. MINIBS 2 keeps GBS 8 (= MINIBS x 8 GPUs / CP);
# LR and the other hyperparameters stay NVIDIA's.
export CP=2 MINIBS=2
export DGXSYSTEM=DGXB200_1x8x2xtp1pp1cp2_fp4
