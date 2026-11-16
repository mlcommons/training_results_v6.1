# Llama 2 70B LoRA, 32 nodes x 8 GPUs (DGXB200), GBS 128, CP 2, NVFP4 + FP8 healing: the October config with borrowed hyperparameters.
# Borrowed group (hyperparameter borrowing, review period): NVIDIA hecate_ngpu512_ngc26.09_nemo (512 GPUs, GBS 64, our code):
# LR 6e-4, 600 training steps (logged opt_base_learning_rate 0.0006, opt_learning_rate_training_steps 600).
# The GBS stays 128 (the rule allows a batch size change; our October GBS is kept because CP stays 2).
# All other logged hyperparameters are already equal (LoRA rank 16, alpha 32, weight decay 1e-4, clip 0.3).
# Software, system, parallelism (TP 1, PP 1, CP 2) and precision stay as in October.
# HEALING_ITER 88 (October value) is kept: the source has no healing (FP8), and 88 < 600 steps.
source "$(dirname "${BASH_SOURCE[0]}")/config_DGXB200_32x8x1xtp1pp1cp2_fp4.sh"
export LR=0.0006 MAX_STEPS=600 HEALING_ITER=88
export DGXSYSTEM=DGXB200_32x8x1xtp1pp1cp2_fp4_borrow
