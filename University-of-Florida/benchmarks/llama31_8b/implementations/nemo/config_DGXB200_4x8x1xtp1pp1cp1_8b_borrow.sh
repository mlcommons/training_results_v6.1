# Llama 3.1 8B, 4 nodes x 8 GPUs (DGXB200), GBS 32: the October config with borrowed hyperparameters.
# Borrowed group (hyperparameter borrowing, review period): HPE HPE-GB300-ngpu32_ngc26.09 (32 GPUs, GBS 32):
# LR 8e-4, end LR 8e-5, warmup 64 steps (logged opt_base_learning_rate 0.0008, opt_end_learning_rate 8e-05,
# opt_learning_rate_warmup_steps 64; weight decay 0.1 and clip 1.0 are already equal). Only the warmup changes against October.
# Software, system, parallelism and precision stay as in October.
source "$(dirname "${BASH_SOURCE[0]}")/config_DGXB200_4x8x1xtp1pp1cp1_8b.sh"
export LR=0.0008 WARMUP_STEPS=64
export DGXSYSTEM=DGXB200_4x8x1xtp1pp1cp1_8b_borrow
