# GPT-OSS 20B, 8 nodes x 8 GPUs (DGXB200), GBS 64: the October config with borrowed hyperparameters.
# Borrowed group (hyperparameter borrowing, review period; review bot issue #61): NVIDIA hecate_ngpu64_ngc26.09_nemo
# (64 GPUs, GBS 64): LR 5.2e-4, end LR 5.2e-5, warmup 32 steps (logged opt_base_learning_rate 0.00052,
# opt_end_learning_rate 5.2e-05, opt_learning_rate_warmup_steps 32; weight decay 0.1 and clip 1.0 are already equal).
# Crusoe crusoe_b300_n16_ngc26.04_nemo uses the same values on the same pre-fix code as ours.
# Only the LR changes against October (4e-4). MIN_LR is not set, so the code uses LR / 10 = 5.2e-5, as the source.
# Software, system, parallelism and precision stay as in October.
source "$(dirname "${BASH_SOURCE[0]}")/config_DGXB200_8x8x1xtp1pp1cp1ep4_mxfp8_lr4e4.sh"
export LR=0.00052 LR_WARMUP_STEPS=32
export DGXSYSTEM=DGXB200_8x8x1xtp1pp1cp1ep4_mxfp8_borrow
