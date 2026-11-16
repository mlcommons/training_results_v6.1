# Llama 3.1 8B, 2 nodes x 8 GPUs (DGXB200), GBS 32 (MINIBS 2, micro-batch 2): the October config with borrowed hyperparameters.
# Borrowed group (hyperparameter borrowing, review period): HPE HPE-Cray-XD690-B300-SXM6-288GB_n2 (16 GPUs, GBS 32):
# GBS 32, LR 8e-4, end LR 8e-5, warmup 64 steps (logged global_batch_size 32, opt_base_learning_rate 0.0008,
# opt_end_learning_rate 8e-05, opt_learning_rate_warmup_steps 64; weight decay 0.1 and clip 1.0 are already equal).
# Local batch 2 per GPU as in our 1-node config (the rule allows local and global batch size changes).
# Eval every 12288 samples: VAL_CHECK_INTERVAL = 12288 / GBS = 384 steps.
# Software, system, parallelism (TP 1, PP 1, CP 1) and precision stay as in October.
source "$(dirname "${BASH_SOURCE[0]}")/config_DGXB200_2x8x1xtp1pp1cp1_8b.sh"
export MINIBS=2 MICRO_BATCH_SIZE=2
export LR=0.0008 WARMUP_STEPS=64 VAL_CHECK_INTERVAL=384
export DGXSYSTEM=DGXB200_2x8x2xtp1pp1cp1_8b_borrow
