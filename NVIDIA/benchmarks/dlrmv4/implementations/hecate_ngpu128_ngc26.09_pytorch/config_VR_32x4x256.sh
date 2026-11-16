source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params
# 256 samples/rank x 32 nodes x 4 GPUs = global batch 32768.
# Preserve the GBS-32768 reference schedule from config_VR_16x4x512.sh.
export BATCH_SIZE=256

# Reference RCP hyperparameters for GBS 32768. The target (post-warmup) LR is
# the ONLY LR the reference allows to be tuned, and it scales linearly with
# global batch: 1e-6 @ 8192, 2e-6 @ 16384, 4e-6 @ 32768. 4e-6 is therefore the
# RCP value for this config -- confirmed against all 20 seeds in
# rcp_logs/gbs_32768/, which log opt_base_learning_rate =
# opt_sparse_base_learning_rate = 4e-06.
export DENSE_LR=0.000004
export SPARSE_LR=0.000004
# Warmup steps and start LR are FIXED (non-tunable) for a submission: the
# reference pins 24000 / 0.0 across all three batch sizes and produced its
# whole 60-run RCP sweep with them. Same as the gin defaults; set explicitly
# so the submission-relevant values are visible in the config rather than
# inherited.
# NOTE: at GBS 32768 this ramp spans 24000*32768 = 786.4M samples (34.3% of an
# epoch), while the reference seeds converge at 98.6M-128.5M samples -- i.e.
# convergence lands ~12.5-16.3% into the ramp. The warmup IS the schedule up
# to the target here, so shortening it is not a free speedup.
export LR_WARMUP_STEPS=24000
export LR_WARMUP_START_LR=0.0
export GRAD_CLIP_NORM=1.0

# Full reference sweep to the AUC target. EVAL_EVERY_DATA_PCT=0.001 is the
# cadence the RCP sweep used; samples-to-converge is quantized to this grid,
# so a coarser value is not comparable against rcp_logs/gbs_32768.
export START_TS=0
export NUM_TRAIN_TS=299
export EVAL_EVERY_DATA_PCT=0.001
# Skip periodic eval until 4.1% of the epoch is trained. GBS-DEPENDENT and
# FIXED (not tunable): the reference pins 0.025 / 0.0303 / 0.041 at GBS
# 8192 / 16384 / 32768, from
#   start_eval_samples  = FLOOR((4480 * GBS + 135331840) / 3) = 94,044,160
#   skip_eval_epoch_pct = start_eval_samples / 2,290,835,423  = 0.041
# (README.MD Sec. 5, "Choosing the eval schedule"). Only valid on the 0.001
# eval grid set above. Resolves to skip_eval_until_step=2867, so the first
# eval runs at step 2870 (eval #41) while the earliest RCP seed crosses AUC
# 0.75 at step 3010 (eval #43, 98.6M samples) -- 2 evals of margin.
# The closed rules recompute this start point from the logged
# global_batch_size and require the first eval_accuracy to land on EXACTLY
# that eval, so any other value fails the compliance checker. Cuts eval from
# ~29% to ~5.5% of end-to-end; does NOT touch the training trajectory, only
# measurement cadence.
export SKIP_EVAL_EPOCH_PCT=0.041
# = config_common.sh; restated because it is the convergence criterion this
# config exists to measure.
export AUC_THRESHOLD=0.75

## System run parms
export DGXNNODES=32
export DGXNGPU=4
export NCCL_PXN_DISABLE=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
# Allocate 32 nodes in segments of 16; the job spans multiple racks.
export SEGMENT="${SEGMENT:-16}"

export WALLTIME_RUNANDTIME=30
# 30 min prolog headroom: cold nodes rsync the ~284 GB dataset to NVMe
export WALLTIME=$((30 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
