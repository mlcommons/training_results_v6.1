source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params
# 256 per rank x 16 nodes x 4 GPUs = global batch 16384.
# Same GBS and training hyperparameters as config_VR200_4x4x1024.sh,
# spread across four times as many GPUs.
export BATCH_SIZE=256

# Reference RCP hyperparameters for GBS 16384. The target (post-warmup) LR is the
# ONLY LR the reference allows to be tuned, and it scales linearly with global
# batch: 1e-6 @ 8192, 2e-6 @ 16384, 4e-6 @ 32768. 2e-6 is therefore the RCP
# value for this config -- confirmed against all 20 seeds in
# rcp_logs/gbs_16384/, which log opt_base_learning_rate =
# opt_sparse_base_learning_rate = 2e-06.
export DENSE_LR=0.000002
export SPARSE_LR=0.000002
# Warmup steps and start LR are FIXED (non-tunable) for a submission: the
# reference pins 24000 / 0.0 across all three batch sizes and produced its whole
# 60-run RCP sweep with them. Same as the gin defaults; set explicitly so the
# submission-relevant values are visible in the config rather than inherited.
# NOTE: at GBS 16384 this ramp spans 24000*16384 = 393.2M samples (17.2% of an
# epoch), while the reference seeds converge at 78.0M-100.9M samples -- i.e.
# convergence lands ~20-26% into the ramp. The warmup IS the schedule up to the
# target here, so shortening it is not a free speedup.
export LR_WARMUP_STEPS=24000
export LR_WARMUP_START_LR=0.0
export GRAD_CLIP_NORM=1.0

# Full reference sweep to the AUC target. EVAL_EVERY_DATA_PCT=0.001 is the
# cadence the RCP sweep used; samples-to-converge is quantized to this grid, so
# a coarser value is not comparable against rcp_logs/gbs_16384.
export START_TS=0
export NUM_TRAIN_TS=299
export EVAL_EVERY_DATA_PCT=0.001
# Skip periodic eval until 3.03% of the epoch is trained. GBS-DEPENDENT: 0.0303
# is the GBS-16384 value from the upstream fit, and it is only valid on the
# 0.001 eval grid set above. Resolves to skip_eval_until_step=4237, so the first
# eval runs at step 4340 (eval #31) while the earliest RCP seed crosses AUC 0.75
# at step 4760 (eval #34) -- 3 evals of margin (16384 gets an extra one free
# because the fit interpolates below its observed crossing). Cuts eval from ~29%
# to ~6.7% of end-to-end. Does NOT touch the training trajectory, only
# measurement cadence.
export SKIP_EVAL_EPOCH_PCT=0.0303
# = config_common.sh; restated because it is the convergence criterion this
# config exists to measure.
export AUC_THRESHOLD=0.75

## System run parms
export DGXNNODES=16
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=40
# 30 min prolog headroom: cold nodes rsync the dataset to NVMe
export WALLTIME=$((30 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
