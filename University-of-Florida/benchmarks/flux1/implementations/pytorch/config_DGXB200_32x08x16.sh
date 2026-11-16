# Flux.1, 32 nodes x 8 GPUs (DGXB200): GBS 4096 = 16 per GPU (train_flat.py: per-GPU batch = GBS / GPUs).
# 16 per GPU as in config_DGXB200_08x08x16.sh and config_DGXB200_16x08x16.sh. LR 4e-4, warmup 100 steps:
# the RCP flux_ref_4096 row (v6.1 ships no GBS-4096 file).
source "$(dirname "${BASH_SOURCE[0]}")/config_GB300_02x04x64.sh"
export DGXNNODES=32 DGXNGPU=8
export BATCHSIZE=4096 LEARNING_RATE=0.0004 WARMUP_STEPS=100
# One NVLink domain per HGX node (GB300 NVL72: SEGMENT = nodes per rack domain)
export SEGMENT=1
export DGXSYSTEM=DGXB200_32x08x16
# x86 workaround (BACKLOG P0-5): skip the ctypes call that segfaults on x86
export FLAT_AMAX_ARENA=0
export DATAROOT=/blue/rc-rse/mlperf/training/training/flux/preprocessed/energon
# 10-run set (job 45489684): 16.2-18.3 min per run.
export WALLTIME_RUNANDTIME=40
