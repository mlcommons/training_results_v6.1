# Flux.1, 16 nodes x 8 GPUs (DGXB200): GBS 2048 = 16 per GPU (train_flat.py: per-GPU batch = GBS / GPUs).
# 16 per GPU as in config_DGXB200_08x08x16.sh. LR 2.5e-4, warmup 0: the RCP flux_ref_2048 row (v6.1 ships
# no GBS-2048 file; NVIDIA's v6.0 config_common_gbs_2048.sh used LR 2.15e-4, warmup 0).
source "$(dirname "${BASH_SOURCE[0]}")/config_GB300_02x04x64.sh"
export DGXNNODES=16 DGXNGPU=8
export BATCHSIZE=2048 LEARNING_RATE=0.00025 WARMUP_STEPS=0
# One NVLink domain per HGX node (GB300 NVL72: SEGMENT = nodes per rack domain)
export SEGMENT=1
export DGXSYSTEM=DGXB200_16x08x16
# x86 workaround (BACKLOG P0-5): skip the ctypes call that segfaults on x86
export FLAT_AMAX_ARENA=0
export DATAROOT=/blue/rc-rse/mlperf/training/training/flux/preprocessed/energon
# No full trial yet. NVIDIA's limit for 18 GB300 nodes (72 GPUs, GBS 1152) is 45 min.
export WALLTIME_RUNANDTIME=45
