# Flux.1, 8 nodes x 8 GPUs (DGXB200): GBS 1024 = 16 per GPU (train_flat.py: per-GPU batch = GBS / GPUs).
# LR 2.5e-4, warmup 800 steps: RCP flux_ref_1024 = NVIDIA's v6.0 config_common_gbs_1024.sh (v6.1 ships no
# GBS-1024 file, so set them here; the gbs_512 defaults would leak through otherwise).
source "$(dirname "${BASH_SOURCE[0]}")/config_GB300_02x04x64.sh"
export DGXNNODES=8 DGXNGPU=8
export BATCHSIZE=1024 LEARNING_RATE=0.00025 WARMUP_STEPS=800
# One NVLink domain per HGX node (GB300 NVL72: SEGMENT = nodes per rack domain)
export SEGMENT=1
export DGXSYSTEM=DGXB200_08x08x16
# x86 workaround (BACKLOG P0-5): skip the ctypes call that segfaults on x86
export FLAT_AMAX_ARENA=0
export DATAROOT=/blue/rc-rse/mlperf/training/training/flux/preprocessed/energon
export WALLTIME_RUNANDTIME=60
