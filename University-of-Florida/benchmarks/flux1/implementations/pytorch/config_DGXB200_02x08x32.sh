# Flux.1, 2 nodes x 8 GPUs (DGXB200): GBS 512 = 32 per GPU (an RCP batch size; LR 0.0002,
# warmup 2400 from NVIDIA's config_common_gbs_512.sh).
# From NVIDIA v6.1 config_GB300_02x04x64.sh (8 GPUs, 64 per GPU). train_flat.py has no gradient
# accumulation, so the per-GPU batch is GBS / GPUs.
source "$(dirname "${BASH_SOURCE[0]}")/config_GB300_02x04x64.sh"
export DGXNNODES=2 DGXNGPU=8
# One NVLink domain per HGX node (GB300 NVL72: SEGMENT=2 nodes per domain)
export SEGMENT=1
export DGXSYSTEM=DGXB200_02x08x32
# x86 workaround: amax_arena.py calls CUDA 13 cudaStreamGetCaptureInfo via ctypes with 6 of its 7
# arguments; on x86-64 that segfaults at step 1. 0 skips the call (it changes only the allocation
# of the amax buffers). See BACKLOG P0-5.
export FLAT_AMAX_ARENA=0
export DATAROOT=/blue/rc-rse/mlperf/training/training/flux/preprocessed/energon
