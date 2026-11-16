# Flux.1, 1 node x 8 GPUs (DGXB300, b300 node): GBS 512 = 64 per GPU (an RCP batch size; LR 0.0002,
# warmup 2400 from NVIDIA's config_common_gbs_512.sh).
# From NVIDIA v6.1 config_GB300_02x04x64.sh (8 GPUs, 64 per GPU, the same per-GPU batch).
source "$(dirname "${BASH_SOURCE[0]}")/config_GB300_02x04x64.sh"
export DGXNNODES=1 DGXNGPU=8
# One NVLink domain per HGX node (GB300 NVL72: SEGMENT=2 nodes per domain)
export SEGMENT=1
export DGXSYSTEM=DGXB300_01x08x64
# x86 workaround: amax_arena.py calls CUDA 13 cudaStreamGetCaptureInfo via ctypes with 6 of its 7
# arguments; on x86-64 that segfaults at step 1. 0 skips the call (it changes only the allocation
# of the amax buffers). See BACKLOG P0-5.
export FLAT_AMAX_ARENA=0
# NVIDIA's limit for 8 GB300 GPUs is 170 min. No full trial on b300 yet. Estimate: 0.6-0.75 s/step
# (B200 2-node smoke: 0.37 s/step at half the per-GPU batch), so the slowest RCP seed
# (~14,850 steps) + ~28 evals needs ~200-215 min. 300 min, so that a slow run is not stopped.
export WALLTIME_RUNANDTIME=300
# Local copy of training/flux/preprocessed/energon (copy it before the first run)
export DATAROOT=/raid/home/rse-ce/minh/flux_data/energon
