# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Keep optimizer state and master weights in pinned CPU memory between updates,
# then update them on the GPU in chunks. This cannot run with the other CPU-offload
# mode, so OPTIMIZER_CPU_OFFLOAD stays disabled.
export OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-false}
export CHUNKED_OPTIMIZER_STATE_OFFLOAD=${CHUNKED_OPTIMIZER_STATE_OFFLOAD:-true}
export OPTIMIZER_STATE_OFFLOAD_CHUNK_SIZE_MB=${OPTIMIZER_STATE_OFFLOAD_CHUNK_SIZE_MB:-256}
export OPTIMIZER_STATE_OFFLOAD_FRACTION=${OPTIMIZER_STATE_OFFLOAD_FRACTION:-1.0}
