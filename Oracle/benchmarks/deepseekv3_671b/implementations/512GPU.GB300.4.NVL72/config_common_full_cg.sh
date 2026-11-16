export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
export CUDA_GRAPH_WARMUP_STEPS=2
export OPTIMIZER_CG=True
export ON_DEVICE_CLIP_GRAD=True

# Full-iteration CG configs use allocator stream reuse during graph capture.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"

# Paged stashing is enabled for full-iteration CG configs. Small proxy configs
# can override the capacity factor.
export MOE_PAGED_STASH=True
export MOE_EXPERT_RANK_CAPACITY_FACTOR=4
export MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CUDA=1.5
export MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CPU=1

# perf knobs for TE
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVTE_EXT_MARGIN_SM=16

export WARMUP_TRAIN_STEPS=3
export WARMUP_VALIDATION_STEPS=3

export HIGH_PRIORITY_A2A_COMM_STREAM=True

export SKIP_TRAIN_CG_PARAM_AG=False
