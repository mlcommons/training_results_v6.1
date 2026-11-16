# DeepSeek v3 671B, 32 nodes x 8 GPUs (DGXB200) = 256 GPUs, GBS 15360 (NVIDIA's GBS = the smallest RCP batch
# size; LR and schedule from NVIDIA's config): TP 2 (sequence parallel), PP 16 x VPP 2, EP 8 inside each node,
# DP 8, MINIBS 1920 (GBS = 1920 x DP 8). The config of the submitted set (hpg.dgxb200x8.n32).
# Base: NVIDIA v6.1 config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh (64 GB300 nodes, PP 4 x VPP 4, EP 32
# inside one NVL72 NVLink domain). On HGX B200 the NVLink domain is 1 node and the image's HybridEP has no
# inter-node path, so each EP group of 8 and each TP pair stays inside one node (rank order tp-cp-ep-dp-pp).
# TP 2 as in NVIDIA's v6.0 GB200 config (config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh): sequence
# parallel splits the activations of each micro-batch over 2 GPUs. Expert-DP is 2 (TP 2 x DP 8 / EP 8).
# Real runs need the Megatron-Bridge checkpoint: LOAD_CHECKPOINTS_DIR=<dir with iter_0000000/> (1.4 TB, MLCommons).
source "$(dirname "${BASH_SOURCE[0]}")/config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh"
export DGXNNODES=32 DGXNGPU=8 SEGMENT=1
export USE_MNNVL=0
export TENSOR_MODEL_PARALLEL=2
export PIPELINE_MODEL_PARALLEL=16 INTERLEAVED_PIPELINE=2
# 32 segments: [E t] [t t] x 29 [t] [t m L] = 1 + 58 + 1 + 1 = 61 layers + MTP + loss
# (VPP > 1: NVIDIA's EP all-to-all overlap needs it).
export PIPELINE_LAYOUT="Et*1|(t*2|)*29t*1|tmL"
export EXPERT_PARALLEL=8 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export MINIBS=1920

# PP 16 fills the pipeline one stage after the other, and the first forward of each MoE stage takes ~67 s
# (one-time kernel setup), so a later stage can wait > 600 s (the default timeout) for its first input.
# NVIDIA's mock pipeline compile (init phase, not timed) first runs all stages at the same time with mocked P2P.
export WARMUP_MOCK_PIPELINE_COMPILE_MODE=schedule

# Selective recompute of the dense MLP layers (NVIDIA's GB200 value). pretrain.py copies RECOMPUTE_MODULES
# but never sets recompute_granularity, and Megatron ignores the modules unless it is "selective".
# pretrain.py applies model.architecture_overrides with setattr before its recompute block, so set it there.
# 60 min distributed timeout: the 1.3 TB checkpoint load can keep ranks in a barrier for a long time.
export RECOMPUTE_MODULES="mlp"
export EXTRA_ARGS="dist.distributed_timeout_minutes=60 +model.architecture_overrides.recompute_granularity=selective"

# The MLCommons checkpoint has only iter_0000000/ (its train_state.pt: step 0), but its tracker
# latest_train_state.pt says step 100, so "load the latest" looks for iter_0000100 and fails with
# "Unknown checkpoint format in /checkpoints/iter_0000100". Load iteration 0 explicitly.
: "${LOAD_CHECKPOINT_STEP:=0}"
export LOAD_CHECKPOINT_STEP

# Shared expert at TP 2: the v6.1 fused shared expert (one-group GroupedLinear + ScaledSwiGLU,
# config_common_mxfp8.sh) fails with TP 2 ("scaled activation expects one scale per input row"), so use the
# standard TP shared expert MLP. It reads fc1 as contiguous, but Megatron-Bridge interleaves the shared-expert
# fc1 on load whenever MOE_SHARED_EXPERT_GLU_INTERLEAVE_SIZE is set (checkpointing.py
# _get_model_glu_interleave_sizes), so unset it.
export USE_GROUPED_GEMM_FOR_SHARED_EXPERT=False
unset MOE_SHARED_EXPERT_GLU_INTERLEAVE_SIZE

# MXFP8 params without the overlapped param all-gather: with OVERLAP_PARAM_GATHER=True the first all-gather
# after the checkpoint load uses stale buffer data as weights (eval lm loss 15.62); without it the loaded
# weights are used (5.155; NVIDIA GB200 v6.0 start: 5.16). See README_UF.md.
export OVERLAP_PARAM_GATHER=False

# HybridEP capacity factor 8: the budget per rank is tokens per rank x top-k 8 x factor (token_dispatcher.py:1103).
# The dispatch group is ETP 1 x EP 8 = 8 ranks, so one rank can receive at most 8 x tokens x 8 rows = the budget
# at factor 8, and "over budget" cannot occur. The paged stash size does not grow with the factor
# (experts.py:789: average = budget // factor). A stash overflow still starts NVIDIA's rerun, so part of the
# stash is in pinned host memory (CUDA 1.0, CPU 2).
export MOE_EXPERT_RANK_CAPACITY_FACTOR=8
export MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CUDA=1.0
export MOE_PAGED_STASH_BUFFER_SIZE_FACTOR_CPU=2

# Two patches to the image, bound read-only over the image files (see README_UF.md):
# paged_stash.py: the schedule key uses base 10000 instead of 1000 (MINIBS 1920 is over NVIDIA's limit of
# 999 micro-batches per step), and a validation rerun does not zero the grad buffer.
# callback_warmup.py: the warmup steps zero the grad buffers before each optimizer step, so a NaN grad in the
# warmup cannot change the weights.
_ds_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EXTRA_MOUNTS="${_ds_dir}/patches/megatron/core/transformer/moe/paged_stash.py:/workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/moe/paged_stash.py:ro"
export EXTRA_MOUNTS="${EXTRA_MOUNTS},${_ds_dir}/patches/llm/callback_warmup.py:/workspace/llm/callback_warmup.py:ro"

# HybridEP JIT-compiles its kernels into <HYBRID_EP_CACHE_DIR or the home dir>/.deepep/hybrid_ep/jit/proc-<pid>,
# one new folder per process and job, so they can fill a small home dir. Use node-local /tmp.
export HYBRID_EP_CACHE_DIR=/tmp
# PyTorch NCCL flight recorder: on a timeout each rank dumps its last collectives to /results.
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_FR_DUMP_TEMP_FILE=/results/nccl_trace_rank_

# Run limit: about 48 steps at 174-180 s per step with real weights (expert load imbalance at EP 8), about
# 140 min, plus about 28 min init.
export WALLTIME_RUNANDTIME=240
export DGXSYSTEM=DGXB200_32x8x1920xtp2pp16ep8cp1_mxfp8_full_cg_cf8_noopg
