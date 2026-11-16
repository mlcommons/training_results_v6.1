#!/bin/bash
# 32 HiPerGator B200 nodes (8 GPUs, 178 GiB each; NVLink domain = 1 node): 16 training + 16 generation.
# Same per-node layout as config_B200_16x8_t8g8_tp4pp4ep8_4p4d8_mxfp8.sh with twice the nodes:
# the trainer goes from DP4 to DP8 (same TP4 x PP4 x EP8 replica), generation from 10 to 20 engines.
# The 256 GPUs equal NVIDIA's config_GB300_64x4_t32g32_tp4pp2ep32gtp4_mxfp8.sh (32 + 32 GB300 nodes).
# Port of NVIDIA's config_GB300_14x4_t8g6_tp4pp2ep16_2p4d16_mxfp8.sh (20261006 image):
# MXFP8 forward and backward, BF16 Adam state, chunked optimizer-state CPU offload as in
# NVIDIA's profile, but full recompute instead of NVIDIA's selective recompute (see below).
# Invariant: CTX_ENGINES*CTX_TP + GEN_ENGINES*GEN_TP == GEN_NODES*DGXNGPU  (8*4 + 12*8 = 128)

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/config_optimizer_cpu_offload.sh"
# Full recompute (Megatron "uniform", 1 layer per checkpoint). NVIDIA's selective set keeps
# ~90 GB of activations on the first pipeline stage of a 178 GiB B200 (4 micro-batches x
# 15 layers in flight): smoke job 45462048 hit OOM at 169 GiB allocated in the warmup step.
export ACTIVATION_CHECKPOINTING=true
export RECOMPUTE_GRANULARITY=full
unset RECOMPUTE_MODULES
export MOE_SHARED_EXPERT_OVERLAP=false
export GBS="${GBS:-512}"
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/config_GBS${GBS}.sh"

# config_common.sh sets DGXNGPU=4 (GB300); B200 has 8 GPUs per node.
export DGXNGPU=8

export RECIPE="/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm_disagg_mxfp8.yaml"
export MLPERF_LINEAR_PRECISION="fp8"

export TRAIN_NODES=16
export GEN_NODES=16
export COLOCATED_GENERATION=0

export GYM_EVAL_CONCURRENCY=${GYM_EVAL_CONCURRENCY:-256}

# Trainer: TP4 x PP4 = 16 GPUs per model replica, DP8 over 128 GPUs (16 nodes = 8 node pairs).
# EP8 (ETP1) = the 8 GPUs of one node, so NVIDIA's HybridEP dispatcher works (the image
# builds it for one NVLink domain only). Experts per GPU = 512 / 8 per stage, 15 layers per
# stage: the same expert load per GPU as NVIDIA's PP2 x EP16 GB300 profile.
export TP=${TP:-4}
export ETP=${ETP:-1}
export EP=${EP:-8}
export PP=${PP:-4}
# 60 layers -> 15/15/15/15.
export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-15}

# Generation: 8 TP4 context engines (2 per node, 4 nodes) + 12 TP8 decode engines (1 per node,
# 12 nodes). Every engine stays inside one node's NVLink domain (the 16-node layout, doubled).
export CTX_ENGINES=8
export CTX_TP=4
export CTX_MTP=1
export CTX_MEP=4
export GEN_ENGINES=12
export GEN_TP=8
export GEN_MTP=1
export GEN_MEP=8
# The top-level TRT-LLM configuration describes the widest engine.
export TRTLLM_TP=8
export TRTLLM_MEP=8
# Split the 251 validation prompts into waves of at most 126 prompts.
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-126}
export TRTLLM_GPU_UTIL="${TRTLLM_GPU_UTIL:-0.8}"

# KV-cache transfer (NIXL over UCX) and NCCL: only the 8 GPU-local 400G NDR rails.
# NVIDIA's GB300 setup uses UCX_TLS=all; on hpg-b200 "all" also picks up the HDR/storage HCAs.
export UCX_TLS=rc_x,cuda_copy,cuda_ipc,gdr_copy,sm,self
export UCX_NET_DEVICES=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1
export UCX_TCP_AF_PRIO=inet
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
# "=" prefix = exact name match, so mlx5_1 does not match mlx5_1x.
export NCCL_IB_HCA==mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15
# Full list on purpose: run_hpg_qwen.sub applies its default only when this is empty.
export NRL_CONTAINER_ENV_FORWARD="UCX_TLS,UCX_NET_DEVICES,UCX_TCP_AF_PRIO,UCX_LOG_LEVEL,NCCL_IB_HCA,EXTRA_ARGS,MLPERF_WARMUP_TRAIN_STEPS,MLPERF_WARMUP_TRAIN_TIMEOUT_S,MLPERF_WARMUP,MLPERF_WARMUP_TRAIN,NCCL_NVLS_ENABLE,TRTLLM_HYBRID_DISAGG_REUSE,TLLM_MAMBA_FOLD_SAVE_LAST,TLLM_ADP_PACKED_SYNC,NRL_SKIP_FAILING_EMPTY_ROLLOUT"

# Disagg frontends below the 9000 ephemeral-port floor (NVIDIA default 17300).
export DISAGG_FE_BASE_PORT="${DISAGG_FE_BASE_PORT:-7300}"

# Trainer allocator: the recipe sets expandable_segments:False (288 GB GB300). On 178 GiB B200,
# Yang's first training warmup OOMed with ~48 GB reserved but unallocated (fragmentation).
export EXTRA_ARGS="${EXTRA_ARGS:-} ++policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:True\""

# Untimed MLPerf warmup: keep NVIDIA's synthetic training step (an early memory test, before
# the ~20 min rollout phase); allow the logprob warmup more than the 900 s default
# (574 s on 16 nodes in Yang's runs). A failed rank can leave the others waiting until this limit.
export MLPERF_WARMUP_TRAIN_TIMEOUT_S="${MLPERF_WARMUP_TRAIN_TIMEOUT_S:-1800}"

# Bound refit scratch space and preserve whole-node CPU scheduling.
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.01}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-112}"   # CPUTot on hpg-b200 nodes

# No NVL72 racks and no `sbatch --segment` here. Training placement in node pairs.
export NEMORL_TRAIN_SEGMENT_SIZE="${NEMORL_TRAIN_SEGMENT_SIZE:-2}"
export SEGMENT="${SEGMENT:-2}"

export NEMO_GYM_NUM_WORKERS=16
export NRL_USE_FASTOKENS=1

export DGXNNODES=$((TRAIN_NODES + GEN_NODES))
export DGXSYSTEM="$(basename "$(readlink -f "${BASH_SOURCE[0]}")" | sed 's/^config_//' | sed 's/\.sh$//')"

export WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-120}
export WALLTIME=$((21 + ${NEXP:-1} * (WALLTIME_RUNANDTIME + 5)))
add_grpo_eval_walltime
