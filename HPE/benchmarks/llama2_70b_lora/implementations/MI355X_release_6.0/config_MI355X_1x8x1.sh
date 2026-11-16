#!/bin/bash

export DGXSYSTEM=MI355X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

export PRIMUS_PATH=/workspace/Primus
# /workspace/code/src carries sitecustomize.py, which pins each rank to the NUMA
# node of its GPU when MLPERF_NUMA_BIND=1 (see src/_numa_affinity.py).
export PYTHONPATH="/workspace/code/src:${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-Bridge:${PYTHONPATH}"
export MLPERF_NUMA_BIND=${MLPERF_NUMA_BIND:-0}
export EXP=/workspace/code/conf/llama2_70b_lora_posttrain.yaml
export DATA_PATH=/data

# Packed data paths (inside container)
export PACKED_TRAIN_DATA_PATH=/data/train.npy
export PACKED_VAL_DATA_PATH=/data/validation.npy
export PACKED_METADATA_PATH=/data/packed_metadata.jsonl

# Pretrained Megatron checkpoint (mounted via MODEL -> /ckpt, same as nemo convention)
export PRETRAINED_CHECKPOINT=/ckpt/megatron_checkpoints/Llama-2-70b-hf/iter_0000000

export LR=0.0006

export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_ENABLE_INTERRUPT=0
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export OMP_NUM_THREADS=1

export NVTE_USE_AITER_ROPE=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_USE_RMSNORM_TRITON=0
export USE_TE_SWIGLU=1
export ENABLE_TRANSPOSE_CACHE=0
export NVTE_MXFP4_USE_HADAMARD=${NVTE_MXFP4_USE_HADAMARD:-1}
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

export HEALING_ITER=${HEALING_ITER:-340}
export HEALING_PRECISION=${HEALING_PRECISION:-FP8_DS} 
export PRE_QUANTIZED_MODEL=${PRE_QUANTIZED_MODEL:-True}
export NCCL_MIN_P2P_NCHANNELS=${NCCL_MIN_P2P_NCHANNELS:-32} 
export NCCL_MIN_CTAS=${NCCL_MIN_CTAS:-32} 
export NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-32} 
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} 

export MEGATRON_BRIDGE_LOGGING_LEVEL=50
export PYTHONWARNINGS=ignore
export PRIMUS_LOG_LEVEL=ERROR

export SYNTH_WARMUP_STEPS=5
export SYNTH_WARMUP_VALID_STEPS=5

export ENABLE_MLLOG=1
# Must be exported here: run_with_docker.sh harvests variable NAMES from this file
# to build the docker --env allowlist, so a host-only export would be dropped.
export MLLOG_VERBOSE_LOGS=${MLLOG_VERBOSE_LOGS:-0}
# Keep Primus per-rank logs on the mounted results volume instead of the
# container-local ./output, which is destroyed with the container.
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/results/primus}
# The recipe's hf_path (meta-llama/Llama-2-70b-hf) is a gated repo. Model weights
# come from the converted Megatron checkpoint, so only config+tokenizer are
# needed and those are pre-staged into ${DATADIR}/data/huggingface. Force offline
# so transformers reads that cache instead of issuing an unauthenticated request.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=0.925
export MLLOG_SUBMISSION_BENCHMARK=llama2_70b_lora
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X

export MLLOG_TENSOR_PARALLELISM=1
export MLLOG_PIPELINE_PARALLELISM=1
export MLLOG_CONTEXT_PARALLELISM=1
export MLLOG_EXPERT_PARALLELISM=1
export MLLOG_MICRO_BATCH_SIZE=1
export MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR='mxfp4'

export TP_COMM_OVERLAP=False 
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE

export LORA_A2A=1
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

export TP=1
export PP=1
export CP=1
export SP=False
export VBOOST_VALUE=1
export MBS=1
export MINIBS=1
export SKIP_EVALS=3
export VAL_CHECK_INTERVAL=384
export HYDRA_FULL_ERROR=1

export FP8_DPA=0    
export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True

export FUSED_SOFTMAX=0
export RMSNORM_CAST=0

export PT_TENSOR_VALIDATION=0
export PROFILE_RPD=0

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

export LOGGING_INTERVAL=5000
export FP4=True
export FP4_RECIPE=mxfp4
export MAX_STEPS=550
export NEXP=1

export LOAD_CKPT=True
export MCORE_CUDA_GRAPH=False
export RESET_CG_AFTER_HEALING=False

export RECOMPUTE_GRANULARITY=null
export RECOMPUTE_METHOD=null
export RECOMPUTE_NUM_LAYERS=null

export FP8_ACT=0
export AITER_CONFIG_GEMM_A4W4=/workspace/code/a4w4_tuned_gemms.csv
export AITER_LOG_TUNED_CONFIG=0
export NVTE_FP4_LOG_GEMM_SHAPES=0
export AITER_LOG_LEVEL=ERROR
export AITER_LOG_MORE=0
