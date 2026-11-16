#!/bin/bash
export WARMUP=True
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export DGXNNODES=1
export WALLTIME_MINUTES=50
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))

export NCCL_MIN_P2P_NCHANNELS=32;
export NCCL_MIN_CTAS=32;
export NCCL_NCHANNELS_PER_NET_PEER=32;
export NCCL_NVLS_ENABLE=0

export TP_COMM_OVERLAP=False 
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE

export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export USE_TE_SWIGLU=1

export NVTE_CK_USES_BWD_V3=1  
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0 

export LORA_A2A=1
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

export MAX_STEPS=1024
export TP=1
export PP=1
export CP=1
export SP=False
export VBOOST_VALUE=1
export MBS=1
export LR=0.00055
export MINIBS=1
export SKIP_EVALS=3
export VAL_CHECK_INTERVAL=384
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export FP8_DPA=0    
export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True

export ACG=full && export ACM=block && export ACL=21
export FUSED_SOFTMAX=0
export RMSNORM_CAST=0

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI300X"

export NVTE_USE_RMSNORM_TRITON=1
export ENABLE_TRANSPOSE_CACHE=0
export NVIDIA_PYTORCH_VERSION=25.04
export OMP_NUM_THREADS=1

# --- Krai v6: kernel fixes + no recomputation + packed LoRA dropout + eval trimming; everything above is AMD's 5.1 config ---
export MLPERF_SUBMISSION_ORG=Krai         # submitter in every log
export ACG=null ACM=null ACL=null         # no activation recomputation (Megatron rejects ACL 0 with recompute on)
export FP8_ACT_STORE=1                    # Megatron activation_func_fp8_input_store: SwiGLU input kept in e4m3
export HS_ACT_STORE_FUSED=1               # fix 1: the store is written by the fused Triton SwiGLU
export HS_LORA_EPILOGUE_ADD=1             # fix 2: LoRA up-projection, scale and add in one kernel
export HS_LORA_EPILOGUE_RESIDUAL=1        # fix 2: attention residual folded into the same kernel
export HS_LORA_EPILOGUE_BWD=1             # fix 2: LoRA input gradient folded into the base dgrad
export HS_SWIGLU_BWD_FP8=1                # fix 3a: the SwiGLU backward writes fc1's grad as e5m2
export HS_SWIGLU_FWD_FP8=1                # fix 3b: the SwiGLU forward writes fc2's input as e4m3
export HS_ROPE_FUSED=1                    # fix 4: fused qkv split + RoPE
export HS_TCACHE=qkv,proj                 # fix 6b: cached FP8 weight transposes for qkv and proj
export TE_HIPBLASLT_ALGO_LOAD=/workspace/trainloom_patches/te_algos_fix7.csv  # fix 7: fc2 forward hipBLASLt solution
export HS_QKVNORM_E4M3=1 HS_RESGRAD_FOLD=1 HS_QKVGRAD_E5M2=1  # fixes 8a-c: norm and gradient casts fused
export HS_HOST_SYNC=1                     # fix 11: fewer host syncs
export HS_RMSNORM_GRID=1                  # fix 13: RMSNorm launch grid
export HS_LORA_EPILOGUE_FAST=1            # fix 14: faster LoRA epilogue kernels
export HS_LORA_DGRAD_PACKED_FAST=1        # v5.0 item A: packed-dropout LoRA fold on fix 14's kernel
export HS_GRAD_CAST_ONLY=1                # grad-output cast without the transpose: baked into this image's TE files
export NVTE_CK_HOW_V3_BF16_CVT=2          # round-toward-zero BF16 conversion in the v3 attention backward
export HS_EMPTY_CACHE_AT_EVAL=1           # release the allocator cache around evals
# imported by src/train.py, in this order; eval_trim cuts each validation row after its last loss token
export TRAINLOOM_PY_PATCHES=hs_transpose_cache,hs_act_store,hs_lora_epilogue,lora_dropout_packed,hs_lora_dgrad_packed,hs_rmsnorm_grid,hs_resgrad_fold,hs_qkvnorm_e4m3,hs_host_sync,eval_trim
