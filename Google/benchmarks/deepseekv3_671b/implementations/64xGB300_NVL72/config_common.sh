# dont download anything
export HF_HUB_OFFLINE=1
export HF_HOME="/workspace/llm/hf_hub/v1"
export TRANSFORMERS_OFFLINE=1

export TOKENIZERS_PARALLELISM=False
# PyTorch ProcessGroupNCCL knob (not an NCCL-library var): create NCCL communicators
# on high-priority CUDA streams. Kept by all other benchmarks; consumed by torch itself.
export TORCH_NCCL_HIGH_PRIORITY=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


export MODEL_SIZE="671b"
export DATASET="8b"
export MOE_AUX_LOSS_COEFF=0.01

# MoE fusion knobs
export MOE_PERMUTE_FUSION=True
export MOE_ROUTER_FUSION=True
export APPLY_ROPE_FUSION=True
export MLA_DOWN_PROJ_FUSION=True

export MOE_ROUTER_DTYPE="bf16"
export BINDCMD="bindpcie --cpu=node"

export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

# fused grouped MLP
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=True

export GRAD_ACCUM_FUSION=True
export AVERAGE_IN_COLLECTIVE=True

export WARMUP_TRAIN_STEPS=2
export WARMUP_VALIDATION_STEPS=2
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export MODEL_TFLOP_PER_SAMPLE=1068

export MLPERF_LINEAR_PRECISION="bfloat16"
export MLPERF_ATTN_PRECISION="bfloat16"
export MLPERF_COMM_PRECISION="bfloat16"

export FP8=False
export FP4=False

# Convergence schedule defaults for production runs; proxy/debug configs override as needed.
export MAX_STEPS=100
export LIMIT_VAL_BATCHES=1
export VAL_CHECK_INTERVAL=1
export LR_WARMUP_ITERS=4

export SKIP_FIRST_EVALS=1