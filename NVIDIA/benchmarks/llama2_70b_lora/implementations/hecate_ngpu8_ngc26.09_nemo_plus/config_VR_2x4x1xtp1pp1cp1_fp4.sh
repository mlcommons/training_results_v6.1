#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=550
export LR=0.0006
export MINIBS=1
export CP=1
export CUDA_GRAPH_IMPLEMENTATION="full_iteration"
export CUDA_GRAPH_MODULES=""
export NUM_WORKERS=4

export HEALING_ITER=350

# Keep the pinned FP8 healing weights on the NUMA node of each GPU.
# Without this, two GPUs per node copy them at about 19 GB/s at HEALING_ITER.
export BINDCMD="bindpcie --cpu=node"

# Bit-identical NVFP4 kernel and TE patches (lora_kernels/, te_patches/). The losses and the
# eval losses stay the same, and the run gets about 12% faster on Rubin.
export LORA_SWIZZLE_WEIGHT_SCALES=1
export LORA_PREBUILD_FP8_NORM_PLAN=1
export NVTE_LORA_PATCH_BATCH_DPA_AMAX=1
export LORA_PATCH_FAST_ROPE=1
export LORA_PATCH_FAST_KERNELS=1
export LORA_SWIGLU_SR_TWO_PASS=1
export LORA_PATCH_FAST_DROPOUT=1
export LORA_PATCH_FP8_ROPE=1
export LORA_PATCH_AMAX_PRODUCERS=1
# Fused LoRA branch kernels. This one changes the fp32 sum order of the LoRA dots, so the bits
# differ from TE. A 20-seed study kept the mean samples to converge (3130 in both arms).
export LORA_PATCH_FUSED_LORA=1
# Fused residual add + RMSNorm + NVFP4 quantize, and the fused norm backward with the residual
# grad. The rstd sum order and the grad sum change, so the bits differ from TE (20 seeds: mean
# 3206 samples to converge with the LoRA dropout in eval, 3130 with no dropout in eval).
export LORA_PATCH_RESNORM=1
# The quantize kernels also write the alpha of the next NVFP4 GEMM, and the GEMM runs with no
# alpha kernel (the same bits as TE).
export LORA_PATCH_GEMM_ALPHA=1
# The attention returns its FP8 output O; the proj quantize and the LoRA branch read the FP8
# bytes and dequantize them in registers (the same bits as TE).
export LORA_PATCH_FP8_O=1

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=2
export DGXNGPU=4
export WALLTIME_RUNANDTIME=35
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export SEGMENT="${SEGMENT:-${DGXNNODES}}"
