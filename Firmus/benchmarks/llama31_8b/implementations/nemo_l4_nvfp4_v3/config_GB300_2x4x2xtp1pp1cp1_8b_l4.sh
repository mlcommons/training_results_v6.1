# Firmus L4 v3 (llama31_8b, "Kernel opt NVFP4 v3"): NVIDIA's 2-node leaf of the 20261006 recipe (GB300 copy, unchanged) plus
# the switches that turn on the Firmus kernels built into this image (firmus/ak; KERNEL_SUPPLEMENTARY_INFORMATION_L4.md).
# Nothing else of the recipe changes: same hyperparameters, batch size, precision recipe, evaluation and stopping rule.
# Where each kernel lives is set by the image itself (Dockerfile ENV AK_*_DIR=/opt/firmus/ak/...); no directory is set here.
# Every switched-on kernel checks its first eager call against the stock path byte for byte and raises on a mismatch, and
# an extension or kernel file that cannot be loaded raises at import (before run_start).
source $(dirname ${BASH_SOURCE[0]})/config_GB300_2x4x2xtp1pp1cp1_8b.sh

# K1-K4, N1: fused QKV RoPE (Triton) + zero-copy QKV pack + RoPE-backward/QKV-direct passes (ak_l8b_qkvx)
export AK_X12_ROPE=1 AK_X12_TAB_CACHE=1 AK_QKV_PACK=1 AK_QKV_AMAX=1 AK_QKV_GRAD_AMAX=1 AK_QKVX_BWD_VER=34
export AK_QKV_DIRECT=1 AK_QKV_DIRECT_IMPL=cuda AK_QKV_DIRECT_P1=2 AK_QKV_DIRECT_W1=2 AK_QKV_DIRECT_W2=2 AK_QKV_DIRECT_REV=1
export AK_QKV_DIRECT_ASM=0 AK_QKV_DIRECT_P1_REV=0 AK_QKV_DIRECT_P2_REV=1 AK_QKVX_P1_VER=37 AK_QKVX_P2_VER=37
# K5, N2: embedding weight gradient into main_grad (Triton)
export AK_EMB=1 AK_EMB_MAIN_GRAD=1
# N3: TE cross-entropy without the input copy (training and evaluation)
export AK_TCE_OVERWRITE=1 AK_TCEV=1
# N4: gradient 1/DP scaling inside the collective
export AK_GSA=1
# K6, N5: fused AdamW-master step + grad-norm fold + early all-gather + weight-amax epilogue (Triton)
export AK_X9_E0=1 AK_X9_FOLD=1 AK_X9_EARLY_AG=1 AK_X9_WAMAX=1
# K7-K13, N6: NVFP4 SwiGLU/dSwiGLU, residual add + RMSNorm statistics, norm apply + RHT cast, RHT casts, weight amax and
# 2D weight cast, RMSNorm backward + residual add + amaxes + finalize
export AK_NVFP4P=1 AK_NVFP4P_KVER=8 AK_NVFP4P_NORM=1 AK_NVFP4P_NORM_VER=6 AK_NVFP4P_WEIGHTS=amax AK_NVFP4P_BWDNORM=1
export AK_NVFP4P_BWDNORM_VER=7 AK_NVFP4P_ADDNORM=1 AK_NVFP4P_ADDSTATS_VER=6 AK_NVFP4P_RHTAMAX=1 AK_NVFP4P_WAMAX=1
export AK_NVFP4P_WQ=1 AK_NVFP4P_RHTCAST=1 AK_NVFP4P_RHTCAST_VP=1 AK_NVFP4P_NORMCAST=1 AK_NVFP4P_NORMCAST_MODE=apply7
export AK_NVFP4P_NORMCAST_APPLY_VER=7 AK_NVFP4P_W2D=1 AK_NVFP4P_DGFOLD=1 AK_NVFP4P_RHT8=1 AK_NVFP4P_BX=1
# N7: weight gradients stored, not accumulated, into the zeroed grad buffer
export AK_X7_SKIP_GRAD_ZERO=1
# N8, N9: linear_proj keeps its NVFP4 column-wise input; op-fuser weight quantizers write GEMM-swizzled scales
export AK_PROJCOL=1 AK_OPSWZ=1
# K14, N10: FP8 current-scaling quantize for the FP8 attention's q|k|v and dO (ak_l8b_fp8q)
export AK_FP8CS=1 AK_FP8CS_AMAX=1 AK_FP8CS_CAST=4,1,4 AK_FP8CS_AMAX_CFG=4,1,4,4,0,8
# K15, N11: NVFP4 parameter all-gather (ak_l8b_fp4g); the gathered weights are checked at the first optimizer step
export AK_FP4G=1 AK_FP4G_CHECK=first AK_FP4G_POISON=0 AK_FP4G_CBPC=16
# L4 v3: no parameter gather of the forward overlaps compute (each gather finished at its dispatch; same arithmetic)
export AK_FP4G_NOOVERLAP=1
# N12: embedding gathered in [s, b, h]; batched delayed-scaling updates
export AK_SMALLK2=1
# K16, N13: LM head forward and dgrad on pinned cuBLASLt algorithms (ak_l8b_lmh)
export AK_LMH=1
