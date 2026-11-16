export FP4=True
export FP8=False
export FP8_RECIPE=null
export FP4_RECIPE="nvfp4"
export FP8_PARAM_GATHER=False

# TP overlap not supported yet
export TP_COMM_OVERLAP=False

# FP4 recipe
export NVTE_NVFP4_DISABLE_RHT=0
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0

# FP8 attention recipe
export NVTE_DPA_FP8_RECIPE="Float8CurrentScaling"

# Fused DenseMLP
export USE_TE_OPS=True
export DENSE_GROUPED_GEMM=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP_FC1_GLU_RHT_AMAX_TMEM=1

export MLPERF_LINEAR_PRECISION="nvfp4"
export MLPERF_ATTN_PRECISION="fp8"
export MLPERF_COMM_PRECISION="nvfp4"
