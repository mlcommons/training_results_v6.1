source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

# cudnn-frontend has no Rubin kernel for the fused dense MLP's grouped GEMM + GLU + RHT-amax step
export DENSE_GROUPED_GEMM=False
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=0

export MINIBS=2
export MICRO_BATCH_SIZE=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export LR=0.0004
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768

export DGXNNODES=2
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=60
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export SEGMENT="${SEGMENT:-${DGXNNODES}}"
