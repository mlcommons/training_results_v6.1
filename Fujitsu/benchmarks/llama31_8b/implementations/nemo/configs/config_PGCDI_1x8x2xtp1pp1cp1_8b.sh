source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh

export MINIBS=2
export MICRO_BATCH_SIZE=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export LR=0.00045

# Dell's config
export WARMUP_STEPS=16

# For RCP's config
#export WARMUP_STEPS=256
#export VAL_CHECK_INTERVAL=192
export FP8_AMAX_HISTORY=4
export FP8_AMAX_ALGO=max

export TP_COMM_OVERLAP=False
export BUCKET_SIZE=768000000
export USE_TE_OPS=True
export NCCL_NET_PLUGIN=spcx

export DGXNNODES=1
export DGXNGPU=8
export SEGMENT=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=300
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
