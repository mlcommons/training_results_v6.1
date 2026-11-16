source $(dirname ${BASH_SOURCE[0]})/config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh

export MINIBS=480
export TENSOR_MODEL_PARALLEL=2

export RECOMPUTE_MODULES="mlp"

export MLA_DOWN_PROJ_FUSION=False
export FP8_OUTPUT_PROJ=False

export USE_GROUPED_GEMM_FOR_SHARED_EXPERT=False
unset MOE_SHARED_EXPERT_GLU_INTERLEAVE_SIZE

export NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE=0
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1

export MOE_EXPERT_RANK_CAPACITY_FACTOR=8

export NCCL_GRAPH_REGISTER=0
export FP8_DPA=False
export MLPERF_ATTN_PRECISION="bfloat16"

export TRAIN_SRUN_EXTRA="--cpus-per-task=36 --cpu-bind=none"
export BINDCMD_OVERRIDE=/workspace/llm/numabind.sh

export NEXP=${RUN_NEXP:-3}
export WALLTIME_RUNANDTIME=85
export WALLTIME=$((5 + ${NEXP} * ($WALLTIME_RUNANDTIME + 5)))

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
