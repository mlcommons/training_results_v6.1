# Crusoe-derived 4-node (32 GPU) B300-SXM config for llama2_70b_lora.
# Derived 2026-09-17 from Cisco's v6.0 config_c880_B300_2x8x1xtp1pp1cp1_fp4.sh
# (our config_DGXB300_2x8x1xtp1pp1cp1_fp4.sh): only CP 2 -> 4 and DGXNNODES 2 -> 4
# changed, so DP stays 8 and the global batch stays 8 with MINIBS=1.
# Hyperparameters (MAX_STEPS 550, LR 6e-4, HEALING_ITER 350) are identical to
# NVIDIA's shipped DGXB300 1x8x1 fp4 config and to the RCP GBS-8 record.
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

export MAX_STEPS=550
export LR=0.0006
export MINIBS=1
export CP=4

export MCORE_CUDA_GRAPH=1
export NCCL_NVLS_ENABLE=0
export NUM_WORKERS=4
export HEALING_ITER=350
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False
export VBOOST_VALUE=0

# system parameters
export DGXNNODES=4
export DGXNGPU=8
export WALLTIME_RUNANDTIME=25
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
