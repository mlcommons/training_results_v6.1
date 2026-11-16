#!/bin/bash
# Submit NVIDIA's v6.1 Qwen3.5-397B GRPO benchmark (our x86 SIF, NVIDIA code from the image)
# to HiPerGator B200 with Apptainer (pytorch/run_hpg_qwen.sub).
# Usage: ./run_hpg_qwen.sh <config> [extra sbatch args]
#   <config>            config file in this dir, for example config_B200_16x8_t8g8_tp4pp4ep8_4p4d8_mxfp8.sh
#   SMOKE_STEPS=N       N training steps, no deferred evaluation, output in runs_smoke/
#   NEXP=N              runs in the job (default 1), each with a new seed
#   RESERVATION=<name>  add --reservation (for example mlperf-benchmark)
#   CONT, NRL_MEGATRON_CHECKPOINT_DIR, LOGDIR: override the defaults below
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMPL=${HERE}
# Kit root: this file sits in University-of-Florida/benchmarks/qwen35_397b_grpo/implementations/pytorch/
KIT=$(cd "${HERE}/../../../../.." && pwd)
CONFIG=${1:?usage: run_hpg_qwen.sh <config> [extra sbatch args]}
shift
[[ -f "${IMPL}/${CONFIG}" ]] || { echo "ERROR: ${IMPL}/${CONFIG} not found" >&2; exit 1; }

# ---- inputs (Yang's downloads, complete since 2026-10-08) ---------------
QWEN=/blue/rc-rse/mlperf/training/training/qwen35
export CONT=${CONT:-${QWEN}/qwen35_397b_grpo_trtllm-amd_20261006.sif}
export HF_CKPT_PATH=${QWEN}/checkpoints/qwen35-397b-a17b
# Yang's HF -> Megatron conversion (2026-10-08). It has iter_0000000/run_config.yaml, so the
# run only reads it and does not convert again.
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${QWEN}/checkpoints/qwen35-397b-a17b-megatron}
export NEMO_GYM_SWE_TRAIN_DATA_PATH=${QWEN}/dataset/benchmark-r2e-gym-easy/benchmark_r2e_gym_easy_train.jsonl
export NEMO_GYM_SWE_VALIDATION_DATA_PATH=${QWEN}/dataset/benchmark-r2e-gym-easy/benchmark_r2e_gym_easy_val.jsonl
export NEMO_GYM_SWE_SIF_DIR=${QWEN}/dataset/r2egym-sifs-x86_64/images

for f in "${CONT}" "${NEMO_GYM_SWE_TRAIN_DATA_PATH}" "${NEMO_GYM_SWE_VALIDATION_DATA_PATH}" "${HF_CKPT_PATH}/config.json"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
n=$(ls "${HF_CKPT_PATH}"/model.safetensors-*-of-00094.safetensors | wc -l)
[[ "$n" == 94 ]] || { echo "ERROR: HF checkpoint has ${n} of 94 shards" >&2; exit 1; }

# ---- config ---------------------------------------------------------------
# Smoke: no deferred evaluation (set before the config, which sizes WALLTIME from it).
[[ -n "${SMOKE_STEPS:-}" ]] && export OFFLINE_EVAL=0
set +u
source "${IMPL}/${CONFIG}"
set -u
# config_GBS512.sh sets MAX_STEPS=20, so override it after the config.
[[ -n "${SMOKE_STEPS:-}" ]] && export MAX_STEPS="${SMOKE_STEPS}"
export NEXP=${NEXP:-1}

# Read by the MLPerf loggers for submission_org and submission_platform
export MLPERF_SUBMISSION_ORG="University-of-Florida"
export MLPERF_SUBMISSION_PLATFORM="hpg.dgxb200x8"
export MLPERF_CLUSTER_NAME="HiPerGator B200"
export VERIFY_MOUNTS=${VERIFY_MOUNTS:-0}   # expected-mounts.csv lists the aarch64 SIF bundle

_runs=runs
[[ -n "${SMOKE_STEPS:-}" ]] && _runs=runs_smoke
export LOGDIR=${LOGDIR:-${KIT}/${_runs}/qwen35_397b_grpo/${DGXSYSTEM}}
mkdir -p "${LOGDIR}" "${KIT}/runs/slurm"

echo "DGXNNODES=${DGXNNODES} TRAIN=${TRAIN_NODES} GEN=${GEN_NODES} TP=${TP} PP=${PP} EP=${EP}" \
     "MAX_STEPS=${MAX_STEPS} OFFLINE_EVAL=${OFFLINE_EVAL} NEXP=${NEXP} WALLTIME=${WALLTIME} min"
echo "CONT=${CONT}"
echo "LOGDIR=${LOGDIR}"
sbatch -A rc-rse -p hpg-b200 -N "${DGXNNODES}" -t "${WALLTIME}" \
    ${RESERVATION:+--reservation="${RESERVATION}"} \
    -J v61_qwen35 -o "${KIT}/runs/slurm/%x_%j.out" --export=ALL \
    "$@" "${IMPL}/run_hpg_qwen.sub"
