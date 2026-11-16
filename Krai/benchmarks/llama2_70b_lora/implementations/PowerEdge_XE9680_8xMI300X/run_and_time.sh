#!/bin/bash
# Run NEXP training runs to the target (eval loss <= 0.925) and write result_<i>.txt to $LOGDIR.
#   NEXP=10 DATADIR=/data/mlperf_llama2 LOGDIR=./results ./run_and_time.sh
# Every setting is baked into the image: ../nemo/config_MI300X_1x8x1.sh (AMD's config plus the block at its end).
set -euo pipefail
IMPL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo" && pwd)"
export DGXSYSTEM=MI300X_1x8x1
export CONT="${CONT:-krai4ai/mlperf:training_v6.1-llama2_70b_lora-mi300x}"
export DATADIR="${DATADIR:-/data/mlperf_llama2}"
export LOGDIR="${LOGDIR:-$PWD/results}"
export NEXP="${NEXP:-10}"
exec bash "$IMPL/run_with_docker.sh"
