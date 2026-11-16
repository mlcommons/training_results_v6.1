#!/bin/bash
#
# Fast throughput probe for Llama 3.1 8B.
#
# Runs a short slice of real training and reports steady-state samples/s measured
# strictly between two evaluations, so startup, JIT warmup and eval cost are all
# excluded. Roughly 7 min per probe instead of a 90 min scored run.
#
# Usage:  probe.sh <label> [VAR=VAL ...]
# Example: probe.sh mbs4 PRIMUS_MICRO_BATCH_SIZE=4

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "${SCRIPT_DIR}"

LABEL="${1:?usage: probe.sh <label> [VAR=VAL ...]}"; shift || true

PROBE_ROOT=${PROBE_ROOT:-/nas/SHARED/MLPERF/probes}
export LOGDIR="${PROBE_ROOT}/${LABEL}"
rm -rf "${LOGDIR}"; mkdir -p "${LOGDIR}"

export MLPERF_DATA_ROOT=${MLPERF_DATA_ROOT:-/lvol0/mlperf/llama31_8b}
export DATADIR=${DATADIR:-${MLPERF_DATA_ROOT}/data}
export MODELDIR=${MODELDIR:-${MLPERF_DATA_ROOT}/model}
export CONT=${CONT:-rocm/amd-mlperf:llama31_8b_training_6.0}

export PYTHONPATH=""
source config_MI355X_1x8x1.sh

# --- probe-only overrides: short run, frequent cheap evals as timing markers ---
export PRIMUS_EVAL_INTERVAL=100
export PRIMUS_EVAL_ITERS=4
export PRIMUS_TRAIN_ITERS=350
export NEXP=1
export CHECK_COMPLIANCE=0

# --- per-probe overrides from the command line ---
for kv in "$@"; do export "${kv?}"; done

echo "=== PROBE ${LABEL} ===" | tee "${LOGDIR}/probe.env"
for kv in "$@"; do echo "  override: ${kv}"; done | tee -a "${LOGDIR}/probe.env"

./run_with_docker.sh > "${LOGDIR}/stdout.log" 2>&1 || echo "launcher exit=$?"

python3 "${SCRIPT_DIR}/probe_report.py" "${LABEL}" "${LOGDIR}"
