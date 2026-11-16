#!/bin/bash
#
# Llama 3.1 8B scored-run driver.
#
# Defaults to the node-local staged dataset. Hyperparameter overrides (LR,
# warmup, micro-batch) are passed as VAR=VAL arguments and applied AFTER the
# system config is sourced, which is required because run_with_docker.sh
# forwards the *current host value* of each name it harvests from that config.
#
# Usage:
#   run.llama31-8b.training.sh <tag> [NEXP] [VAR=VAL ...]
# Example:
#   run.llama31-8b.training.sh lr1e-3 3 PRIMUS_LR=1e-3 PRIMUS_MIN_LR=1e-4

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "${SCRIPT_DIR}"

TAG="${1:-baseline}"; shift || true
NEXP_ARG="${1:-1}"; shift || true

export MLPERF_DATA_ROOT=${MLPERF_DATA_ROOT:-/lvol0/mlperf/llama31_8b}
export DATADIR=${DATADIR:-${MLPERF_DATA_ROOT}/data}
export MODELDIR=${MODELDIR:-${MLPERF_DATA_ROOT}/model}
export LOGDIR=${LOGDIR:-/nas/SHARED/MLPERF/results/llama31_8b/${TAG}}
export CONT=${CONT:-rocm/amd-mlperf:llama31_8b_training_6.0}

[[ -e "${DATADIR}/c4-train.en_6_text_document.bin" ]] || { echo "FATAL: dataset missing in ${DATADIR}" >&2; exit 1; }
[[ -e "${MODELDIR}/tokenizer.json" ]] || { echo "FATAL: tokenizer missing in ${MODELDIR}" >&2; exit 1; }

mkdir -p "${LOGDIR}"
chmod -R 777 "${LOGDIR}" 2>/dev/null || true

export PYTHONPATH=""
source config_MI355X_1x8x1.sh

export NEXP="${NEXP_ARG}"
export CHECK_COMPLIANCE=${CHECK_COMPLIANCE:-1}

# Per-experiment overrides, applied after the config so they win.
for kv in "$@"; do export "${kv?}"; done

# min_lr must stay at exactly 0.1 * base_lr (MLPerf fixed hyperparameter) and
# decay steps must stay at 1_200_000 - warmup (machine-checked by the compliance
# checker), so derive them rather than trusting the caller.
export PRIMUS_MIN_LR=$(python3 -c "print('%.10g' % (float('${PRIMUS_LR}') * 0.1))")
export PRIMUS_EVAL_INTERVAL=$(( EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE ))
export PRIMUS_LR_DECAY_ITERS=$(( PRIMUS_TRAIN_ITERS - PRIMUS_LR_WARMUP_ITERS ))

echo "=== llama31_8b run: tag=${TAG} nexp=${NEXP} ==="
echo "    lr=${PRIMUS_LR} min_lr=${PRIMUS_MIN_LR} warmup=${PRIMUS_LR_WARMUP_ITERS} decay=${PRIMUS_LR_DECAY_ITERS}"
echo "    gbs=${PRIMUS_GLOBAL_BATCH_SIZE} mbs=${PRIMUS_MICRO_BATCH_SIZE} eval_interval=${PRIMUS_EVAL_INTERVAL}"
echo "    numa_bind=${MLPERF_NUMA_BIND} data=${DATADIR}"
echo "    logdir=${LOGDIR}"

./run_with_docker.sh
