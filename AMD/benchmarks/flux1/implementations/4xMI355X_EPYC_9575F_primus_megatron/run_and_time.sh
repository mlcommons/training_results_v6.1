#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# One timed MLPerf run, executed inside the container on every node.
#
# Expects the config_<MLPERF_CONFIG>.sh variables plus, from the launcher:
#   NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE   distributed layout
#   SEED RUN_INDEX                                           this run
# The timed region, the page-cache drop, the MLPerf log and the compliance check are
# implemented by Primus's examples/mlperf/flux1/megatron/run_and_time.sh, which this
# script calls after checking the configuration.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

: "${MLPERF_CONFIG:?source config_<MLPERF_CONFIG>.sh}"
: "${NNODES:?}" "${NODE_RANK:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}" "${SEED:?}" "${RUN_INDEX:?}"
: "${GPUS_PER_NODE:=${NUM_GPUS}}"
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE

# ---- The configuration must describe what will actually run -----------------------------
recipe_gbs=$(grep -oP 'global_batch_size: \$\{PRIMUS_GLOBAL_BATCH_SIZE:\K[0-9]+' "${EXP}")
recipe_mbs=$(grep -oP '^\s*micro_batch_size: \K[0-9]+' "${EXP}")
gbs=${PRIMUS_GLOBAL_BATCH_SIZE:-$recipe_gbs}
if [[ -z "${MLPERF_DEV_OVERRIDES:-}" ]]; then
    for v in PRIMUS_GLOBAL_BATCH_SIZE PRIMUS_LR PRIMUS_LR_WARMUP_ITERS PRIMUS_EVAL_INTERVAL PRIMUS_TRAIN_ITERS \
             PRIMUS_MXFP6_FWD_FP4_HEAL_ITER PRIMUS_MXFP6_FWD_FP4_HEAL_LAYERS_PER_ITER PRIMUS_MXFP6_FWD_FP4_HEAL_TARGET; do
        [[ -z "${!v:-}" ]] || { echo "[run_and_time] ERROR: $v is set; scored runs use the recipe's values" >&2; exit 1; }
    done
fi
# A scored run takes one microbatch per step. Development runs (MLPERF_DEV_OVERRIDES) may
# ask for accumulation, to start the 4-node recipe unchanged on fewer GPUs and check its
# MLPerf log. The model then refuses to build: the MXFP6 fused weight-gradient path writes
# gradients in place and cannot accumulate microbatches.
per_step=$(( NNODES * GPUS_PER_NODE * recipe_mbs ))
if [[ -n "${MLPERF_DEV_OVERRIDES:-}" ]] && (( gbs % per_step == 0 )); then
    per_step=${gbs}
fi
if (( gbs != per_step )) || (( gbs != GLOBAL_BATCH_SIZE )) || (( recipe_mbs != MICRO_BATCH_SIZE )); then
    echo "[run_and_time] ERROR: GBS ${gbs} != ${NNODES} nodes x ${GPUS_PER_NODE} GPUs x MBS ${recipe_mbs}" \
         "(config: GBS ${GLOBAL_BATCH_SIZE}, MBS ${MICRO_BATCH_SIZE}); one microbatch per step is required" >&2
    exit 1
fi

# ---- Results: one MLPerf log per run, written by global rank 0 ---------------------------
export PRIMUS_SEED="${SEED}"
export MLLOG_OUTPUT_FILE="/results/result_${RUN_INDEX}.txt"
export RESULTS_DIR="/results/raw/${MLPERF_CONFIG}_run${RUN_INDEX}_node${NODE_RANK}"
export CHECK_COMPLIANCE=$([[ "${NODE_RANK}" == 0 ]] && echo 1 || echo 0)
export MLPERF_RULESET=6.1.0

# The base image sets the TransformerEngine attention-backend switches; Megatron rejects them
# with its default (auto) attention backend, so they are removed for the run.
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

cd "${PRIMUS_PATH}"   # the recipe's relative paths resolve against the Primus checkout
exec bash "${PRIMUS_PATH}/examples/mlperf/flux1/megatron/run_and_time.sh"
