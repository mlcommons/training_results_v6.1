#!/usr/bin/env bash
# HPE: one node at global batch 512, the smallest batch with a reference RCP
# (AMD's 1x8x32 config runs batch 256, which has none). Hyperparameters of the
# batch-512 RCP: lr 2e-4, warmup 1600.

source "$(dirname -- "${BASH_SOURCE[0]}")/config_common.sh"

export NNODES=1
export DGXNNODES=${NNODES}
export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-64}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$LOCAL_BATCH_SIZE}
export GLOBAL_BATCH_SIZE=$(( LOCAL_BATCH_SIZE * 8 * GRADIENT_ACCUMULATION_STEPS ))
export LR=${LR:-0.0002}
export WARMUP_STEPS=${WARMUP_STEPS:-1600}
