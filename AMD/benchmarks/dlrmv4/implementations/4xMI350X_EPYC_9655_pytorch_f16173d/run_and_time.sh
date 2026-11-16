#!/usr/bin/env bash
# MLPerf Training reference script: run the benchmark and report wall-clock time.
#
# Runs the full-reference HSTU / yambda-5b streaming train+eval sweep to the
# MLPerf quality target (eval AUC >= 0.75) and prints the elapsed time of the
# timed region. This is the canonical single-host (8-GPU) entry point; for
# multi-node SLURM launches use scripts/launch_slurm.sh (which calls into the
# same trainer).
#
# Usage:
#   DLRM_DATA_PATH=/path/to/dlrm_data ./run_and_time.sh
#
# Env (run shape / cadence -- defaults are the FULL reference sweep):
#   DLRM_DATA_PATH        data root (required).
#   SEED                  RNG seed (default -1: a fresh random seed per run).
#   START_TS / NUM_TRAIN_TS   window range (default 0 / 299 = full sweep).
#   EVAL_EVERY_DATA_PCT   eval cadence as a fraction of train data (default 0.001).
#   AUC_THRESHOLD         convergence target (default 0.75).
#   GPUS_PER_NODE         GPUs on this host (default 8).
#   RUN_NAME              results dir name under results/ (default reference_run).
set -euo pipefail

: "${DLRM_DATA_PATH:?Set DLRM_DATA_PATH to the data root}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ---- Reference run shape (full sweep to the quality target) -----------------
export SEED="${SEED:--1}"
export START_TS="${START_TS:-0}"
export NUM_TRAIN_TS="${NUM_TRAIN_TS:-299}"
export NUM_TRAIN_BATCHES="${NUM_TRAIN_BATCHES:-0}"
export NUM_EVAL_BATCHES="${NUM_EVAL_BATCHES:-0}"
export EVAL_EVERY_N_WINDOWS="${EVAL_EVERY_N_WINDOWS:-0}"
export EVAL_EVERY_DATA_PCT="${EVAL_EVERY_DATA_PCT:-0.001}"
export AUC_THRESHOLD="${AUC_THRESHOLD:-0.75}"
export RUN_NAME="${RUN_NAME:-reference_run}"

# ---- Single-host distributed topology (override for multi-node) -------------
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-$((NNODES * GPUS_PER_NODE))}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ---- MLPerf compliance logging ----------------------------------------------
export MLPERF_LOGGING="${MLPERF_LOGGING:-1}"
export MLPERF_LOG_PATH="${MLPERF_LOG_PATH:-${REPO_ROOT}/results/${RUN_NAME}/mlperf/yambda_5b_mlperf.log}"
# The cluster nodes are MI350X (1000 W). Override for other systems; the value must
# match the submission's system description.
export MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-MI350X}"
mkdir -p "$(dirname "${MLPERF_LOG_PATH}")"

# ---- Cache flush (MLPerf rules: Data State at Start of Run) -------------------
# "You must flush the cache or restart the system prior to benchmarking."
# Needs root or passwordless sudo on the host; the trainer logs cache_clear with
# the outcome. CLEAR_CACHES=0 skips it for non-timed runs.
export MLPERF_CACHE_CLEARED=0
if [ "${CLEAR_CACHES:-1}" = 1 ]; then
    if [ "$(id -u)" = 0 ]; then _sysctl=(/sbin/sysctl); else _sysctl=(sudo -n /sbin/sysctl); fi
    if sync && "${_sysctl[@]}" -q vm.drop_caches=3; then
        export MLPERF_CACHE_CLEARED=1
    else
        echo "ERROR: page-cache flush failed (needs root/sudo); set CLEAR_CACHES=0 for a non-timed run" >&2
        exit 1
    fi
fi

# ---- Timed region -----------------------------------------------------------
# Pull the start timestamp into a clear region per the MLPerf run_and_time.sh idiom.
start=$(date +%s)
echo "STARTING TIMING RUN AT $(date -u '+%Y-%m-%d %r')"

python -m generative_recommenders.dlrm_v4.train.train_ranker \
    --dataset yambda-5b \
    --mode streaming-train-eval

end=$(date +%s)
result=$(( end - start ))
echo "ENDING TIMING RUN AT $(date -u '+%Y-%m-%d %r')"
echo "RESULT,recommendation_hstu_yambda_5b,${SEED},${result},$(whoami),$(date -u '+%Y-%m-%d %r')"
