#!/bin/bash
# 4-GPU PRE-VALIDATION submission (single run, NCCL_DEBUG=INFO for TP-group check).
# Run this yourself:  ./submit_4gpu_preval.sh
# After completion:   ./validate_4gpu_prerun.sh <slurm-out-file> <jobid>
set -euo pipefail
cd "$(dirname "$0")"

# DATADIR is the *dataset* location and is deliberately unchanged from v5.1:
# the preprocessed c4 corpus is identical between rounds (see docs/V61_BASELINE.md).
# Only the outputs move to a v6.1 tree so the two rounds never mix.
export DATADIR=/lustre/agent/mltraining_v5.1_llama3_8b
export LOGDIR=/lustre/agent/mltraining_v6.1_llama31_8b/log
export POWERCMDDIR=$PWD
export POWERLOGDIR=${LOGDIR}/power
# MLPerf v6.1 requires 9 new mllog keys. Rather than rebuild the image (which would
# change the training environment and break continuity with the v5.1 validated runs),
# overlay ONLY the patched pretrain.py onto the container's copy. The patch is purely
# additive logging -- see docs/V61_MLLOG_KEYS.md. pretrain.py.container-original holds
# the image's unmodified file for diffing.
export EXTRA_MOUNTS="$PWD/pretrain.py:/workspace/llm/pretrain.py:ro"
export CONT=sbj8388/mlperf-nvidia:training5.1_llama31_8b-pyt
export NEXP=1

# Storage/resource monitoring. Kept ON for the official runs: the 60-step overhead
# smoke (job 25 vs job 23/24) measured step time -0.28%, power inside the baseline
# spread, and sampling gap 4.85% vs 4.91/4.96% -- all within run-to-run variation.
# Measured CPU cost is 0.36% of ONE core out of 192 vCPUs. See docs/SMOKE_RESULTS.md §9.
# MONLOGDIR is OUTSIDE the results tree so monitoring never mixes with submission
# artifacts (package_checker warns about unexpected files under power/).
export MONCMDDIR=$PWD/monitor
export MONLOGDIR=/lustre/agent/mltraining_v6.1_llama31_8b/monitoring
export MON_ENABLE=${MON_ENABLE:-1}
export MON_INTERVAL=${MON_INTERVAL:-10}
export MON_NUMANODE=${MON_NUMANODE:-1}

# Pre-validation only: verify TP groups [0,1],[2,3] and per-rank CPU affinity
# ("Setting affinity for GPU n to 48-71 / 72-95" lines). pyxis inherits job env
# into the container; if no "NCCL INFO" lines appear in the log, append
# NCCL_DEBUG to run.sub's --container-env list.
export NCCL_DEBUG=INFO

source config_NVLink_H100_BF16_RCP_BS32_4GPU.sh

echo "GBS check: MINIBS=${MINIBS} x (DGXNGPU=${DGXNGPU} / TP=${TENSOR_MODEL_PARALLEL}) = $((MINIBS * DGXNGPU / TENSOR_MODEL_PARALLEL)) (must be 32)"
echo
echo "sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub"
JID=$(sbatch --parsable -N "${DGXNNODES}" --time="${WALLTIME}" run.sub)
echo "submitted: job ${JID}"
echo
echo "next steps after the run finishes:"
echo "  ./process_power_log.sh ${JID}"
echo "  ./validate_4gpu_prerun.sh slurm-${JID}.out ${JID}"
