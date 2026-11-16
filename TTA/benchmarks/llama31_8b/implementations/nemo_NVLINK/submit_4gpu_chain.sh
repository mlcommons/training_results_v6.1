#!/bin/bash
# 4-GPU official 10-run chain (afterany dependency, like jobs 206-216).
# NCCL_DEBUG is OFF here. Run this yourself:  ./submit_4gpu_chain.sh
# Optional: NRUNS=<n> ./submit_4gpu_chain.sh   (default 10)
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
# MLPerf submission metadata. mlperf_common/logging.py reads these two names
# (lines 104 and 108) and otherwise writes SUBMISSION_ORG_PLACEHOLDER /
# 1xSUBMISSION_PLATFORM_PLACEHOLDER into every result log -- which is what the first
# attempt produced. They must be exported, not merely assigned: sbatch propagates the
# exported environment. Verified in the container without a GPU:
#   submission_org      -> "TTA"
#   submission_platform -> "1xTTA_4xH100NVL_NVLink"   (mlperf_common prefixes node count)
#
# The platform string is burned into every log at run time and cannot be changed
# afterwards, so the identifier is frozen here. A storage product name, when its
# use is confirmed, goes only into systems/<id>.json's system_name field -- see
# docs/V61_ORIENTATION_GAP.md [K].
export MLPERF_SUBMISSION_ORG=TTA
export MLPERF_SUBMISSION_PLATFORM=TTA_4xH100NVL_NVLink

# Deliberately NOT setting SEED. run.sub:38 does
#   : "${SEED_BASE:=${SEED-$RANDOM}}"
# so an inherited SEED would freeze SEED_BASE and every run would log the same seed,
# which makes seed_checker reject the whole results directory.
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
unset NCCL_DEBUG   # debug logging off for the official chain

source config_NVLink_H100_BF16_RCP_BS32_4GPU.sh

NRUNS=${NRUNS:-10}
dep=""
for i in $(seq 1 "${NRUNS}"); do
  if [ -z "${dep}" ]; then
    jid=$(sbatch --parsable -N "${DGXNNODES}" --time="${WALLTIME}" run.sub)
  else
    jid=$(sbatch --parsable --dependency=afterany:"${dep}" -N "${DGXNNODES}" --time="${WALLTIME}" run.sub)
  fi
  echo "run #${i} -> job ${jid}${dep:+ (afterany:${dep})}"
  dep=${jid}
done
echo
echo "power logs will be at: ${POWERLOGDIR}/IPMIPower_<jobid>.txt (one file per run)"
