#!/bin/bash
# 60-step smoke for MLPerf v6.1. Verifies, in one run:
#   step time vs v5.1 (~10.2 s), power sampling at 1 s with no gaps,
#   NCCL/TP group vs the physical NVLink pairs, the 9 new v6.1 mllog keys,
#   and that stripe_count=2 did not regress read throughput.
set -euo pipefail
cd "$(dirname "$0")"

export DATADIR=/lustre/agent/mltraining_v5.1_llama3_8b
export LOGDIR=/lustre/agent/mltraining_v6.1_llama31_8b/log/smoke
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
# Monitoring output lives OUTSIDE the results tree so it never mixes with submission
# artifacts (the package checker warns about unexpected files under power/).
export MONCMDDIR=$PWD/monitor
export MONLOGDIR=/lustre/agent/mltraining_v6.1_llama31_8b/monitoring/smoke
export MON_ENABLE=${MON_ENABLE:-1}
export MON_INTERVAL=${MON_INTERVAL:-10}
export MON_NUMANODE=${MON_NUMANODE:-1}
export NCCL_DEBUG=INFO          # needed to read the actual TP/DP rank mapping
unset  NCCL_DEBUG_SUBSYS || true

./lustre_mount_guard.sh /lustre/agent || exit 1
( umask 0002; mkdir -p "${POWERLOGDIR}" )

source config_smoke60_4GPU.sh
jid=$(sbatch --parsable -N "${DGXNNODES}" --time="${WALLTIME}" run.sub)
echo "smoke job: ${jid}"
echo "  slurm log : $PWD/slurm-${jid}.out"
echo "  power log : ${POWERLOGDIR}/IPMIPower_${jid}.txt"
echo "  results   : ${LOGDIR}"
