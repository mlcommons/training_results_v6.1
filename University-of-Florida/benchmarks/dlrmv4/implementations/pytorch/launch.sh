#!/bin/bash
# Submit dlrmv4 runs on HiPerGator B200.
# Usage: ./launch.sh <config_file> [extra sbatch args]
#   NEXP=10               number of runs in the job (default 10)
#   RESERVATION=<name>    add --reservation (for example mlperf-benchmark)
#   STAGE_DATA=0          read the dataset from /blue instead of node-local disk
#   LOGDIR=<dir>          results dir (default: the run kit, runs/dlrmv4/<DGXSYSTEM>)
#   KIT=<dir>             run kit root (default /blue/rc-rse/mlperf/training/v61_runkit)
#   SMOKE=1               one short run (config_smoke.sh) to check the setup;
#                         results go to runs_smoke/. Not a valid MLPerf run.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Output goes to the v6.1 run kit (the submission package), as for the other benchmarks
: "${KIT:=/blue/rc-rse/mlperf/training/v61_runkit}"
CONFIG=${1:?usage: ./launch.sh <config_file> [extra sbatch args]}
shift

export NEXP=${NEXP:-10}
export CONT=/blue/rc-rse/mlperf/training/training/dlrmv4/dlrmv4-amd_20261006.sif
export SLOW_DATADIR=/blue/rc-rse/mlperf/training/minh_training/v6.1/dlrmv4_data

source "${HERE}/${CONFIG}"
source "${HERE}/config_hpg.sh"

if [[ "${SMOKE:-0}" == 1 ]]; then
    source "${HERE}/config_smoke.sh"
    export LOGDIR="${LOGDIR:-${KIT}/runs_smoke/dlrmv4/${DGXSYSTEM}}"
else
    export LOGDIR="${LOGDIR:-${KIT}/runs/dlrmv4/${DGXSYSTEM}}"
fi
mkdir -p "${LOGDIR}" "${KIT}/runs/slurm"

sbatch -A rc-rse -p hpg-b200 --gpus-per-node 8 -N "${DGXNNODES}" \
    ${RESERVATION:+--reservation="${RESERVATION}"} \
    -t $((60 + WALLTIME)) -J "dlrmv4_${DGXSYSTEM}" \
    -o "${KIT}/runs/slurm/%x_%j.out" \
    "$@" "${HERE}/dlrmv4.sub"
