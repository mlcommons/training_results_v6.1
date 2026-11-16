#!/bin/bash
# Submit NVIDIA v6.1 benchmark runs from the x86 SIFs to HiPerGator B200 (run_hpg.sub). See README.md.
# Usage: launcher/run_hpg.sh <bench> <config> [extra sbatch args]
#   <bench>   llama31_8b, gpt_oss_20b, llama2_70b_lora, flux1, deepseekv3_671b
#   <config>  config file in University-of-Florida/benchmarks/<bench>/implementations/<impl>/,
#             for example config_DGXB200_1x8x2xtp1pp1cp1_8b.sh
#   NEXP=10                runs in the job (default 10), each with a new seed
#   SMOKE=1                also source configs/smoke_<bench>.sh (1 short run, results_smoke/)
#   PINNED=1               use the pinned SIF variant (<bench>-amd_20261006_pinned.sif)
#   RESERVATION=<name>     add --reservation (for example mlperf-benchmark)
#   CONT, DATADIR, MODEL, LOGDIR: override the defaults in launcher/kit_defaults.sh
set -euo pipefail

LAUNCHER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export KIT=$(dirname "${LAUNCHER}")
export BENCH=${1:?usage: launcher/run_hpg.sh <bench> <config> [extra sbatch args]}
CONFIG=${2:?usage: launcher/run_hpg.sh <bench> <config> [extra sbatch args]}
shift 2
HOST=hpg
set -a +u
source "${LAUNCHER}/kit_defaults.sh"   # sets CTX: the benchmark implementation dir
[[ -f "${CTX}/${CONFIG}" ]] || { echo "ERROR: ${CTX}/${CONFIG} not found" >&2; exit 1; }
source "${CTX}/${CONFIG}"
[[ "${SMOKE:-0}" == 1 ]] && source "${LAUNCHER}/smoke/smoke_${BENCH}.sh"
: "${NEXP:=10}"
set +a -u
[[ -f "${CONT}" ]] || { echo "ERROR: ${CONT} not found" >&2; exit 1; }

# Read by the MLPerf loggers for submission_org and submission_platform
export MLPERF_SUBMISSION_ORG="University-of-Florida"
export MLPERF_SUBMISSION_PLATFORM="hpg.dgxb200x8"
_runs=runs
[[ "${SMOKE:-0}" == 1 ]] && _runs=runs_smoke
export LOGDIR=${LOGDIR:-${KIT}/${_runs}/${BENCH}/${DGXSYSTEM}}
mkdir -p "${LOGDIR}" "${KIT}/runs/slurm"

# Slurm time: per run the run limit + 10 min (cache drop, start, checker), + 10 min
sbatch -A rc-rse -p hpg-b200 --gpus-per-node 8 -N "${DGXNNODES}" \
    ${RESERVATION:+--reservation="${RESERVATION}"} \
    -t $((10 + NEXP * (WALLTIME_RUNANDTIME + 10))) -J "v61_${BENCH}" \
    -o "${KIT}/runs/slurm/%x_%j.out" \
    "$@" "${LAUNCHER}/run_hpg_llama3.sub"
