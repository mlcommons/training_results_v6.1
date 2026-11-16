#!/bin/bash
# Run an NVIDIA v6.1 benchmark from its x86 SIF on the b300 node (no Slurm). See README.md.
# The container's own code runs (no repo bind). As NVIDIA's run.sub does, the configs and
# config_mounts.sh are sourced on the host, from the benchmark implementation dir
# (University-of-Florida/benchmarks/<bench>/implementations/<impl>: the image's workspace + our configs).
#
# Usage (the node is shared, so run it in the background):
#   nohup launcher/run_b300.sh <bench> <config> > runs/b300_<bench>_$(date +%y%m%d%H%M).out 2>&1 &
#   <bench>   llama31_8b, gpt_oss_20b, llama2_70b_lora
#   <config>  config file in the benchmark implementation dir, for example config_DGXB300_1x8x2xtp1pp1cp1_8b.sh
#   NEXP=10   runs (default 10), each with a new seed
#   SMOKE=1   also source launcher/smoke/smoke_<bench>.sh (1 short run, output in runs_smoke/)
#   PINNED=1  use the pinned SIF variant (<bench>-amd_20261006_pinned.sif)
#   CONT, DATADIR, MODEL, LOGDIR: override the defaults in launcher/kit_defaults.sh
set -exo pipefail

LAUNCHER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=$(dirname "${LAUNCHER}")
BENCH=${1:?usage: launcher/run_b300.sh <bench> <config>}
CONFIG=${2:?usage: launcher/run_b300.sh <bench> <config>}
HOST=b300
source "${LAUNCHER}/kit_defaults.sh"   # sets CTX: the benchmark implementation dir
[[ -f "${CTX}/${CONFIG}" ]] || { echo "ERROR: ${CTX}/${CONFIG} not found" >&2; exit 1; }
source "${CTX}/${CONFIG}"
[[ "${SMOKE:-0}" == 1 ]] && source "${LAUNCHER}/smoke/smoke_${BENCH}.sh"
: "${NEXP:=10}"
[[ -f "${CONT}" ]] || { echo "ERROR: ${CONT} not found" >&2; exit 1; }

# Read by the MLPerf loggers for submission_org and submission_platform
export MLPERF_SUBMISSION_ORG="University-of-Florida"
export MLPERF_SUBMISSION_PLATFORM="hpg.dgxb300x8"

_runs=runs
[[ "${SMOKE:-0}" == 1 ]] && _runs=runs_smoke
: "${LOGDIR:=${KIT}/${_runs}/${BENCH}/${DGXSYSTEM}}"
: "${MLPERF_RULESET:=6.1.0}"
: "${MLPERF_CLUSTER_NAME:='unknown'}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DROPCACHE_CMD:="sudo -n /sbin/sysctl vm.drop_caches=3"}"
: "${CHECK_COMPLIANCE:=1}"
export DGXSYSTEM DATESTAMP MLPERF_CLUSTER_NAME CONT
export MODEL_FRAMEWORK="pytorch"
export LOGBASE="${DATESTAMP}" NEMO_RESULTS_SUBDIR="${DATESTAMP}"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=${DGXNGPU} LOCAL_WORLD_SIZE=${DGXNGPU}
export PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${DATESTAMP}
# The benchmark code reads some Slurm variables (llama utils.get_rank() reads SLURM_PROCID and
# defaults to 0, so every rank logs; flux reads SLURM_NTASKS_PER_NODE, default 4). Set them as srun
# would for 1 node; SLURM_PROCID and SLURM_LOCALID are set per rank below.
export SLURM_NODEID=0 SLURM_NNODES=1 SLURM_NTASKS=${DGXNGPU} SLURM_NTASKS_PER_NODE=${DGXNGPU}
export SLURM_JOB_ID=${DATESTAMP:0:12} SLURM_JOBID=${DATESTAMP:0:12}

( umask 0002; mkdir -p "${LOGDIR}" )
# config_mounts.sh sets _cont_mounts (and makes its host dirs), as for srun --container-mounts.
_cont_mounts=""
source "${CTX}/config_mounts.sh"
MOUNTS="${_cont_mounts}"
[[ "${MOUNTS}" == *":/results"* ]] || MOUNTS="${LOGDIR}:/results,${MOUNTS}"
# NVIDIA's run.sub runs the image writable (pyxis --container-writable); the SIF is read-only.
# Hydra writes outputs/<date>/<time> in the work dir (LoRA, GPT-OSS): bind a host dir there.
mkdir -p "${LOGDIR}/${DATESTAMP}_outputs"
MOUNTS="${MOUNTS},${LOGDIR}/${DATESTAMP}_outputs:${WORK_DIR}/outputs"
if [[ "${MCORE_BUILD_BIND:-0}" == 1 ]]; then
    mkdir -p "${LOGDIR}/mcore_build"
    MOUNTS="${MOUNTS},${LOGDIR}/mcore_build:/workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/build"
fi
echo "MOUNTS=${MOUNTS}"

appt() {
    apptainer exec --nv -B "${MOUNTS}" --pwd "${WORK_DIR}" "${CONT}" "$@"
}

readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
for _experiment_index in $(seq 1 "${NEXP}"); do
    # The node is shared: do not start when another job uses the GPUs
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [[ "${busy}" -ne 0 ]]; then
        nvidia-smi
        echo "ERROR: ${busy} processes use the GPUs, stop before run ${_experiment_index}" >&2
        exit 1
    fi
    export SEED=${SEED_OVERRIDE:-$RANDOM}   # a new seed for each run
    echo "EXPERIMENT_START $(date +%s)"
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
        echo ":::DLPAL ${CONT} none 1 $(hostname) ${MLPERF_CLUSTER_NAME} ${DGXSYSTEM}"
        echo ":::SYSJSON $(appt mlperf-sysjson.sh)"
        appt bash -c 'echo ":::GITCOMMITID ${GIT_COMMIT_ID:-}"'

        # Clear caches. Log cache_clear only when the drop succeeds.
        if [[ "${CLEAR_CACHES}" -eq 1 ]]; then
            cache_before=$(awk '/^Cached:/ {print $2}' /proc/meminfo)
            if sync && ${DROPCACHE_CMD}; then
                cache_after=$(awk '/^Cached:/ {print $2}' /proc/meminfo)
                echo "$(hostname) cache_cleared ${cache_before}kB to ${cache_after}kB"
                appt python3 -c "
import mlperf_logging.mllog as mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=True)"
            else
                echo "WARNING: cache drop failed, cache_clear not logged"
            fi
        fi

        export -p > "${_logfile_base}_${_experiment_index}.env"
        set +e
        echo "RUNANDTIME_START $(date +%s)"
        # one process per GPU; "N: " labels match srun -l.
        # --pid: when timeout stops the container, the kernel also stops every rank.
        timeout --signal=TERM --kill-after=60 "${WALLTIME_RUNANDTIME}m" \
            apptainer exec --nv --pid -B "${MOUNTS}" --pwd "${WORK_DIR}" "${CONT}" \
            bash -c '
                set -o pipefail
                pids=()
                for ((i = 0; i < LOCAL_WORLD_SIZE; i++)); do
                    ( RANK=$i LOCAL_RANK=$i SLURM_PROCID=$i SLURM_LOCALID=$i slurm2pytorch ./run_and_time.sh 2>&1 | sed -u "s/^/$i: /" ) &
                    pids+=($!)
                done
                rc=0
                for p in "${pids[@]}"; do wait "$p" || rc=$?; done
                exit $rc'
        echo "$?" > "${LOGDIR}/.exit_status_${DATESTAMP}_${_experiment_index}"
        echo "RUNANDTIME_STOP $(date +%s)"
        set -e
    ) |& tee "${_logfile_base}_raw_${_experiment_index}.log"

    RUN_AND_TIME_EXIT_STATUS=$(cat "${LOGDIR}/.exit_status_${DATESTAMP}_${_experiment_index}" 2>/dev/null || echo 255)
    echo "RUN_AND_TIME_EXIT_STATUS=${RUN_AND_TIME_EXIT_STATUS}"

    # MLPerf lines sorted by timestamp: the file for the submission and the checkers
    grep ":::" "${_logfile_base}_raw_${_experiment_index}.log" | sed -E 's/^ *[0-9]+: (:::)/\1/' \
        | sort -k5 -n -s > "${_logfile_base}_${_experiment_index}.log" || true

    if [[ "${CHECK_COMPLIANCE}" -eq 1 ]]; then
        appt python3 -m mlperf_logging.compliance_checker --usage training \
            --ruleset "${MLPERF_RULESET}" \
            --log_output "/results/compliance_${DATESTAMP}_${_experiment_index}.out" \
            "/results/${DATESTAMP}_${_experiment_index}.log" || true
    fi
    echo "EXPERIMENT_STOP $(date +%s)"
done

set +x
python3 "${KIT}/launcher/summarize.py" "${LOGDIR}" "${DATESTAMP}" | tee "${_logfile_base}_summary.txt"
