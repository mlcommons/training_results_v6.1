#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Per-node launcher: starts this node's training container and runs NEXP timed runs.
# Under Slurm (job.sh: `srun bash run_with_docker_slurm.sh`, one task per node) the layout
# comes from SLURM_*; without Slurm, set NNODES, NODE_RANK and MASTER_ADDR on each node.
#
# Required: MLPERF_CONFIG CONT DATADIR LOGDIR ROOT_SEED
# Optional: NEXP (10)  CACHEDIR (node-local compile caches)  MASTER_PORT (29501)
#           CONT_PULL (1: docker pull CONT; 0: use a local image)
#           RUNTIME_TUNABLES (1: log and check the host settings, runtime_tunables.sh)  CHECK_GPUS_IDLE (1)
#           HOST_LIBIONIC (1: mount the host's AI NIC verbs provider over the image's, when the host has one)
#           IP_INTERFACE (bootstrap interface; default: the host's route to MASTER_ADDR)
#           NODE_GPUS (comma list of GPU ids for this node's container; default all)
#           CONFIG_DIR (where config_<MLPERF_CONFIG>.sh lives; default this directory)
#           EXTRA_MOUNTS (space-separated host:container volume specs, for debugging)
set -euo pipefail

if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

: "${MLPERF_CONFIG:?MLPERF_CONFIG not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"
: "${ROOT_SEED:?ROOT_SEED not set}"
: "${NEXP:=10}"
: "${CONT_PULL:=1}"
: "${RUNTIME_TUNABLES:=1}"
: "${HOST_LIBIONIC:=1}"
: "${CHECK_GPUS_IDLE:=1}"
: "${MASTER_PORT:=29501}"
: "${CACHEDIR:=/tmp/mlperf_flux1_cache}"

# ---- Node layout: Slurm or explicit -------------------------------------------------------
if [[ -n "${SLURM_JOB_NODELIST:-}" && -z "${NODE_RANK:-}" ]]; then
    NNODES=${SLURM_NNODES}
    NODE_RANK=${SLURM_NODEID}
    MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
fi
: "${NNODES:?set NNODES (or run under Slurm)}"
: "${NODE_RANK:?set NODE_RANK (or run under Slurm)}"
: "${MASTER_ADDR:?set MASTER_ADDR (or run under Slurm)}"

readonly _config_file="${CONFIG_DIR:-${SCRIPT_DIR}}/config_${MLPERF_CONFIG}.sh"
[[ -f "${_config_file}" ]] || { echo "missing ${_config_file}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${_config_file}"
if (( NNODES != NUM_NODES )) && [[ -z "${MLPERF_DEV_OVERRIDES:-}" ]]; then
    echo "[$(hostname)] ERROR: ${MLPERF_CONFIG} is a ${NUM_NODES}-node config but NNODES=${NNODES}" >&2
    exit 1
fi
if [[ -n "${NODE_GPUS:-}" ]]; then
    GPUS_PER_NODE=$(awk -F, '{print NF}' <<< "${NODE_GPUS}")
else
    GPUS_PER_NODE=${NUM_GPUS}
fi
readonly _cont_name="mlperf_flux1_node${NODE_RANK}"
echo "[$(hostname)] node ${NODE_RANK}/${NNODES}, master ${MASTER_ADDR}:${MASTER_PORT}, ${GPUS_PER_NODE} GPUs, ${NEXP} runs"

mkdir -p "${LOGDIR}" "${CACHEDIR}"
_cont_mounts=(
    "--volume=${DATADIR}:/data:ro"
    "--volume=${LOGDIR}:/results"
    "--volume=${CACHEDIR}:/cache"
    "--volume=${SCRIPT_DIR}:/workspace/code"
)
# AI NIC (ionic) verbs provider: the image's does not support every host kernel's ABI, and with a mismatch RCCL sees no
# RDMA devices. The host's library matches its kernel driver; mounted read-only over the image's copy.
_ionic=/usr/lib/x86_64-linux-gnu/libionic.so.1
if [[ "${HOST_LIBIONIC}" == 1 && -e "${_ionic}" ]]; then
    _cont_mounts+=("--volume=$(readlink -f "${_ionic}"):${_ionic}:ro")
fi
for _m in ${EXTRA_MOUNTS:-}; do _cont_mounts+=("--volume=${_m}"); done
# Bootstrap interface (NCCL / Gloo sockets): the one that routes to the master. Primus otherwise takes the first address
# of `hostname -I`, which on some hosts belongs to a down docker0 bridge.
if [[ -z "${IP_INTERFACE:-}" ]]; then
    _master_ip=$(getent ahostsv4 "${MASTER_ADDR}" | awk 'NR == 1 {print $1}')
    IP_INTERFACE=$(ip -o route get "${_master_ip:-${MASTER_ADDR}}" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')
fi
[[ "${IP_INTERFACE:-}" == lo ]] && IP_INTERFACE=$(ip -o route get 1.1.1.1 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')
echo "[$(hostname)] bootstrap interface: ${IP_INTERFACE:-<Primus default>}"
# RCCL topology file (config_common.sh): mounted read-only, and NCCL_TOPO_FILE points at it.
_topo_env=()
if [[ -n "${RCCL_TOPO_FILE:-}" ]]; then
    [[ -f "${RCCL_TOPO_FILE}" ]] || { echo "[$(hostname)] RCCL_TOPO_FILE ${RCCL_TOPO_FILE} not found" >&2; exit 1; }
    _cont_mounts+=("--volume=${RCCL_TOPO_FILE}:/etc/rccl/topo.xml:ro")
    _topo_env=("--env=NCCL_TOPO_FILE=/etc/rccl/topo.xml")
fi
_devices=(--device /dev/dri --device /dev/kfd)
[[ -e /dev/infiniband ]] && _devices+=(--device /dev/infiniband)

if [[ "${CONT_PULL}" == 1 ]]; then
    for _attempt in 1 2 3; do
        ${DOCKER} pull "${CONT}" && break
        [[ ${_attempt} == 3 ]] && { echo "[$(hostname)] failed to pull ${CONT}" >&2; exit 1; }
        sleep 10
    done
fi

# Every variable the config exports is passed into the container.
mapfile -t _config_env < <(env -i bash -c "source '${_config_file}' && compgen -e" | grep -E -v '^(PWD|SHLVL|_)$')
_env=()
for v in "${_config_env[@]}"; do _env+=("--env=${v}=${!v}"); done
# ROCR_VISIBLE_DEVICES, not HIP_VISIBLE_DEVICES: Primus's launcher sets HIP_VISIBLE_DEVICES to
# 0..GPUS_PER_NODE-1, which then indexes the GPUs the ROCm runtime exposes.
[[ -n "${NODE_GPUS:-}" ]] && _env+=("--env=ROCR_VISIBLE_DEVICES=${NODE_GPUS}")
[[ -n "${MLPERF_DEV_OVERRIDES:-}" ]] && for v in MLPERF_DEV_OVERRIDES PRIMUS_GLOBAL_BATCH_SIZE PRIMUS_LR \
        PRIMUS_LR_WARMUP_ITERS PRIMUS_EVAL_INTERVAL PRIMUS_TRAIN_ITERS USING_AINIC NCCL_IB_HCA PRIMUS_RCCL_GDR \
        PRIMUS_MXFP6_FWD_FP4_HEAL_ITER PRIMUS_MXFP6_FWD_FP4_HEAL_LAYERS_PER_ITER PRIMUS_MXFP6_FWD_FP4_HEAL_TARGET; do
    [[ -n "${!v+x}" ]] && _env+=("--env=${v}=${!v}")
done

cleanup() { ${DOCKER} container rm -f "${_cont_name}" >/dev/null 2>&1 || true; }
cleanup
trap cleanup EXIT

if [[ "${CHECK_GPUS_IDLE}" == 1 ]]; then
    bash "${SCRIPT_DIR}/check_gpus_idle.sh"
fi

${DOCKER} run --rm --init --detach \
    --net=host --uts=host --ipc=host \
    "${_devices[@]}" \
    --group-add video --cap-add SYS_PTRACE --cap-add IPC_LOCK --privileged \
    --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
    --security-opt seccomp=unconfined \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
sleep 5

if [[ "${RUNTIME_TUNABLES}" == 1 ]]; then
    bash "${SCRIPT_DIR}/runtime_tunables.sh"
fi

for _experiment_index in $(seq 1 "${NEXP}"); do
    RUN_INDEX=$((_experiment_index - 1))
    echo "[$(hostname)] beginning run ${RUN_INDEX} (${_experiment_index} of ${NEXP})"
    # MLPerf: flush the page cache before every run so the data starts on durable storage (config_common.sh sets
    # MLPERF_CLEAR_CACHES=true; false, for development runs only, keeps the cache as run_and_time.sh does).
    # Host sudo if it needs no password, else through the privileged training container, whose /proc/sys is the host
    # kernel's.
    if [[ "${MLPERF_CLEAR_CACHES:-true}" == "true" ]]; then
        if ! { sync && { sudo -n /sbin/sysctl -q vm.drop_caches=3 2>/dev/null \
                || ${DOCKER} exec "${_cont_name}" sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'; }; }; then
            echo "[$(hostname)] ERROR: could not drop the page cache; not starting the run" >&2
            exit 1
        fi
        echo "[$(hostname)] page cache dropped"
    else
        echo "[$(hostname)] page cache kept (MLPERF_CLEAR_CACHES=${MLPERF_CLEAR_CACHES}): not a scored run"
    fi
    SEED=$(( ROOT_SEED + 10#${SLURM_ARRAY_TASK_ID:-0} * 100 + _experiment_index ))
    ${DOCKER} exec \
        --env=NNODES="${NNODES}" --env=NODE_RANK="${NODE_RANK}" \
        --env=MASTER_ADDR="${MASTER_ADDR}" --env=MASTER_PORT="${MASTER_PORT}" \
        --env=GPUS_PER_NODE="${GPUS_PER_NODE}" ${IP_INTERFACE:+--env=IP_INTERFACE="${IP_INTERFACE}"} \
        --env=SEED="${SEED}" --env=RUN_INDEX="${RUN_INDEX}" \
        "${_env[@]}" "${_topo_env[@]}" \
        "${_cont_name}" bash /workspace/code/run_and_time.sh \
        2>&1 | tee "${LOGDIR}/${MLPERF_CONFIG}_run${RUN_INDEX}_node${NODE_RANK}.log"
done
