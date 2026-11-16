# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

if [[ "${RANDOM_INIT_POLICY:-0}" == "1" || -n "${PROXY_NUM_LAYERS:-}" ]]; then
    echo "ERROR: Random-init proxy models are not supported by this TRT-LLM benchmark" >&2
    exit 1
fi

: "${LOGDIR:=./results}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${EXTRA_MOUNTS:=""}"

# Host input paths, set by config_data_<cluster>.sh
: "${HF_CKPT_PATH:?HF_CKPT_PATH not set (source config_data_<cluster>.sh)}"
: "${NRL_MEGATRON_CHECKPOINT_DIR:?NRL_MEGATRON_CHECKPOINT_DIR not set}"
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    : "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?NEMO_GYM_SWE_TRAIN_DATA_PATH not set}"
    : "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?NEMO_GYM_SWE_VALIDATION_DATA_PATH not set}"
    : "${NEMO_GYM_SWE_SIF_DIR:?NEMO_GYM_SWE_SIF_DIR not set}"
fi

# Container-side input locations; run_and_time.sh re-exports the original
# names to these values inside the container.
CONTAINER_INPUT_ROOT=${CONTAINER_INPUT_ROOT:-/inputs/nemo_gym}
export CONTAINER_HF_CKPT_PATH=${CONTAINER_HF_CKPT_PATH:-${HF_CKPT_PATH}}
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-${CONTAINER_INPUT_ROOT}/mcore_ckpt}
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/train.jsonl}
    export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/validation.jsonl}
    export CONTAINER_NEMO_GYM_SWE_SIF_DIR=${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-${CONTAINER_INPUT_ROOT}/sif}
fi

_nemo_log_dir="${LOGDIR}/nemo_logs_${SLURM_JOB_ID:-${DATESTAMP}}"
# CHECKPOINTS_DIR selects a shared base, not an allocation's output directory.
# Concurrent --npar jobs share LOGDIR but must never share /checkpoint contents.
unset _checkpoint_dir
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: shared checkpoints require a numeric SLURM_JOB_ID" >&2
    exit 1
fi
_shared_checkpoint_base="${CHECKPOINTS_DIR:-${LOGDIR}/checkpoint}"
_hf_cache_dir="${HF_CACHE_DIR:-${LOGDIR}/hf_cache}"
( umask 0002; mkdir -p "${LOGDIR}" "${_nemo_log_dir}" "${_shared_checkpoint_base}" "${_hf_cache_dir}" ) || exit $?
_shared_checkpoint_base="$(realpath -e -- "${_shared_checkpoint_base}")" || exit $?
_shared_checkpoint_job="${_shared_checkpoint_base}/qwen35-${SLURM_JOB_ID}"
[[ ! -L "${_shared_checkpoint_job}" ]] || { echo "ERROR: checkpoint job root is a symlink" >&2; exit 1; }
( umask 0002; mkdir -p -- "${_shared_checkpoint_job}" ) || exit $?
_grpo_validate_scratch_target "${_shared_checkpoint_base}" "${_shared_checkpoint_job}" || exit $?
_checkpoint_dir="${_shared_checkpoint_job}"

_cont_mounts="${LOGDIR}:/results"
_cont_mounts+=",${_nemo_log_dir}:/logs"
_cont_mounts+=",${_checkpoint_dir}:/checkpoint"
_cont_mounts+=",${_hf_cache_dir}:/opt/nemo-rl/.cache"

if [[ -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]]; then
    prepare_grpo_local_checkpoints || exit $?
    export CONTAINER_GRPO_LOCAL_CHECKPOINT_DIR=/local_checkpoint
    _local_checkpoint_suffix=""
    if [[ "${_GRPO_ISOLATED_EXPERIMENT:-0}" == 1 ]]; then
        _local_checkpoint_suffix="-exp${_experiment_index}"
    fi
    _cont_mounts+=",${GRPO_LOCAL_CHECKPOINT_DIR}/qwen35-${SLURM_JOB_ID}${_local_checkpoint_suffix}:/local_checkpoint"
fi

if [[ "${_GRPO_ISOLATED_EXPERIMENT:-0}" == 1 ]]; then
    prepare_grpo_runtime_scratch || exit $?
    # Enroot normally binds host /dev/shm. Do not expose old Ray/CUDA IPC files.
    # Both directories remain on tmpfs and are shared only within this EXP.
    _runtime_root="/dev/shm/qwen35-${SLURM_JOB_ID}-exp${_experiment_index}-runtime"
    _cont_mounts+=",${_runtime_root}/tmp:/tmp,${_runtime_root}/shm:/dev/shm"
fi

# Dev-only NeMo-RL patch: mount the patch read-only and apply it inside each
# node's writable container before Ray starts. This avoids both an image
# rebuild and mutating a host source checkout.
if [[ -n "${NRL_RUNTIME_PATCH:-}" && "${NRL_SOURCE_OVERLAY:-0}" == "1" ]]; then
    echo "NRL_RUNTIME_PATCH cannot be combined with NRL_SOURCE_OVERLAY=1" >&2
    exit 1
fi
if [[ -n "${NRL_RUNTIME_PATCH:-}" ]]; then
    if [[ ! -f "${NRL_RUNTIME_PATCH}" ]]; then
        echo "NRL_RUNTIME_PATCH does not name a file: ${NRL_RUNTIME_PATCH}" >&2
        exit 1
    fi
    _runtime_patch_host="$(realpath "${NRL_RUNTIME_PATCH}")"
    export NRL_RUNTIME_PATCH_CONTAINER=/tmp/nemo-rl-runtime.patch
    _cont_mounts+=",${_runtime_patch_host}:${NRL_RUNTIME_PATCH_CONTAINER}:ro"
fi

# Identity mount: run.sub polls STARTED_RAY_HEAD on the host while the head
# container touches the same path inside.
if [[ -n "${RAY_CLUSTER_LOG_DIR:-}" ]]; then
    _cont_mounts+=",${RAY_CLUSTER_LOG_DIR}:${RAY_CLUSTER_LOG_DIR}"
fi

# HF checkpoint: identity mount. For HF-hub cache layouts mount the model
# cache root two levels up so snapshot symlinks into blobs/ resolve.
if [[ "${HF_CKPT_PATH}" == */snapshots/* ]]; then
    _hf_model_cache_dir="$(dirname "$(dirname "${HF_CKPT_PATH}")")"
    _cont_mounts+=",${_hf_model_cache_dir}:${_hf_model_cache_dir}:ro"
else
    _cont_mounts+=",${HF_CKPT_PATH}:${HF_CKPT_PATH}:ro"
fi

# Initial weights are inputs, never a cross-experiment conversion/output cache.
# Materialize MCore inputs before this launcher; emitted checkpoints use /checkpoint.
# Clusters that fill the cache on first use (HF -> mcore conversion) opt in to a
# writable mount with NRL_MEGATRON_CHECKPOINT_RW=1 in config_data_<cluster>.sh.
if [[ "${NRL_MEGATRON_CHECKPOINT_RW:-0}" == "1" ]]; then
    _cont_mounts+=",${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}"
else
    _cont_mounts+=",${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}:ro"
fi

if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    # jsonl + SIF dir: canonical container paths + host-path compat mounts (for
    # recipes carrying absolute host-side data_path/container_formatter entries)
    _cont_mounts+=",${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}:ro"
    _cont_mounts+=",${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}:ro"
    _cont_mounts+=",${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${NEMO_GYM_SWE_TRAIN_DATA_PATH}:ro"
    if [[ "${NEMO_GYM_SWE_VALIDATION_DATA_PATH}" != "${NEMO_GYM_SWE_TRAIN_DATA_PATH}" ]]; then
        _cont_mounts+=",${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:ro"
    fi
    _cont_mounts+=",${NEMO_GYM_SWE_SIF_DIR}:${CONTAINER_NEMO_GYM_SWE_SIF_DIR}:ro"
    _cont_mounts+=",${NEMO_GYM_SWE_SIF_DIR}:${NEMO_GYM_SWE_SIF_DIR}:ro"

    # apptainer needs /dev/fuse for the SIF sandboxes
    _cont_mounts+=",/dev/fuse:/dev/fuse"
fi

# Dev-only source overlay (production runs the baked sources): mount a host nemo-rl
# checkout's source dirs over the baked ones. 3rdparty/ stays baked (Gym
# patches are applied at image build).
if [[ "${NRL_SOURCE_OVERLAY:-0}" == "1" ]]; then
    : "${REPO_LOCATION:?NRL_SOURCE_OVERLAY=1 requires REPO_LOCATION (host nemo-rl checkout)}"
    for overlay_dir in nemo_rl examples qwen_35; do
        _cont_mounts+=",${REPO_LOCATION}/${overlay_dir}:/opt/nemo-rl/${overlay_dir}"
    done
fi

if [[ -n "${EXTRA_MOUNTS}" ]]; then
    _cont_mounts+=",${EXTRA_MOUNTS}"
fi

mounts_to_verify="HF_CKPT:${CONTAINER_HF_CKPT_PATH}"
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    mounts_to_verify+=" SIF_DIR:${CONTAINER_NEMO_GYM_SWE_SIF_DIR} TRAIN_DATA:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH} VAL_DATA:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
fi

# Gym SWE results dir, which the node-spread SWE runner writes inside the
# container and therefore needs on a shared filesystem.
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    _gym_results_dir="${LOGDIR}/nemo_gym_results"
    mkdir -p "${_gym_results_dir}"
    _cont_mounts+=",${_gym_results_dir}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/results"
fi
