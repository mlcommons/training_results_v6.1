# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

: "${LOGDIR:=./results}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${EXTRA_MOUNTS:=""}"

: "${HF_CKPT_PATH:?HF_CKPT_PATH not set (source config_data_<cluster>.sh)}"
if [[ "${RANDOM_INIT_POLICY:-0}" != "1" ]]; then
    : "${NRL_MEGATRON_CHECKPOINT_DIR:?NRL_MEGATRON_CHECKPOINT_DIR not set}"
fi
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    : "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?NEMO_GYM_SWE_TRAIN_DATA_PATH not set}"
    : "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?NEMO_GYM_SWE_VALIDATION_DATA_PATH not set}"
    : "${NEMO_GYM_SWE_SIF_DIR:?NEMO_GYM_SWE_SIF_DIR not set}"
fi

CONTAINER_INPUT_ROOT=${CONTAINER_INPUT_ROOT:-/inputs/nemo_gym}
export CONTAINER_HF_CKPT_PATH=${CONTAINER_HF_CKPT_PATH:-${HF_CKPT_PATH}}
if [[ "${RANDOM_INIT_POLICY:-0}" == "1" ]]; then
    export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=/tmp/nemo_rl_random_init_no_checkpoint
else
    export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-${CONTAINER_INPUT_ROOT}/mcore_ckpt}
fi
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/train.jsonl}
    export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/validation.jsonl}
    export CONTAINER_NEMO_GYM_SWE_SIF_DIR=${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-${CONTAINER_INPUT_ROOT}/sif}
fi

_nemo_log_dir="${LOGDIR}/nemo_logs_${SLURM_JOB_ID:-${DATESTAMP}}"
unset _checkpoint_dir
_grpo_fresh_output_paths_owned=0
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

_checkpoint_mount="${_shared_checkpoint_job}:/checkpoint"
_input_mount_mode=""
if [[ -n "${GRPO_SINGLE_EXPERIMENT_INDEX:-}" ]]; then
    if [[ "${OFFLINE_EVAL:-0}" != 1 ||
          ! "${GRPO_SINGLE_EXPERIMENT_INDEX}" =~ ^0*[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid fresh experiment identity" >&2; exit 1
    fi
    _fresh_index="${GRPO_SINGLE_EXPERIMENT_INDEX}"
    _input_mount_mode=":ro"
    _nemo_log_dir="${_nemo_log_dir}/experiments/${_fresh_index}"
    _hf_cache_dir="${_hf_cache_dir}/qwen35-${SLURM_JOB_ID}/experiments/${_fresh_index}"
    _fresh_checkpoint_dir="${_shared_checkpoint_job}/${_fresh_index}"
    for _fresh_path in "${_nemo_log_dir}" "${_hf_cache_dir}" "${_fresh_checkpoint_dir}"; do
        [[ ! -e "${_fresh_path}" && ! -L "${_fresh_path}" &&
           "$(realpath -m -- "${_fresh_path}")" == "${_fresh_path}" ]] || {
            echo "ERROR: refusing stale or non-canonical experiment output: ${_fresh_path}" >&2; exit 1;
        }
    done
    for _fresh_path in "${_nemo_log_dir}" "${_hf_cache_dir}" "${_fresh_checkpoint_dir}"; do
        ( umask 0002; mkdir -p -- "$(dirname "${_fresh_path}")" ) || exit $?
        [[ "$(realpath -e -- "$(dirname "${_fresh_path}")")/$(basename "${_fresh_path}")" == "${_fresh_path}" ]] || {
            echo "ERROR: fresh experiment path has a non-canonical ancestor" >&2; exit 1;
        }
        ( umask 0002; mkdir -- "${_fresh_path}" ) || exit $?
    done
    _grpo_fresh_output_paths_owned=1
    _checkpoint_mount="${_fresh_checkpoint_dir}:/checkpoint/${_fresh_index}"
    echo "GRPO_FRESH_EXPERIMENT_PATHS: experiment=${_fresh_index} logs=${_nemo_log_dir} cache=${_hf_cache_dir} checkpoint=${_fresh_checkpoint_dir}"
fi
_checkpoint_dir="${_shared_checkpoint_job}"

_cont_mounts="${LOGDIR}:/results"
if [[ -n "${GRPO_SINGLE_EXPERIMENT_INDEX:-}" ]]; then
    _cont_mounts="${_nemo_log_dir}:/results"
fi
_cont_mounts+=",${_nemo_log_dir}:/logs"
_cont_mounts+=",${_checkpoint_mount}"
_cont_mounts+=",${_hf_cache_dir}:/opt/nemo-rl/.cache"

if [[ -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]]; then
    prepare_grpo_local_checkpoints || exit $?
    export CONTAINER_GRPO_LOCAL_CHECKPOINT_DIR=/local_checkpoint
    _cont_mounts+=",${GRPO_LOCAL_CHECKPOINT_DIR}/qwen35-${SLURM_JOB_ID}:/local_checkpoint"
fi

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

if [[ -n "${RAY_CLUSTER_LOG_DIR:-}" ]]; then
    _cont_mounts+=",${RAY_CLUSTER_LOG_DIR}:${RAY_CLUSTER_LOG_DIR}"
fi

if [[ "${HF_CKPT_PATH}" == */snapshots/* ]]; then
    _hf_model_cache_dir="$(dirname "$(dirname "${HF_CKPT_PATH}")")"
    _cont_mounts+=",${_hf_model_cache_dir}:${_hf_model_cache_dir}${_input_mount_mode}"
else
    _cont_mounts+=",${HF_CKPT_PATH}:${HF_CKPT_PATH}${_input_mount_mode}"
fi

if [[ "${RANDOM_INIT_POLICY:-0}" != "1" ]]; then
    _cont_mounts+=",${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}${_input_mount_mode}"
fi

if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    _cont_mounts+=",${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}:ro"
    _cont_mounts+=",${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}:ro"
    _train_data_dir="$(dirname "${NEMO_GYM_SWE_TRAIN_DATA_PATH}")"
    _val_data_dir="$(dirname "${NEMO_GYM_SWE_VALIDATION_DATA_PATH}")"
    _cont_mounts+=",${_train_data_dir}:${_train_data_dir}${_input_mount_mode}"
    if [[ "${_val_data_dir}" != "${_train_data_dir}" ]]; then
        _cont_mounts+=",${_val_data_dir}:${_val_data_dir}${_input_mount_mode}"
    fi
    _cont_mounts+=",${NEMO_GYM_SWE_SIF_DIR}:${CONTAINER_NEMO_GYM_SWE_SIF_DIR}:ro"
    _cont_mounts+=",${NEMO_GYM_SWE_SIF_DIR}:${NEMO_GYM_SWE_SIF_DIR}:ro"

    _cont_mounts+=",/dev/fuse:/dev/fuse"
fi

if [[ "${NRL_SOURCE_OVERLAY:-0}" == "1" ]]; then
    : "${REPO_LOCATION:?NRL_SOURCE_OVERLAY=1 requires REPO_LOCATION (host nemo-rl checkout)}"
    for overlay_dir in nemo_rl examples qwen_35; do
        _cont_mounts+=",${REPO_LOCATION}/${overlay_dir}:/opt/nemo-rl/${overlay_dir}"
    done
fi

if [[ -n "${EXTRA_MOUNTS}" ]]; then
    _cont_mounts+=",${EXTRA_MOUNTS}"
fi

if [[ "${JET:-0}" -eq 1 && -z "${GRPO_SINGLE_EXPERIMENT_INDEX:-}" ]]; then
    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
fi

mounts_to_verify="HF_CKPT:${CONTAINER_HF_CKPT_PATH}"
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    mounts_to_verify+=" SIF_DIR:${CONTAINER_NEMO_GYM_SWE_SIF_DIR} TRAIN_DATA:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH} VAL_DATA:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
fi
