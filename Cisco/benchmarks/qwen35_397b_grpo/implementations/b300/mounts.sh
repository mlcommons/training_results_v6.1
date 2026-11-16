# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

if [[ "${RANDOM_INIT_POLICY:-0}" == "1" || -n "${PROXY_NUM_LAYERS:-}" ]]; then
    echo "ERROR: Random-init proxy models are not supported by this TRT-LLM benchmark" >&2
    exit 1
fi

: "${LOGDIR:=./results}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${EXTRA_MOUNTS:=""}"

                                                   
: "${HF_CKPT_PATH:?HF_CKPT_PATH not set (source config_data_<cluster>.sh)}"
: "${NRL_MEGATRON_CHECKPOINT_DIR:?NRL_MEGATRON_CHECKPOINT_DIR not set}"
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    : "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?NEMO_GYM_SWE_TRAIN_DATA_PATH not set}"
    : "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?NEMO_GYM_SWE_VALIDATION_DATA_PATH not set}"
    : "${NEMO_GYM_SWE_SIF_DIR:?NEMO_GYM_SWE_SIF_DIR not set}"
fi

                                                                         
                                             
CONTAINER_INPUT_ROOT=${CONTAINER_INPUT_ROOT:-/inputs/nemo_gym}
export CONTAINER_HF_CKPT_PATH=${CONTAINER_HF_CKPT_PATH:-${HF_CKPT_PATH}}
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-${CONTAINER_INPUT_ROOT}/mcore_ckpt}
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/train.jsonl}
    export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/validation.jsonl}
    export CONTAINER_NEMO_GYM_SWE_SIF_DIR=${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-${CONTAINER_INPUT_ROOT}/sif}
fi

_nemo_log_dir="${LOGDIR}/nemo_logs_${SLURM_JOB_ID:-${DATESTAMP}}"
                                                                              
                                                                                
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

                                                                   
                                                                         
: "${QWEN_RUNTIME_SUPPORT_DIR:=${RUNSUB_DIR}}"
[[ -f "${QWEN_RUNTIME_SUPPORT_DIR}/utility/qwen35_mcore_deferred.py" &&
   -f "${QWEN_RUNTIME_SUPPORT_DIR}/run_and_time.sh" ]] || exit 1
_cont_mounts+=",${QWEN_RUNTIME_SUPPORT_DIR}/utility/qwen35_mcore_deferred.py:${BENCHMARK_DIR}/utility/qwen35_mcore_deferred.py:ro"
_cont_mounts+=",${QWEN_RUNTIME_SUPPORT_DIR}/run_and_time.sh:${BENCHMARK_DIR}/run_and_time.sh:ro"
_cont_mounts+=",${QWEN_RUNTIME_SUPPORT_DIR}/run_grpo_nemo_gym.py:${BENCHMARK_DIR}/run_grpo_nemo_gym.py:ro"
_cont_mounts+=",${QWEN_RUNTIME_SUPPORT_DIR}/mlperf_warmup.py:${BENCHMARK_DIR}/mlperf_warmup.py:ro"
if [[ -n "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT:-}" ]]; then
    [[ "${OFFLINE_EVAL:-0}" == 1 && "${NEXP:-1}" == 1 &&
       "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}" == "${QWEN_EVAL_CHECKPOINT_BASE:?Set QWEN_EVAL_CHECKPOINT_BASE}"/*/checkpoint/qwen35-*/1 &&
       "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}" != *[:,]* &&
       "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}" != *'/../'* ]] || {
        echo 'ERROR: unexpected evaluation recovery checkpoint root' >&2; exit 1;
    }
    for step in 10 11; do
        [[ -f "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}/step_${step}/training_info.json" &&
           -d "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}/step_${step}/policy/weights" ]] || exit 1
    done
    _cont_mounts+=",${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}:/recovery-checkpoints:ro"
fi

                                                                           
                                                                             
if [[ -n "${QWEN_COMPILE_CACHE_ROOT:-}" ]]; then
    [[ "${QWEN_COMPILE_CACHE_ROOT}" == /var/tmp/qwen35-b300-jit/* &&
       "${QWEN_COMPILE_CACHE_ROOT}" != *[:,]* &&
       "${QWEN_COMPILE_CACHE_ROOT}" != *'/../'* ]] || {
        echo 'ERROR: unexpected compile-cache root' >&2; exit 1;
    }
    srun --overlap --exact --nodes="${SLURM_JOB_NUM_NODES}" \
         --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
         --cpus-per-task=1 --kill-on-bad-exit=1 /bin/bash -euc '
        [[ ! -L /var/tmp/qwen35-b300-jit && ! -L "$1" ]]
        mkdir -p -- "$1"
        [[ "$(stat -c %u "$1")" == "$(id -u)" ]]
        for kind in triton torchinductor torch_extensions cuda xdg; do
            [[ ! -L "$1/$kind" ]]
            mkdir -p -- "$1/$kind"
        done
        printf "QWEN_COMPILE_CACHE: node=%s root=%s\n" "$(hostname)" "$1"
    ' -- "${QWEN_COMPILE_CACHE_ROOT}"
    for _qwen_cache_kind in triton torchinductor torch_extensions cuda xdg; do
        _cont_mounts+=",${QWEN_COMPILE_CACHE_ROOT}/${_qwen_cache_kind}:/opt/nemo-rl/.cache/${_qwen_cache_kind}"
    done
    unset _qwen_cache_kind
fi

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
                                                                                
                                                                           
    _runtime_root="/dev/shm/qwen35-${SLURM_JOB_ID}-exp${_experiment_index}-runtime"
    _cont_mounts+=",${_runtime_root}/tmp:/tmp,${_runtime_root}/shm:/dev/shm"
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
    _cont_mounts+=",${_hf_model_cache_dir}:${_hf_model_cache_dir}:ro"
else
    _cont_mounts+=",${HF_CKPT_PATH}:${HF_CKPT_PATH}:ro"
fi

                                                                           
                                                        
_cont_mounts+=",${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}"

if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
                                                                               
                                                                                
    _cont_mounts+=",${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}:ro"
    _cont_mounts+=",${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}:ro"
    _cont_mounts+=",${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${NEMO_GYM_SWE_TRAIN_DATA_PATH}:ro"
    if [[ "${NEMO_GYM_SWE_VALIDATION_DATA_PATH}" != "${NEMO_GYM_SWE_TRAIN_DATA_PATH}" ]]; then
        _cont_mounts+=",${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:ro"
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

                                                                          
                                                                         
                                                      
: "${QWEN_NATIVE_INPUT_MANIFEST:?native smoke manifest is required}"
[[ -f "${QWEN_NATIVE_INPUT_MANIFEST}" && ! -L "${QWEN_NATIVE_INPUT_MANIFEST}" ]] || {
    echo 'ERROR: missing regular native input manifest' >&2
    exit 1
}
_cont_mounts+=",${QWEN_NATIVE_INPUT_MANIFEST}:${BENCHMARK_DIR}/expected-mounts.csv:ro"

mounts_to_verify="HF_CKPT:${CONTAINER_HF_CKPT_PATH}"
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    mounts_to_verify+=" SIF_DIR:${CONTAINER_NEMO_GYM_SWE_SIF_DIR} TRAIN_DATA:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH} VAL_DATA:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
fi

                                                                         
                                                       
if [[ "${MOCK_ROLLOUT:-0}" != "1" ]]; then
    _gym_results_dir="${LOGDIR}/nemo_gym_results"
    mkdir -p "${_gym_results_dir}"
    _cont_mounts+=",${_gym_results_dir}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/results"
fi
