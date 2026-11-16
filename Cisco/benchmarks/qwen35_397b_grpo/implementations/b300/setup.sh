#!/usr/bin/env bash
                                                                           
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/config_Zeus_8xB300_SXM_288GB_multi_planar.sh"
export CONTAINER_HF_CKPT_PATH="${HF_CKPT_PATH}"
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH="${NEMO_GYM_SWE_TRAIN_DATA_PATH}"
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH="${NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
export CONTAINER_NEMO_GYM_SWE_SIF_DIR="${NEMO_GYM_SWE_SIF_DIR}"
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}"
cpu_env_args=()
IFS=',' read -ra cpu_env_names <<< "${NRL_CONTAINER_ENV_FORWARD},CONTAINER_HF_CKPT_PATH,CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH,CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH,CONTAINER_NEMO_GYM_SWE_SIF_DIR,CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR,NEMO_GYM_SWE_TRAIN_DATA_PATH,NEMO_GYM_SWE_VALIDATION_DATA_PATH,NEMO_GYM_SWE_SIF_DIR"
for cpu_env_name in "${cpu_env_names[@]}"; do
    [[ -z "${cpu_env_name}" || ! -v "${cpu_env_name}" ]] || cpu_env_args+=(--env "${cpu_env_name}")
done
cpu_env_args+=(--env NRL_RUNTIME_PATCH)
mkdir -p "$(dirname "${QWEN_CPU_PREFLIGHT_LOG}")"
docker --config "${DOCKER_CONFIG:-${HOME}/.docker}" run --rm --runtime=runc --network=none \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/smoke,readonly" \
    --mount "type=bind,src=${HF_CKPT_PATH},dst=${HF_CKPT_PATH},readonly" \
    --mount "type=bind,src=$(dirname "$(dirname "${HF_CKPT_PATH}")")/blobs,dst=$(dirname "$(dirname "${HF_CKPT_PATH}")")/blobs,readonly" \
    "${cpu_env_args[@]}" --entrypoint /bin/bash \
    qwen35-trtllm:v61-b300-x86-20261012-r1 -euc '
        runtime_patch=/smoke/patches/$(basename "${NRL_RUNTIME_PATCH}")
        git -C /opt/nemo-rl apply --check "${runtime_patch}"
        git -C /opt/nemo-rl apply "${runtime_patch}"
        /opt/nemo_rl_venv/bin/python /smoke/tools/test_deferred_eval_cpu.py
        /opt/nemo_rl_venv/bin/python /smoke/tools/validate_smoke_cpu.py
    ' 2>&1 | tee "${QWEN_CPU_PREFLIGHT_LOG}"
