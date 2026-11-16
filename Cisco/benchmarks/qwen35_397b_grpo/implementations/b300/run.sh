#!/usr/bin/env bash
set -euo pipefail
(($# == 0)) || { echo 'No arguments expected; settings live in the adjacent config.' >&2; exit 2; }
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
QWEN_CONFIG="${SCRIPT_DIR}/config_Zeus_8xB300_SXM_288GB_multi_planar.sh"
source "${QWEN_CONFIG}"
export CONT="${CONT:?Set CONT}"
export RUNSUB_DIR="${SCRIPT_DIR}"
export QWEN_NATIVE_INPUT_MANIFEST="${SCRIPT_DIR}/expected-mounts-x86_64.csv"
export LOGDIR=${LOGDIR:-${SCRIPT_DIR}/results/run_$(date +%Y%m%d_%H%M%S)}
export SEED=${SEED:-43}
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo 'SEED must be a nonnegative integer' >&2; exit 2; }
export QWEN_EVAL_ONLY_CHECKPOINT_ROOT=
unset EXTRA_MOUNTS REPO_LOCATION CPUS_PER_WORKER NRL_GPU_CPU_AFFINITY_FILE
                                                                             
                                                                           
unset XDG_CACHE_HOME
export TORCHINDUCTOR_CACHE_DIR=/opt/nemo-rl/.cache/torchinductor
export TORCH_EXTENSIONS_DIR=/opt/nemo-rl/.cache/torch_extensions
export TRITON_CACHE_DIR=/opt/nemo-rl/.cache/triton
export CUDA_CACHE_PATH=/opt/nemo-rl/.cache/cuda
export CUDA_CACHE_MAXSIZE=4294967296
export NRL_CONTAINER_ENV_FORWARD="${NRL_CONTAINER_ENV_FORWARD},TORCHINDUCTOR_CACHE_DIR,TORCH_EXTENSIONS_DIR,TRITON_CACHE_DIR,CUDA_CACHE_PATH,CUDA_CACHE_MAXSIZE,NO_PROXY,no_proxy,QWEN_EVAL_ONLY_CHECKPOINT_ROOT"

qwen_job_name=qwen35-b300-scaleplan-3step-noeval
if [[ "${QWEN_FULL_EVAL:-0}" == 1 ]]; then
    [[ "${VAL_START_AT}" == 10 ]]
    [[ "${CHECK_COMPLIANCE}" == 1 && "${CLEAR_CACHES}" == 1 && "${FORCE_SUCCESS_STATUS}" == 0 ]]
    [[ "${MLPERF_TARGET_ACCURACY}" == 0.69 ]]
    if [[ "${OFFLINE_EVAL}" == 1 ]]; then
        [[ "${MAX_STEPS}" == 11 && "${RETAIN_GRPO_CHECKPOINTS}" == 1 ]]
        qwen_job_name=qwen35-b300-offline-eval
        printf 'Qwen B300: 11 real steps; timestamped checkpoints at 10 and 11; offline pass@4 target 0.69\n'
        printf 'No inline validation; native MCore restore and TRT-LLM refit after training; retain checkpoints\n'
    else
        [[ "${OFFLINE_EVAL}" == 0 && "${MAX_STEPS}" == 20 ]]
        qwen_job_name=qwen35-b300-full-eval
        printf 'Qwen B300 diagnostic: up to 20 real steps; inline evaluation from step 10; target 0.69\n'
        printf 'Inline diagnostics are not the checkpoint/offline-evaluation submission path\n'
    fi
    printf 'No accuracy/TTT qualification claim until actual successful evaluation and log check\n'
else
    printf 'Qwen B300 smoke: 3 real GRPO steps; no validation; not a submission\n'
fi
printf 'Policy TP%s PP%s EP%s ETP%s CP1 VP1 MBS1 MXFP8; 32 policy + 32 generation GPUs\n' "${TP}" "${PP}" "${EP}" "${ETP}"
printf 'Generation TRT-LLM TP4 EP4, stock piecewise graphs; GBS512 and stock LR/clip\n'
printf 'P2P-sync off; dispatch/wgrad overlap; DDP grad/param overlap + aligned gather\n'
printf 'Router/permutation and cuDNN norm fusions; native Qwen RoPE; pinned host-local compilation caches\n'
printf 'Container: %s\nLog directory: %s\n' "${CONT}" "${LOGDIR}"
args=(--export=ALL --no-requeue -N 8 --time="${WALLTIME}"
      --gres=gpu:8 --partition="${SBATCH_PARTITION:?Set SBATCH_PARTITION}"
      --nodelist="${SBATCH_NODELIST:?Set SBATCH_NODELIST}"
      --job-name="${qwen_job_name}"
      --output="${LOGDIR}/slurm-%j.out")
[[ -z "${SBATCH_ACCOUNT:-}" ]] || args+=(--account="${SBATCH_ACCOUNT}")
[[ -z "${SBATCH_DEPENDENCY:-}" ]] || args+=(--dependency="${SBATCH_DEPENDENCY}" --kill-on-invalid-dep=yes)
if [[ -n "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}" ]]; then
    [[ "${OFFLINE_EVAL}" == 1 && "${NEXP}" == 1 ]]
    args+=(--job-name=qwen35-b300-eval-recovery)
    printf 'EVALUATION ONLY: retaining original training and reading checkpoints from %s\n' "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT}"
fi
if [[ "${QWEN_DRY_RUN:-0}" == 1 ]]; then
    printf 'Would submit: sbatch '; printf '%q ' "${args[@]}" "${RUNSUB_DIR}/run.sub"; printf '\n'
    exit 0
fi
[[ -f "${CONT}" && "${CONT}" == *.sqsh && -s "${QWEN_NATIVE_INPUT_MANIFEST}" ]]
                                                                              
unsquashfs -cat "${CONT}" usr/bin/bash | python3 -c '
import struct, sys
elf = sys.stdin.buffer.read()
assert elf[:4] == b"\x7fELF" and len(elf) >= 20
assert struct.unpack(("<" if elf[5] == 1 else ">") + "H", elf[18:20])[0] == 62
'
for recipe in grpo_qwen35_397b_swe_openhands_async.yaml grpo_qwen35_397b_swe_openhands_async_trtllm.yaml grpo_qwen35_397b_swe_openhands_async_trtllm_mxfp8.yaml; do
    unsquashfs -cat "${CONT}" "workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/${recipe}" |
        cmp -s - "${QWEN_OFFICIAL_DIR}/conf/${recipe}"
done
mapfile -t hosts < <(scontrol show hostnames "${SBATCH_NODELIST:?Set SBATCH_NODELIST}")
((${#hosts[@]} == DGXNNODES))
no_proxy_value=${no_proxy:-${NO_PROXY:-localhost,127.0.0.1,::1}}
for host in "${hosts[@]}"; do
    address=$(getent ahostsv4 "${host}" | awk 'NR==1 {print $1}')
    [[ -n "${address}" ]]
                                                                         
                                                                           
    ssh -T -o BatchMode=yes -o ConnectTimeout=10 "${host}" \
        python3 - "${address}" < "${SCRIPT_DIR}/tools/check_service_hostname.py"
    no_proxy_value+=",${host},${address}"
done
export NO_PROXY="${no_proxy_value}" no_proxy="${no_proxy_value}"
mkdir -p "${LOGDIR}" "${NRL_MEGATRON_CHECKPOINT_DIR}" "${HF_CACHE_DIR}"
export QWEN_RUNTIME_SUPPORT_DIR="${LOGDIR}/runtime_support"
mkdir -p "${QWEN_RUNTIME_SUPPORT_DIR}/utility"
cp "${SCRIPT_DIR}/utility/qwen35_mcore_deferred.py" "${QWEN_RUNTIME_SUPPORT_DIR}/utility/qwen35_mcore_deferred.py"
cp "${SCRIPT_DIR}/run_and_time.sh" "${QWEN_RUNTIME_SUPPORT_DIR}/run_and_time.sh"
cp "${SCRIPT_DIR}/run_grpo_nemo_gym.py" "${QWEN_RUNTIME_SUPPORT_DIR}/run_grpo_nemo_gym.py"
cp "${SCRIPT_DIR}/mlperf_warmup.py" "${QWEN_RUNTIME_SUPPORT_DIR}/mlperf_warmup.py"
sha256sum "${QWEN_RUNTIME_SUPPORT_DIR}/utility/qwen35_mcore_deferred.py" "${QWEN_RUNTIME_SUPPORT_DIR}/run_and_time.sh" "${QWEN_RUNTIME_SUPPORT_DIR}/run_grpo_nemo_gym.py" "${QWEN_RUNTIME_SUPPORT_DIR}/mlperf_warmup.py" > "${QWEN_RUNTIME_SUPPORT_DIR}/sha256sums.txt"
cp "${QWEN_CONFIG}" "${LOGDIR}/config.sh"
cp "${NRL_RUNTIME_PATCH}" "${LOGDIR}/runtime_adapter.patch"
cp "${SCRIPT_DIR}/ipc_login_guard.sh" "${LOGDIR}/ipc_login_guard.sh"
cp "${QWEN_NATIVE_INPUT_MANIFEST}" "${LOGDIR}/expected-mounts-x86_64.csv"
sbatch "${args[@]}" "${RUNSUB_DIR}/run.sub" | tee "${LOGDIR}/submission.txt"
printf 'Logs: %s\n' "${LOGDIR}"
