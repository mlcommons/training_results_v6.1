#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

                                                
set -e

                                                                          
                                                                            
                                          

                       
: "${SEED:?SEED not set}"
: "${DGXNNODES:?DGXNNODES not set}"
: "${DGXNGPU:?DGXNGPU not set}"
: "${GEN_NODES:?GEN_NODES not set}"

                    
: "${RECIPE:=/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm.yaml}"
: "${MLPERF_TARGET_ACCURACY:=0.69}"
: "${MLPERF_BENCHMARK_NAME:=qwen35_397b_grpo}"
: "${EXP_NAME:=${SPREFIX:-qwen35_397b_grpo_trtllm}}"
: "${EXTRA_ARGS:=$@}"
: "${OFFLINE_EVAL:=0}"
export OFFLINE_EVAL
case "${OFFLINE_EVAL}" in
    0|1) ;;
    *) echo "ERROR: OFFLINE_EVAL must be 0 or 1" >&2; exit 1 ;;
esac

                                                      
case "${MCORE_GDN_FUSION:-1}" in
    1|true|True|TRUE) _gdn_fusion=true ;;
    0|false|False|FALSE) _gdn_fusion=false ;;
    *) echo "MCORE_GDN_FUSION must be 0/1 or true/false" >&2; exit 2 ;;
esac

[ "${DEBUG:-0}" = "0" ] || set -x

cd /opt/nemo-rl

                                                                         
                                                                      
export CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH:?CONTAINER_HF_CKPT_PATH not set (config_mounts.sh)}"
export NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:?not set (config_mounts.sh)}"
export NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:?not set (config_mounts.sh)}"
export NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:?not set (config_mounts.sh)}"
export NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR:?not set (config_mounts.sh)}"
_experiment_suffix=""
if [[ "${OFFLINE_EVAL}" == 1 && "${NEXP:-1}" -gt 1 ]]; then
    _experiment_suffix="/${_experiment_index:?experiment index required}"
fi
export NEMO_GYM_SWE_WORKSPACE_ROOT="/logs/nemo_gym/workspace${_experiment_suffix}"
if [[ -n "${GRPO_LOCAL_CHECKPOINT_DIR:-}" ]]; then
    export GRPO_LOCAL_CHECKPOINT_DIR="${CONTAINER_GRPO_LOCAL_CHECKPOINT_DIR:?local checkpoint mount not configured}"
fi

export HF_HOME="/opt/nemo-rl/.cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-1800}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
                                                                        
                                                                            
                                                                           
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
_ckpt_dir="${CHECKPOINT_DIR:-/checkpoint/${_experiment_index:-01}}"
if [[ -n "${QWEN_EVAL_ONLY_CHECKPOINT_ROOT:-}" ]]; then
    [[ "${OFFLINE_EVAL}" == 1 ]] || exit 1
    if [[ "${1:-}" != "--deferred-eval" ]]; then
        echo "QWEN_EVAL_RECOVERY: skipping training; original checkpoints are read-only"
        exit 0
    fi
    _ckpt_dir=/recovery-checkpoints
fi

if [[ "${1:-}" == "--deferred-eval" ]]; then
    [[ "${OFFLINE_EVAL}" == "1" ]] || { echo "Deferred evaluation is not enabled" >&2; exit 1; }
    export NEMO_GYM_SWE_WORKSPACE_ROOT="/logs/nemo_gym/deferred-workspace${_experiment_suffix}"
    exec uv run --no-sync --project /opt/nemo-rl \
        "${BENCHMARK_DIR:-/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch}/utility/qwen35_mcore_deferred.py" \
        --checkpoint-root "${_ckpt_dir}" \
        --log-dir "/logs/deferred-evaluation${_experiment_suffix}"
fi

if [ "${FORCE_SUCCESS_STATUS:-0}" -eq 1 ]; then
    _force_success=True
else
    _force_success=False
fi

declare -a OVERRIDES
OVERRIDES+=("++policy.megatron_cfg.model_overrides.gdn_pre_gated_delta_rule_fusion=${_gdn_fusion}")
OVERRIDES+=("++policy.megatron_cfg.model_overrides.gdn_gated_output_norm_fusion=${_gdn_fusion}")
OVERRIDES+=("++logger.mlperf_enabled=True")
OVERRIDES+=("++logger.mlperf.benchmark=${MLPERF_BENCHMARK_NAME}")
OVERRIDES+=("++logger.mlperf.target_accuracy=${MLPERF_TARGET_ACCURACY}")
OVERRIDES+=("++grpo.stop_at_validation_threshold=${MLPERF_TARGET_ACCURACY}")
OVERRIDES+=("++logger.mlperf.force_success_status=${_force_success}")
OVERRIDES+=("++cluster.num_nodes=${DGXNNODES}")
OVERRIDES+=("++cluster.gpus_per_node=${DGXNGPU}")
if [[ "${COLOCATED_GENERATION:-0}" == "1" ]]; then
    _generation_nodes="${COLOCATED_GENERATION_NODES:-${TRAIN_NODES}}"
    _generation_mode=" colocated"
else
    _generation_nodes="${GEN_NODES}"
    _generation_mode=""
fi
OVERRIDES+=("++policy.generation.colocated.resources.num_nodes=${_generation_nodes}")
OVERRIDES+=("++policy.generation.colocated.resources.gpus_per_node=${DGXNGPU}")
OVERRIDES+=("++logger.wandb.name=${EXP_NAME}")
                                                                
if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
    OVERRIDES+=("++logger.wandb_enabled=true")
    if [[ -n "${WANDB_PROJECT:-}" ]]; then OVERRIDES+=("++logger.wandb.project=${WANDB_PROJECT}"); fi
    if [[ -n "${WANDB_ENTITY:-}" ]]; then OVERRIDES+=("++logger.wandb.entity=${WANDB_ENTITY}"); fi
fi
OVERRIDES+=("++logger.log_dir=/logs")
                                                                           
                                                                           
                                                                            
mkdir -p "${_ckpt_dir}"
OVERRIDES+=("++checkpointing.checkpoint_dir=${_ckpt_dir}")
OVERRIDES+=("++grpo.seed=${SEED}")

if [[ -n "${MAX_STEPS:-}" ]]; then
    OVERRIDES+=("++grpo.max_num_steps=${MAX_STEPS}")
fi
if [[ -n "${VAL_START_AT:-}" ]]; then
    OVERRIDES+=("++grpo.val_start_at=${VAL_START_AT}")
fi
if [[ -n "${NUM_PROMPTS_PER_STEP:-}" ]]; then
    OVERRIDES+=("++grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}")
fi
if [[ -n "${NUM_GENERATIONS_PER_PROMPT:-}" ]]; then
    OVERRIDES+=("++grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT}")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
    OVERRIDES+=("++policy.megatron_cfg.optimizer.lr=${LEARNING_RATE}")
    OVERRIDES+=("++policy.megatron_cfg.optimizer.min_lr=${LEARNING_RATE}")
fi
if [[ -n "${MAX_GRAD_NORM:-}" ]]; then
    OVERRIDES+=("++policy.max_grad_norm=${MAX_GRAD_NORM}")
fi
                                                                    
if [[ -n "${NEMORL_TRAIN_SEGMENT_SIZE:-}" ]]; then
    OVERRIDES+=("++cluster.segment_size=${NEMORL_TRAIN_SEGMENT_SIZE}")
fi

echo "running GRPO benchmark: recipe=${RECIPE} seed=${SEED}" \
     "nodes=${DGXNNODES} (train=${TRAIN_NODES} gen=${_generation_nodes}${_generation_mode})"
echo "Extra args: ${EXTRA_ARGS}"

                                                                        
                                                             
ret_code=0
                                                                               
if [[ -f "${BENCHMARK_DIR:-/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch}/run_grpo_nemo_gym.py" ]]; then
    _grpo_driver="${BENCHMARK_DIR:-/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch}/run_grpo_nemo_gym.py"
else
    _grpo_driver=examples/nemo_gym/run_grpo_nemo_gym.py
fi
                                                                       
                                                                           
                                                                      
                                                                          
                                                                         
uv run --no-sync --project /opt/nemo-rl "${_grpo_driver}" \
    --config "${RECIPE}" \
    "${OVERRIDES[@]}" \
    ${EXTRA_ARGS} \
    || ret_code=$?

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi
