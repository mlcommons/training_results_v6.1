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

# Comments condensed for publication; launch behavior is unchanged.
set -e

# Run inside the container on each rank, with MRC libraries taking precedence.
if [[ -n "${MRC_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${MRC_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${MRC_PATH:-}" ]]; then
    export PATH="${MRC_PATH}:${PATH}"
fi

: "${RANK:?RANK not set}"
: "${LOCAL_RANK:?LOCAL_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD_SIZE:?LOCAL_WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"

[ "${DEBUG}" = "0" ] && set -x

: "${SEED:?SEED not set}"
: "${WALLTIME:=?WALLTIME not set}"

: "${MULTI_NODE:=''}"
: "${UNITTEST:=0}"

: "${NVTX_FLAG:=0}"
: "${NSYS_METRICS:=0}"
: "${TIME_TAGS:=0}"

: "${LOAD_CHECKPOINT:=""}"
: "${SHARE_RERUNS:=0}"

echo "LOAD_CHECKPOINT=${LOAD_CHECKPOINT}"

: "${USE_DIST_OPTIMIZER:=True}"
: "${CKPT_EVERY_VALIDATION:=False}"
: "${WALLTIME_EXIT_MINUTES:=0}"
: "${EXTRA_ARGS:=$@}"

[[ "${DEBUG}" ]] && echo RANK="${RANK}", LOCAL_RANK="${LOCAL_RANK}", MASTER_ADDR="${MASTER_ADDR}", MASTER_PORT="${MASTER_PORT}", WORLD_SIZE="${WORLD_SIZE}", UCX_NET_DEVICES="${UCX_NET_DEVICES}", NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}", NCCL_IB_HCA="${NCCL_IB_HCA}", NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}", NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING}", SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING="${SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING}", UCX_VFS_ENABLE="${UCX_VFS_ENABLE}"

if [[ "$RANK" -eq 0 ]]; then
    env > /results/container-env-"$SLURM_JOB_ID".log
fi

readonly _explicit_log_dir=/results

# Shared reruns prefer an existing result checkpoint over the initial checkpoint.
if [ -n "${LOAD_CHECKPOINT}" ]; then
  if [ ${SHARE_RERUNS:-0} -eq 1 ] && [ -d "${_explicit_log_dir}/checkpoints" ] && [ -n "$(ls -A "${_explicit_log_dir}/checkpoints")" ]
  then
    [[ "$RANK" -eq 0 ]] && echo \
      "Detected a shared rerun." \
      "Resuming from previous run checkpoint stored in ${_explicit_log_dir}/checkpoints" \
      "instead of the initial checkpoint ${LOAD_CHECKPOINT}"
      unset LOAD_CHECKPOINT
  fi
else
    unset LOAD_CHECKPOINT
fi

if [ -n "${NEMO_RESULTS_SUBDIR}" ]; then
  EXTRA_ARGS+=" exp_manager.explicit_log_dir=\"${_explicit_log_dir}\""
fi

if [ "${TRAIN_ONLY:-0}" -eq 1 ]; then
  EXTRA_ARGS+=" data_prefix@data.data_prefix=train_only_c4"
elif [ "${USE_SYNTHETIC_DATA:-0}" -eq 1 ]; then
  EXTRA_ARGS+=" data_prefix@data.data_prefix=synthetic"
fi

[ "$INTERLEAVED_PIPELINE" == "0" ] && export INTERLEAVED_PIPELINE=null

if [ "$LOCAL_RANK" -eq 0 ]; then
  echo "Hello from: $(hostname)"
fi

if [ "${WALLTIME_EXIT_MINUTES:-0}" -gt 0 ]; then
  [ "${NEXP:-1}" -gt 1 ] && echo "Warning: NEXP>1 and WALLTIME_EXIT_MINUTES>0 makes little sense (max_time for each run is set based on total WALLTIME)."

  max_time_minutes=$(( WALLTIME - WALLTIME_EXIT_MINUTES))
  [[ "$RANK" -eq 0 ]] && echo "Setting max_time to $max_time_minutes minutes"

  EXTRA_ARGS+=" +trainer.max_time=00:00:${max_time_minutes}:00"
fi

if [ "${PRINT_CONFIG_ONLY:-False}" = True ]; then
  EXTRA_ARGS+=" -c job --resolve"
fi

if [ ${NVTX_FLAG} -gt 0 ]; then
  # Limit profiler startup overhead to the selected ranks.
  : "${PROFILE_RANKS:=0}"
  _this_rank="${SLURM_PROCID:-${RANK:-0}}"
  _rank_profiled=0
  IFS=',' read -ra _pr_list <<< "${PROFILE_RANKS}"
  for _r in "${_pr_list[@]}"; do
    [[ "${_r}" == "${_this_rank}" ]] && _rank_profiled=1
  done
  NSYSCMD=" nsys profile --sample=none --cpuctxsw=none --trace=cuda-sw,nvtx --cuda-graph-trace=node --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --output ${NSYS_PREFIX}${SLURM_PROCID}${NSYS_SUFFIX}"
  if [ ${NSYS_METRICS} -gt 0 ]; then
    NSYSCMD="${NSYSCMD} --gpu-metrics-devices=${LOCAL_RANK} --gpu-metrics-set=${NSYS_METRICS_SET} --gpu-metrics-frequency=50000"
  fi
  : "${NSYS_NVML_METRICS_INTERVAL:=0}"
  if [ ${NSYS_NVML_METRICS_INTERVAL} -gt 0 ]; then
    NSYSCMD="${NSYSCMD} --enable nvml_metrics,-i${NSYS_NVML_METRICS_INTERVAL}"
  fi
  if [ ${_rank_profiled} -ne 1 ]; then
    NSYSCMD=""
  fi
fi

[[ "$RANK" -eq 0 ]] && echo "running LLM benchmark"

declare -a CMD

if [[ ${LOCAL_WORLD_SIZE} -gt 1 ]]; then
    CMD=( ${NSYSCMD} 'python' '-u')
else
    # Interactive launches spawn the local GPU ranks through torchrun.
    CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [[ "$RANK" -eq 0 ]]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

if [ "${HANG_MONITOR_TIMEOUT-0}" -gt 0 ] && [ "${ATTEMPT_CUDA_GDB_CORE_DUMP-0}" == "1" ]; then
      mkdir -p "${CUDA_COREDUMP_PIPE_DIR}"
fi

[[ "$RANK" -eq 0 ]] && echo "Extra args: $EXTRA_ARGS"

CUDA_COREDUMP_FILE=/results/llama31.%h.%p ${LOGGER:-} ${BINDCMD:-} ${CMD[@]} /workspace/llm/pretrain.py \
	$EXTRA_ARGS \
	; ret_code=$?

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi
