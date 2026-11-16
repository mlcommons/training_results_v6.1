#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail

# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODELDIR:?MODELDIR not set}"
: "${LOGDIR:?LOGDIR not set}"
: "${CONFIG_NAME:?CONFIG_NAME not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=0}"
: "${MLPERF_RULESET:=6.0.0}"
: "${UTILITIES:="$(pwd)/../../utilities"}"

#    Profiling settings
: "${CONT_NAME:=dev-${CUSTOM_TAG:-${USER:-$(whoami)}-$(date +%Y%m%d%H%M%S)-$$}}"
: "${PROFILER:=""}"
: "${ENABLE_TRAINING_METRICS_COLLECTION:=0}"
: "${LOG_FREQ:=0}"
: "${TORCHPROF_OUTPUT_DIR:=/results/torchprof}"
: "${TRACELENS_EXTENSION_FILE:=""}"
: "${HF_TOKEN:=""}"

# Other vars
readonly _config_file="${CONFIG_NAME}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${MODELDIR}:/model/" "--volume=$(pwd):/workspace/code" "--volume=$(pwd)/../../AMD:/workspace/AMD" "--volume=${UTILITIES}:/workspace/utilities" "--volume=${LOGDIR}:/results")


# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(MODELDIR)
_config_env+=(MODEL)
_config_env+=(DGXSYSTEM)
_config_env+=(PROFILER)
_config_env+=(LOGDIR)
_config_env+=(HIPBLASLT_LOG)
_config_env+=(GEMM_OFFLINE_TUNING)
_config_env+=(GEMM_USE_TUNING_RESULTS)
_config_env+=(HF_TOKEN)
_config_env+=(SEED)

if [[ $ENABLE_TRAINING_METRICS_COLLECTION -eq 1 ]]; then
    _config_env+=(LOG_FREQ)
fi

echo ${_config_env[@]}
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    else
        echo "Container ${_cont_name} does not exist. Skipping removal."
    fi
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
if [[ "${DGXSYSTEM}" == MI* ]]; then
  echo "Using AMD/ROCm container flags"
  docker run --rm --init --detach \
      --net=host --uts=host --ipc=host \
      --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
      --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
      --security-opt=seccomp=unconfined \
      --group-add video \
      --privileged \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity
else
  echo "Using NVIDIA container flags"
  docker run --rm --init --detach \
      --net=host --uts=host \
      --ipc=host --gpus all \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --device=/dev/infiniband \
      --group-add video \
      --privileged \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity
fi

# Make sure container has time to finish initialization
sleep 5
docker exec "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  # Use existing SEED if set; otherwise use a new RANDOM value for this run only
  _run_seed="${SEED:-$RANDOM}"
  _run_config_env=("${_config_env[@]}" "--env=SEED=${_run_seed}")
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi
    echo 'launching experiment using:'  "${_run_config_env[@]}" "${_cont_name}" /workspace/code/dev/run_and_time.sh
    docker exec "${_run_config_env[@]}" "${_cont_name}" bash /workspace/code/dev/run_and_time.sh
  ) | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log"

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_run_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log"
  fi

done

echo "Number of experiments $NEXP"

: "${BASE_DIR:=/workspace/AMD}"
: "${CODE_DIR:=/workspace/code}"
: "${MODEL_NAME:=llama3.1-8b}"
: "${SYSTEM:=MI355X_EPYC_9575F}"
: "${FRAMEWORK:=pytorch}"
: "${SYSTEM_JSON:=/workspace/AMD/systems/${SYSTEM}_${FRAMEWORK}.json}"

# Package submission
if [[ "${NEXP}" -ge 10 ]]; then
  echo "Running RCP check on /results"
  mkdir -p "${LOGDIR}/logs" && mv ${LOGDIR}/*.log "${LOGDIR}/logs/"
  docker exec "${_config_env[@]}" "${_cont_name}" python /workspace/utilities/select_best_runs.py ${MODEL_NAME} /results/logs /results/logs

  docker exec "${_config_env[@]}" "${_cont_name}" bash /workspace/utilities/package_mlperf_submission.sh --base_dir ${BASE_DIR} --output_dir /results --model ${MODEL_NAME} --code ${CODE_DIR} --results /results/logs --system ${SYSTEM} --system_json ${SYSTEM_JSON} --framework ${FRAMEWORK}

fi

if [[ $PROFILER = "torchprof" ]] ; then
    docker exec "${_config_env[@]}" "${_cont_name}" bash -c "
      bash /workspace/utilities/metrics/profiling/tracelens_perf_report.sh  $TORCHPROF_OUTPUT_DIR  $TRACELENS_EXTENSION_FILE
    "
fi

echo "ENABLE_TRAINING_METRICS_COLLECTION is set to $ENABLE_TRAINING_METRICS_COLLECTION"
if [[ $ENABLE_TRAINING_METRICS_COLLECTION -eq 1 ]]; then
  echo "Parsing training metrics"
  mkdir -p "${LOGDIR}/logs" && mv ${LOGDIR}/*.log "${LOGDIR}/logs/"
  docker exec "${_config_env[@]}" "${_cont_name}" python3 /workspace/utilities/metrics/training/parse_training_metrics.py --model_name "llama3.1-8b"
fi
