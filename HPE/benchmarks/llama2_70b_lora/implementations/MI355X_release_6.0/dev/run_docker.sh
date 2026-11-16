#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR/.."
WORKSPACE_DIR=$(pwd)

docker run -it --rm \
    --net=host --uts=host --ipc=host \
    --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt=seccomp=unconfined \
    --group-add video \
    --privileged \
    --volume "${WORKSPACE_DIR}:/workspace/code/" \
    --volume /data/mlperf_llama2/data:/data \
    --volume "${WORKSPACE_DIR}/logs:/results" \
    --volume="${WORKSPACE_DIR}/../../AMD:/workspace/AMD" \
    --volume="${WORKSPACE_DIR}/../../utilities:/workspace/utilities" \
    --name "llama2-70b-sft-training-$(whoami)" mlperf_llama2-70b-sft:v1_dev
