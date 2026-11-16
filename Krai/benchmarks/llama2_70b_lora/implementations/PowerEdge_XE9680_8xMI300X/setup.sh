#!/bin/bash
# One-time setup: build the container and put the host in the tuned state used for the results.
set -euo pipefail
IMPL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo" && pwd)"
docker pull rocm/amd-mlperf:llama2_70b_training_5.1
docker build -t krai4ai/mlperf:training_v6.1-llama2_70b_lora-mi300x "$IMPL"
# AMD's host tuning (performance governor, C1E off, THP always, NUMA balancing / NMI watchdog / ASLR off); needs root
# and is lost on every reboot.
sudo bash "$IMPL/runtime_tunables.sh"
