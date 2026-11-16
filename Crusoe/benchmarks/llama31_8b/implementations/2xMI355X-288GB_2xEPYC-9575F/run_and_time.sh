#!/bin/bash
# One timed run at this scale; result = the rank-0 log (full framework log + MLLOG).
# Run from benchmarks/llama31_8b/implementations/primus_v61drop_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# dataset under datasets/llama31_8b/{data,model}, and the rccl-roce-env ConfigMap.
# SELKEY/SELVAL: a node label selecting the 2 node(s) to run on.
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${SEED:?}" "${SELKEY:?}" "${SELVAL:?}"
cd "$(dirname "$0")/../primus_v61drop_k8s/k8s"
BENCH=llama31_8b NN=2 GBS=32 MBS=2 EXTRA="-MEGATRON_PARAM_GATHER_BACKEND MANUAL_GC=1" bash run-v61drop.sh
