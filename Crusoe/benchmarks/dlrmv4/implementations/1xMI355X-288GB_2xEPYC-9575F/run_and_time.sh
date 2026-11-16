#!/bin/bash
# One timed run at this scale; result = the rank-0 log (full trainer log + MLLOG, MLPERF_VERBOSE_LOGS=1).
# Run from benchmarks/dlrmv4/implementations/pytorch_v61drop_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# dataset under datasets/dlrmv4/data, node-local scratch at /local on every node, and the rccl-roce-env ConfigMap.
# SELKEY/SELVAL: a node label selecting the 1 node(s) to run on.
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${SEED:?}" "${SELKEY:?}" "${SELVAL:?}"
cd "$(dirname "$0")/../pytorch_v61drop_k8s/k8s"
BENCH=dlrmv4 NN=1 GBS=8192 MBS=1024 EXTRA="MLPERF_VERBOSE_LOGS=1" bash run-cdrop.sh
