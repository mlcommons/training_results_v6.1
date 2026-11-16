#!/bin/bash
# One timed run at this scale; result = the rank-0 log (full framework log + MLLOG).
# Run from benchmarks/llama31_8b/implementations/primus_k8s/k8s. Needs: kubectl context + namespace,
# an image built from Dockerfile.llama-libionic, the shared RWX volume with the dataset.
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${SEED:?}"
cd "$(dirname "$0")/../primus_k8s/k8s"
BENCH=llama31_8b NN=8 GBS=64 MBS=1 CP=1 MANUAL_GC=1 bash run-llama-mlperf.sh
