#!/bin/bash
# One timed run at this scale; result = the rank-0 log (full framework log + MLLOG).
# Run from benchmarks/llama2_70b_lora/implementations/primus_k8s/k8s. Needs: kubectl context + namespace,
# an image built from Dockerfile.llama-libionic, the shared RWX volume with the dataset.
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${SEED:?}"
cd "$(dirname "$0")/../primus_k8s/k8s"
BENCH=llama2_70b_lora NN=2 GBS=8 MBS=1 CP=2 LR=0.0006 MANUAL_GC=1 bash run-llama-mlperf.sh
