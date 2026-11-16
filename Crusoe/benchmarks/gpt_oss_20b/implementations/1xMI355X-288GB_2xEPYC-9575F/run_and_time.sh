#!/bin/bash
# One timed run at this scale; result = the rank-0 log (trainer full log + launcher lines + MLLOG).
# Run from benchmarks/gpt_oss_20b/implementations/primus_v61drop_k8s_253c561/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# dataset under datasets/llama31_8b/{data,model}, and the rccl-roce-env ConfigMap.
# SELKEY/SELVAL: a node label selecting the 1 node(s) to run on.
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${SEED:?}" "${SELKEY:?}" "${SELVAL:?}"
cd "$(dirname "$0")/../primus_v61drop_k8s_253c561/k8s"
BENCH=gpt_oss_20b NN=1 GBS=32 MBS=4 EXTRA="" bash run-v61drop.sh
