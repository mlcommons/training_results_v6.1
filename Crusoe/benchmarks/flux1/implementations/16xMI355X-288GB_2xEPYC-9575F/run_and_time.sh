#!/bin/bash
# One timed run at this scale; result = the rank-0 log (full framework log + MLLOG).
# Run from benchmarks/flux1/implementations/pytorch_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.flux-trainer-bundle2.1, the shared RWX volume (PVC mlperf-training-shared)
# with the dataset under datasets/flux1/energon, and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${SEED:?}" "${SELKEY:?}" "${SELVAL:?}"
cd "$(dirname "$0")/../pytorch_k8s/k8s"
N=16 bash run-flux-compliance.sh A "$SEED"
