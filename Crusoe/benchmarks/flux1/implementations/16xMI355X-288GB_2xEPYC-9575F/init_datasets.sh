#!/bin/bash
# One-time dataset preparation onto the shared volume: reference download, then Energon conversion.
# Run from benchmarks/flux1/implementations/pytorch_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.flux-trainer-bundle2.1, the shared RWX volume (PVC mlperf-training-shared)
# with the dataset under datasets/flux1/energon, and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${CTX:?}" "${NS:?}"
cd "$(dirname "$0")/../pytorch_k8s/k8s"
kubectl --context $CTX -n $NS apply -f dataset-finish-download.yaml
kubectl --context $CTX -n $NS wait --for=condition=complete --timeout=24h job/flux-finish-download
kubectl --context $CTX -n $NS apply -f energon-convert.yaml
