#!/bin/bash
# One-time dataset (and for LoRA, checkpoint) preparation into the shared volume.
# Run from benchmarks/llama31_8b/implementations/primus_k8s/k8s. Needs: kubectl context + namespace,
# an image built from Dockerfile.llama-libionic, the shared RWX volume with the dataset.
set -euo pipefail
: "${CTX:?}" "${NS:?}"
cd "$(dirname "$0")/../primus_k8s/k8s"
kubectl --context $CTX -n $NS apply -f prep-data-8b.yaml
