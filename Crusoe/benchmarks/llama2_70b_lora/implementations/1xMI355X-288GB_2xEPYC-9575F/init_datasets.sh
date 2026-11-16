#!/bin/bash
# One-time dataset (and for LoRA, checkpoint) preparation into the shared volume.
# Run from benchmarks/llama2_70b_lora/implementations/primus_k8s/k8s. Needs: kubectl context + namespace,
# an image built from Dockerfile.llama-libionic, the shared RWX volume with the dataset.
set -euo pipefail
: "${CTX:?}" "${NS:?}"
cd "$(dirname "$0")/../primus_k8s/k8s"
: "${IMAGE:?}" "${PULL_SECRET:?}" "${HF_SECRET:?}" "${NODE:?}"
sed -e "s|__IMAGE__|$IMAGE|; s|__PULL_SECRET__|$PULL_SECRET|; s|__HF_SECRET__|$HF_SECRET|; s|__NODE__|$NODE|" prep-data-lora.yaml \
  | kubectl --context $CTX -n $NS apply -f -
