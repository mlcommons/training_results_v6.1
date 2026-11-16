#!/bin/bash
# One-time: build the trainer images (kaniko, in-cluster) and the RCCL env ConfigMap.
# Run from benchmarks/llama31_8b/implementations/primus_k8s/k8s. Needs: kubectl context + namespace,
# an image built from Dockerfile.llama-libionic, the shared RWX volume with the dataset.
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${REGISTRY:?}" "${PUSH_SECRET:?}" "${NODE:?}"
cd "$(dirname "$0")/../primus_k8s/k8s"
kubectl --context $CTX -n $NS create configmap llama-libionic-dockerfile --from-file=Dockerfile=Dockerfile.llama-libionic \
  --dry-run=client -o yaml | kubectl --context $CTX -n $NS apply -f -
sed -e "s|__REGISTRY__|$REGISTRY|g; s|__PUSH_SECRET__|$PUSH_SECRET|g; s|__NODE_8B__|$NODE|g; s|__NODE_LORA__|$NODE|g" \
  build-llama-images-kaniko.yaml | kubectl --context $CTX -n $NS apply -f -
