#!/bin/bash
# One-time: build and push the trainer image (AMD's v6.0 flux1 image + libionic RoCE provider).
# Run from benchmarks/flux1/implementations/pytorch_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.flux-trainer-bundle2.1, the shared RWX volume (PVC mlperf-training-shared)
# with the dataset under datasets/flux1/energon, and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${IMAGE:?}"
cd "$(dirname "$0")/../pytorch_k8s/k8s"
docker build -f Dockerfile.flux-trainer-bundle2.1 -t "$IMAGE" . && docker push "$IMAGE"
