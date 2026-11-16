#!/bin/bash
# One-time: build and push the trainer image (AMD's v6.1 Dockerfile + libionic RoCE provider).
# Run from benchmarks/dlrmv4/implementations/pytorch_v61drop_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# dataset under datasets/dlrmv4/data, node-local scratch at /local on every node, and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${IMAGE:?}"
cd "$(dirname "$0")/../pytorch_v61drop_k8s"
cp k8s/Dockerfile.v61drop-libionic amd_v6.1_MI355X_EPYC_9575F_hstu/Dockerfile.v61drop-libionic
docker build -f amd_v6.1_MI355X_EPYC_9575F_hstu/Dockerfile.v61drop-libionic -t "$IMAGE" amd_v6.1_MI355X_EPYC_9575F_hstu && docker push "$IMAGE"
