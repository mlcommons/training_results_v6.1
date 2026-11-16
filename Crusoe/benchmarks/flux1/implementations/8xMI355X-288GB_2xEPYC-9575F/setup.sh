#!/bin/bash
# One-time: build and push the trainer image (AMD's v6.1 Dockerfile + libionic RoCE provider).
# Run from benchmarks/flux1/implementations/primus_v61drop_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# preprocessed dataset under datasets/flux1/{cc12m_preprocessed,coco_preprocessed,empty_encodings}, node-local
# scratch at /local on every node (inductor cache), and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${IMAGE:?}"
cd "$(dirname "$0")/../primus_v61drop_k8s"
cp k8s/Dockerfile.v61drop-libionic amd_v6.1_MI355X_EPYC_9575F_primus/Dockerfile.v61drop-libionic
docker build -f amd_v6.1_MI355X_EPYC_9575F_primus/Dockerfile.v61drop-libionic -t "$IMAGE" amd_v6.1_MI355X_EPYC_9575F_primus && docker push "$IMAGE"
