#!/bin/bash
# One-time: build and push the trainer image (AMD's v6.1 Dockerfile + libionic RoCE provider).
# Run from benchmarks/gpt_oss_20b/implementations/primus_v61drop_k8s_253c561/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# dataset under datasets/llama31_8b/{data,model}, and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${IMAGE:?}"
cd "$(dirname "$0")/../primus_v61drop_k8s_253c561"
cp k8s/Dockerfile.v61drop-libionic amd_v6.1_MI355X_EPYC_9575F_primus/Dockerfile.v61drop-libionic
docker build -f amd_v6.1_MI355X_EPYC_9575F_primus/Dockerfile.v61drop-libionic -t "$IMAGE" amd_v6.1_MI355X_EPYC_9575F_primus && docker push "$IMAGE"
