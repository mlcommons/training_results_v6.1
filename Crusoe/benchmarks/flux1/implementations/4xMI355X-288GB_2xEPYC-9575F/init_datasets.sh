#!/bin/bash
# One-time: the MLCommons flux1 preprocessed dataset onto the shared volume (same download job as the
# pytorch_k8s implementation), then an md5 check of the directories AMD's v6.1 recipe reads.
# Run from benchmarks/flux1/implementations/primus_v61drop_k8s_253c561/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# preprocessed dataset under datasets/flux1/{cc12m_preprocessed,coco_preprocessed,empty_encodings}, node-local
# scratch at /local on every node (inductor cache), and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${CTX:?}" "${NS:?}"
cd "$(dirname "$0")/.."
kubectl --context $CTX -n $NS apply -f pytorch_k8s/k8s/dataset-finish-download.yaml
kubectl --context $CTX -n $NS wait --for=condition=complete --timeout=24h job/flux-finish-download
kubectl --context $CTX -n $NS apply -f primus_v61drop_k8s_253c561/k8s/flux1-dataset-verify.yaml
kubectl --context $CTX -n $NS wait --for=condition=complete --timeout=6h job/flux1-dataset-verify
