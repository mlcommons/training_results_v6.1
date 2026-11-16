#!/bin/bash
# One-time: node-local RAID0 scratch at /local, the Yambda-5b HSTU dataset (MLCommons source + md5)
# onto the shared volume, and AMD's prebuilt time-window index next to it.
# Run from benchmarks/dlrmv4/implementations/pytorch_v61drop_k8s/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# dataset under datasets/dlrmv4/data, node-local scratch at /local on every node, and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${CTX:?}" "${NS:?}"
cd "$(dirname "$0")/../pytorch_v61drop_k8s/k8s"
: "${IMAGE:?}"
kubectl --context $CTX -n $NS apply -f local-raid0-ds.yaml
kubectl --context $CTX -n $NS apply -f dlrmv4-dataset-download.yaml
kubectl --context $CTX -n $NS wait --for=condition=complete --timeout=24h job/dlrmv4-dataset-download
sed "s|__IMAGE__|$IMAGE|" dlrmv4-index-build.yaml | kubectl --context $CTX -n $NS apply -f -
kubectl --context $CTX -n $NS wait --for=condition=complete --timeout=12h job/dlrmv4-index-build
