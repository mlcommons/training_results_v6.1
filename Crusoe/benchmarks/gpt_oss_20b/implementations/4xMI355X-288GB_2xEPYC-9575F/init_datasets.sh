#!/bin/bash
# One-time: pre-tokenized C4 + tokenizer onto the shared volume (MLCommons R2 downloader).
# Run from benchmarks/gpt_oss_20b/implementations/primus_v61drop_k8s_253c561/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# dataset under datasets/llama31_8b/{data,model}, and the rccl-roce-env ConfigMap.
set -euo pipefail
: "${CTX:?}" "${NS:?}"
cd "$(dirname "$0")/../primus_v61drop_k8s_253c561/k8s"
kubectl --context $CTX -n $NS apply -f prep-data-8b.yaml
kubectl --context $CTX -n $NS wait --for=condition=complete --timeout=12h job/prep-llama31-8b-data
