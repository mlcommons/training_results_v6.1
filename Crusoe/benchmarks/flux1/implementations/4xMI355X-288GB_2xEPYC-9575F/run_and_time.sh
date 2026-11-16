#!/bin/bash
# One timed run at this scale; result = the rank-0 log (trainer stdout + MLLOG).
# Run from benchmarks/flux1/implementations/primus_v61drop_k8s_253c561/k8s. Needs: kubectl context + namespace, an image
# built from Dockerfile.v61drop-libionic, the shared RWX volume (PVC mlperf-training-shared) with the
# preprocessed dataset under datasets/flux1/{cc12m_preprocessed,coco_preprocessed,empty_encodings}, node-local
# scratch at /local on every node (inductor cache), and the rccl-roce-env ConfigMap.
# SELKEY/SELVAL: a node label selecting the 4 node(s) to run on. PRIV=true: privileged pods
# (page-cache flush). The inductor cache lives on node-local disk (INDUCTOR_LOCAL=1).
set -euo pipefail
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${SEED:?}" "${SELKEY:?}" "${SELVAL:?}"
cd "$(dirname "$0")/../primus_v61drop_k8s_253c561/k8s"
PRIV=true BENCH=flux1 NN=4 GBS=1024 MBS=32 EXTRA="NCCL_RMA_DISABLE=1 FSDP_ALL_GATHER_BACKEND=rccl INDUCTOR_LOCAL=1 FLUX_LOCAL_DATA=1" bash run-cdrop.sh
