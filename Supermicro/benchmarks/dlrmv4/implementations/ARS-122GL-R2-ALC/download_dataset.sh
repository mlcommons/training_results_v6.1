#!/usr/bin/env bash
# MLPerf Training reference script: download + preprocess the dataset.
#
# Downloads the Yambda dataset from HuggingFace (yandex/yambda) and runs the
# preprocessing pipeline (event-type encoding, temporal GTS split, session
# segmentation, item-popularity counts) into the on-disk layout that
# DLRMv4YambdaDataset consumes. This is a thin wrapper over
#   generative_recommenders.dlrm_v4.preprocess_public_data
# so the full reference data pipeline lives in one place.
#
# Usage:
#   DATADIR=/path/to/dlrm_data ./download_dataset.sh
#   DATASET=yambda-50m DATADIR=/path/to/dlrm_data ./download_dataset.sh
#
# Env:
#   DATASET         dataset variant (default: yambda-5b). One of
#                   kuairand-1k | kuairand-27k | yambda-50m | yambda-500m | yambda-5b
#   DATADIR  destination data root (required). The same directory is mounted
#            at /data when training is launched through run.sub.
set -euo pipefail

DATASET="${DATASET:-yambda-5b}"
: "${DATADIR:?Set DATADIR to the destination data root}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "[download_dataset] dataset=${DATASET} data-path=${DATADIR}"
mkdir -p "${DATADIR}"

python3 -m generative_recommenders.dlrm_v4.preprocess_public_data \
    --dataset "${DATASET}" \
    --data-path "${DATADIR}"

echo "[download_dataset] done. Preprocessed layout under ${DATADIR}:"
echo "  raw/5b/multi_event.parquet"
echo "  shared_metadata/{artist,album}_item_mapping.parquet, embeddings.parquet"
echo "  processed_5b/{train_sessions,test_events,session_index}.parquet"
echo "  processed_5b/item_popularity.npy, processed_5b/split_meta.json"
echo "[download_dataset] verify integrity with: ./verify_dataset.sh"
