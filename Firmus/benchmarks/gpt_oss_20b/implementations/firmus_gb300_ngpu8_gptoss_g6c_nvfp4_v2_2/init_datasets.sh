#!/bin/bash
# One-time dataset preparation with the MLCommons reference download scripts
# (preprocessed C4 and tokenizer, shared with llama31_8b), then checksum verification.
#   DATADIR=<path on durable storage> ./init_datasets.sh
set -euo pipefail

: "${DATADIR:?set DATADIR to the dataset root}"
IMPL="$(cd "$(dirname "$0")/../nemo_g6c_nvfp4_v2_2" && pwd)"

cd "${IMPL}"
bash data_scripts/download.sh
cd "${DATADIR}/8b"
md5sum -c llama-3-1-8b-preprocessed-c4-dataset.md5
