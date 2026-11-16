#!/bin/bash
# init_datasets.sh -- one-time dataset preparation for gpt_oss_20b (NVIDIA's procedure, README_NVIDIA.md section 3.2).
#
#   DATADIR=/path/to/data bash init_datasets.sh
#
# gpt_oss_20b uses the same dataset as llama31_8b (README_NVIDIA.md 3.2): the MLCommons preprocessed C4 dataset and the
# Llama 3.1 8B tokenizer.  NVIDIA's data_scripts/download.sh downloads both and data_scripts/cleanup.sh moves them to
# $DATADIR/8b/ and $DATADIR/8b/tokenizer/ (config_mounts.sh: DATA_FOLDER=8b); the files are then checked against the md5
# list that comes with the dataset.  Every node that runs the benchmark needs this tree at the same $DATADIR (the results
# were produced with $DATADIR on node-local NVMe).
set -euo pipefail
: "${DATADIR:?set DATADIR to the dataset root}"
export DATADIR
IMPL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo_g1_1012" && pwd)
cd "$IMPL"
bash data_scripts/download.sh
cd "$DATADIR/8b"
ls -la
if [ -f llama-3-1-8b-preprocessed-c4-dataset.md5 ]; then
  md5sum -c llama-3-1-8b-preprocessed-c4-dataset.md5
fi
