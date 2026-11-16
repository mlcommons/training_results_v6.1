#!/bin/bash
# init_datasets.sh -- one-time dataset preparation for llama31_8b (NVIDIA's procedure, README_NVIDIA.md section 3.2).
#
#   DATADIR=/path/to/data bash init_datasets.sh
#
# Downloads the MLCommons preprocessed C4 dataset and the Llama 3.1 8B tokenizer with NVIDIA's data_scripts/download_8b.sh
# into $DATADIR/8b/, then checks the files against the md5 list that comes with the dataset.  Every node that runs the
# benchmark needs this tree at the same $DATADIR (the results were produced with $DATADIR on node-local NVMe).
set -euo pipefail
: "${DATADIR:?set DATADIR to the dataset root}"
export DATADIR
IMPL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo_l1_1012" && pwd)
cd "$IMPL"
bash data_scripts/download_8b.sh
cd "$DATADIR/8b"
ls -la
if [ -f llama-3-1-8b-preprocessed-c4-dataset.md5 ]; then
  md5sum -c llama-3-1-8b-preprocessed-c4-dataset.md5
fi
