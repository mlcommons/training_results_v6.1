#!/bin/bash
# One-time dataset and model preparation, unchanged from AMD's v5.1 implementation (see ../nemo/README_upstream_AMD_v5.1.md):
# downloads the tokenized, packed GovReport dataset and Llama-2-70B (needs a Hugging Face token with the Llama 2 licence
# accepted) and converts the model. Result: $DATADIR/data/{train,validation}.npy and $DATADIR/model/.
set -euo pipefail
IMPL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../nemo" && pwd)"
DATADIR="${DATADIR:-/data/mlperf_llama2}"
: "${HF_TOKEN:?set HF_TOKEN}"
mkdir -p "$DATADIR"
docker run --rm -it --ipc=host -e HF_TOKEN -v "$DATADIR:/data" -v "$IMPL/scripts:/workspace/code/scripts:ro" \
  krai4ai/mlperf:training_v6.1-llama2_70b_lora-mi300x bash -c "cd /workspace/code && bash scripts/prepare_data_and_model.sh"
