#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Prepare the MLPerf flux1 training and evaluation data for Megatron-Energon.
#
# Runs inside the submission image. Downloads the MLCommons-preprocessed CC12M training
# set, the COCO-2014 evaluation subset (with its per-sample evaluation timesteps) and the
# empty encodings from the official MLCommons storage, converts them to Energon WebDataset
# shards and writes the Energon metadata, all into one dataset root:
#
#   ${DATADIR}/mlperf_flux1/{train,val,empty_encodings,.nv-meta}
#
# Streaming: each Arrow file is downloaded, converted and deleted before the next one
# lands (a few files are prefetched), so peak disk use is the final dataset size plus a
# small buffer, rather than raw download plus converted copy.
#
# Usage (from the host):
#   docker run --rm -v ${DATADIR}:/data <image> bash /workspace/code/scripts/prepare_data.sh
# Options pass through to `primus data diffusion-ingest` (e.g. --max-files 2 for a test).
set -euo pipefail
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
OUT="${DATA_OUT:-/data/mlperf_flux1}"
cd "${PRIMUS_PATH}"
# One process: the ingest parallelises downloads with threads and is not rank-aware, so it
# must not be started through torchrun (primus-cli direct launches one process per GPU).
python3 "${PRIMUS_PATH}/primus/cli/main.py" data diffusion-ingest \
    --config primus/configs/data/megatron/diffusion/preprocessing/mlperf_flux1.yaml \
    --output-dir "${OUT}" "$@"
rm -rf "${OUT}/_arrow_tmp"
echo "[prepare_data] done: ${OUT}"
echo "[prepare_data] verify with: bash /workspace/code/scripts/verify_data.sh ${OUT}"
