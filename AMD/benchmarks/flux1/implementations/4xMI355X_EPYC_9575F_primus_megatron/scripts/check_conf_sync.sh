#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# The recipe in conf/ is the one each run uses (EXP). It is a copy of the recipe that is
# versioned with the code in Primus; this check proves the two are identical inside the
# image, so a reviewer can read either. Three run settings may differ and are reported:
#   ddp_head_bucket_size     the package runs 128 MB, the Primus copy at the pinned commit 172 MB. The head bucket
#                            only groups the parameters for communication; it changes nothing that is computed.
#   mxfp6_fwd_fp4_heal_iter  the package heals from step 7325 (the joint blocks' text-stream MXFP4 forwards go back
#                            to MXFP6); the Primus copy's default is 0 (off). The code is the same.
#   exp_name                 the run's name (output directory only).
set -euo pipefail
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
SRC="${PRIMUS_PATH}/examples/megatron/configs/MI355X/diffusion"
status=0
check() {
    if cmp -s "$1" "$2"; then
        echo "[conf-sync] OK       $(basename "$1")"
    else
        echo "[conf-sync] MISMATCH $1 vs $2" >&2
        status=1
    fi
}
# Identical except for the run settings above (and comments).
RUN_SETTINGS='^[<>] *(ddp_head_bucket_size: [0-9]+|mxfp6_fwd_fp4_heal_iter: \$\{PRIMUS_MXFP6_FWD_FP4_HEAL_ITER:[0-9]+\}|exp_name: \$\{PRIMUS_EXP_NAME:[A-Za-z0-9_]+\})$'
check_but_run_settings() {
    local d
    d=$(diff <(grep -v '^ *#' "$1") <(grep -v '^ *#' "$2") | grep -E '^[<>]' | grep -vE "$RUN_SETTINGS" || true)
    if [[ -z "$d" ]]; then
        echo "[conf-sync] OK       $(basename "$1") (run settings: package" \
            "$(grep -oE 'ddp_head_bucket_size: [0-9]+|HEAL_ITER:[0-9]+' "$1" | tr '\n' ' ')/ Primus" \
            "$(grep -oE 'ddp_head_bucket_size: [0-9]+|HEAL_ITER:[0-9]+' "$2" | tr '\n' ' '))"
    else
        echo "[conf-sync] MISMATCH $1 vs $2 beyond the run settings:" >&2
        echo "$d" >&2
        status=1
    fi
}
check_but_run_settings "${CODE_DIR}/conf/flux1_mlperf_4n_gbs1024.yaml" \
      "${SRC}/flux_12b_ddp_energon_schnell_resample_local_spec_mxfp6_fp4_fast_mlperf_4n_gbs1024.yaml"
check "${CODE_DIR}/conf/nccl_comm_dp_max_ctas64.yaml" "${SRC}/nccl_comm_dp_max_ctas64.yaml"
echo "[conf-sync] Primus $(git -C "${PRIMUS_PATH}" rev-parse HEAD)"
exit $status
