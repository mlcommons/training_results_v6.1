#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Verify a prepared flux1 dataset root: shard and sample counts, the evaluation timesteps,
# the empty encodings, and (if present) the sha256 manifest of every shard.
#
# Usage: verify_data.sh <dataset_root> [--write-manifest]
#   --write-manifest regenerates scripts/expected_data_manifest.sha256 from this root.
set -euo pipefail
ROOT="${1:?usage: verify_data.sh <dataset_root> [--write-manifest]}"
MODE="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${HERE}/expected_data_manifest.sha256"
JOBS="${VERIFY_JOBS:-32}"
EXP_TRAIN_SHARDS=4762; EXP_VAL_SHARDS=129
EXP_TRAIN_SAMPLES=1099776; EXP_VAL_SAMPLES=29696
fail=0
ok()  { echo "[verify_data] OK   $*"; }
bad() { echo "[verify_data] FAIL $*" >&2; fail=1; }

n_train=$(find -L "${ROOT}/train" -maxdepth 1 -name '*.tar' | wc -l)
n_val=$(find -L "${ROOT}/val" -maxdepth 1 -name '*.tar' | wc -l)
[[ $n_train == "$EXP_TRAIN_SHARDS" ]] && ok "train shards $n_train" || bad "train shards $n_train != $EXP_TRAIN_SHARDS"
[[ $n_val == "$EXP_VAL_SHARDS" ]] && ok "val shards $n_val" || bad "val shards $n_val != $EXP_VAL_SHARDS"
for f in empty_encodings/t5_empty.npy empty_encodings/clip_empty.npy .nv-meta/split.yaml .nv-meta/dataset.yaml .nv-meta/.info.json .nv-meta/index.sqlite; do
    [[ -f "${ROOT}/$f" ]] && ok "$f" || bad "missing $f"
done

# Sample counts and evaluation timesteps, from the shard contents themselves.
python3 - "$ROOT" "$EXP_TRAIN_SAMPLES" "$EXP_VAL_SAMPLES" "$JOBS" <<'PY' || fail=1
import json, sys, tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root, exp_train, exp_val, jobs = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def count(tar_path, want_timestep):
    keys, ts_missing = set(), 0
    with tarfile.open(tar_path) as t:
        for m in t:
            key, _, ext = m.name.partition(".")
            keys.add(key)
            if want_timestep and ext == "json":
                if "timestep" not in json.load(t.extractfile(m)):
                    ts_missing += 1
    return len(keys), ts_missing
rc = 0
for split, exp, want_ts in (("train", exp_train, False), ("val", exp_val, True)):
    shards = sorted((root / split).glob("*.tar"))
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(lambda p: count(p, want_ts), shards))
    n = sum(r[0] for r in res); missing = sum(r[1] for r in res)
    tag = "OK  " if n == exp else "FAIL"
    print(f"[verify_data] {tag} {split} samples {n} (expected {exp})")
    rc |= n != exp
    if want_ts:
        print(f"[verify_data] {'OK  ' if missing == 0 else 'FAIL'} val samples without an evaluation timestep: {missing}")
        rc |= missing != 0
sys.exit(rc)
PY

if [[ "$MODE" == "--write-manifest" ]]; then
    (cd "$ROOT" && find -L train val empty_encodings -type f \( -name '*.tar' -o -name '*.npy' \) | sort | xargs -P "$JOBS" -n 16 sha256sum | sort -k2) > "$MANIFEST"
    ok "wrote $(wc -l < "$MANIFEST") entries to $MANIFEST"
elif [[ -s "$MANIFEST" ]]; then
    # Check in parallel: split the manifest into JOBS parts and verify each part.
    parts=$(mktemp -d)
    split -n "l/${JOBS}" "$MANIFEST" "${parts}/part_"
    if (cd "$ROOT" && ls "${parts}"/part_* | xargs -P "$JOBS" -n 1 sha256sum --quiet -c); then
        ok "sha256 manifest ($(wc -l < "$MANIFEST") files)"
    else
        bad "sha256 manifest mismatch"
    fi
    rm -rf "$parts"
else
    echo "[verify_data] (no manifest present; counts and timesteps only)"
fi
[[ $fail == 0 ]] && echo "[verify_data] PASS" || { echo "[verify_data] FAILED" >&2; exit 1; }
