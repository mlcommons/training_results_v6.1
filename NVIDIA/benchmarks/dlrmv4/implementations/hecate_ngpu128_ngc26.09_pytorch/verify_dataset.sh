#!/usr/bin/env bash
# MLPerf Training reference script: verify the preprocessed dataset.
#
# Layout-checks every file training reads under ${DATADIR}, then (once
# real hashes are pinned) verifies them against md5sums_yambda_5b_processed.txt.
#
# Usage:
#   DATADIR=/path/to/dlrm_data ./verify_dataset.sh
#
# Env:
#   DATADIR           data root (required).
#   PROCESSED_SUBDIR  processed subdir under the data root (default: processed_5b).
set -euo pipefail

: "${DATADIR:?Set DATADIR to the data root}"
PROCESSED_SUBDIR="${PROCESSED_SUBDIR:-processed_5b}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKSUM_FILE="${REPO_ROOT}/md5sums_yambda_5b_processed.txt"

echo "[verify_dataset] data root: ${DATADIR}"

if [[ ! -d "${DATADIR}/${PROCESSED_SUBDIR}" ]]; then
    echo "[verify_dataset] ERROR: ${DATADIR}/${PROCESSED_SUBDIR} does not exist. Run ./download_dataset.sh first." >&2
    exit 1
fi

EXPECTED_FILES=(
    "${PROCESSED_SUBDIR}/train_sessions.parquet"
    "${PROCESSED_SUBDIR}/test_events.parquet"
    "${PROCESSED_SUBDIR}/session_index.parquet"
    "${PROCESSED_SUBDIR}/item_popularity.npy"
    "${PROCESSED_SUBDIR}/split_meta.json"
    "shared_metadata/artist_item_mapping.parquet"
    "shared_metadata/album_item_mapping.parquet"
)

missing=0
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -s "${DATADIR}/${f}" ]]; then
        echo "  OK   ${f}"
    else
        echo "  MISS ${f}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    echo "[verify_dataset] ERROR: one or more expected files are missing/empty." >&2
    exit 1
fi

if grep -qiE '^[0-9a-f]{32}[[:space:]]' "${CHECKSUM_FILE}"; then
    echo "[verify_dataset] checking md5 checksums from ${CHECKSUM_FILE}"
    # --strict: a placeholder/malformed manifest line fails instead of being skipped
    (cd "${DATADIR}" && md5sum -c --strict "${CHECKSUM_FILE}")
    echo "[verify_dataset] OK: all checksums match."
else
    echo "[verify_dataset] WARNING: ${CHECKSUM_FILE} contains placeholder hashes;" >&2
    echo "[verify_dataset]          layout check only (checksums NOT yet pinned)." >&2
fi
