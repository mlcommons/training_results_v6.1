#!/usr/bin/env bash
                                                                    
 
                                                                      
                                                                                
 
        
                                                  
 
      
                                           
                                                                                   
set -euo pipefail

: "${DATADIR:?Set DATADIR to the data root}"
PROCESSED_SUBDIR="${PROCESSED_SUBDIR:-processed_5b}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKSUM_FILE="${REPO_ROOT}/md5sums_yambda_5b_processed.txt"

echo "[verify_dataset] data root: ${DATADIR}"

HISTORY_LENGTH="${HISTORY_LENGTH:-4086}"
STREAMING_WINDOW_SECONDS="${STREAMING_WINDOW_SECONDS:-86400}"
CACHE_DIR="${DATADIR}/${PROCESSED_SUBDIR}/hstu_cache_L${HISTORY_LENGTH}"
PREPARED_CHECKSUM_FILE="${DATADIR}/dlrmv4_dataset.md5"

if [[ -f "${CACHE_DIR}/_READY" && -f "${PREPARED_CHECKSUM_FILE}" ]]; then
    echo "[verify_dataset] detected prepared mmap cache: ${CACHE_DIR}"
    echo "[verify_dataset] checking prepared cache manifest"
    (cd "${DATADIR}" && md5sum -c --strict "${PREPARED_CHECKSUM_FILE}")

    REQUIRED_CACHE_FILES=(
        "${CACHE_DIR}/store_meta.json"
        "${CACHE_DIR}/positions_L${HISTORY_LENGTH}.npy"
        "${CACHE_DIR}/anchor_ts_L${HISTORY_LENGTH}.npy"
        "${CACHE_DIR}/ts_indices_L${HISTORY_LENGTH}_W${STREAMING_WINDOW_SECONDS}.npy"
        "${CACHE_DIR}/ts_offsets_L${HISTORY_LENGTH}_W${STREAMING_WINDOW_SECONDS}.npy"
        "${DATADIR}/${PROCESSED_SUBDIR}/item_popularity.npy"
        "${DATADIR}/shared_metadata/artist_item_mapping.parquet"
        "${DATADIR}/shared_metadata/album_item_mapping.parquet"
    )
    missing=0
    for file in "${REQUIRED_CACHE_FILES[@]}"; do
        if [[ -s "${file}" ]]; then
            echo "  OK   ${file#${DATADIR}/}"
        else
            echo "  MISS ${file#${DATADIR}/}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        echo "[verify_dataset] ERROR: prepared cache is incomplete." >&2
        exit 1
    fi
    echo "[verify_dataset] OK: prepared cache checksums and required files match."
    exit 0
fi

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
                                                                                    
    (cd "${DATADIR}" && md5sum -c --strict "${CHECKSUM_FILE}")
    echo "[verify_dataset] OK: all checksums match."
else
    echo "[verify_dataset] WARNING: ${CHECKSUM_FILE} contains placeholder hashes;" >&2
    echo "[verify_dataset]          layout check only (checksums NOT yet pinned)." >&2
fi
