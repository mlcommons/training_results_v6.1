#!/bin/bash
# Make our DLRMv4 data dir from the shared dataset. The shared copy is not changed.
#
# The v6.1 code reads <DATADIR>/processed_5b/hstu_cache/. The shared dataset keeps the
# same cache files in processed_5b/hstu_cache_L4086/ (older code layout) and has no
# parquets, so the code would try to rebuild the cache and fail. This script:
#   1. makes a real processed_5b/hstu_cache/ dir with symlinks to the shared cache files,
#   2. builds the offline time-window indices in that dir with the container builder
#      (offline_ts_indices.py, pure numpy), so no run builds them in its init phase.
#
# anchor_ts_L4086.npy is not linked. The v6.1 code does not write it, so a fresh v6.1
# cache builds the indices from flat_timestamps[positions]. We do the same.
#
# Usage: sbatch make_datadir.sh [dest]   (reads about 120 GB from /blue)
#SBATCH -A rc-rse
#SBATCH -p hpg-default
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J dlrmv4_datadir
#SBATCH -o /blue/rc-rse/mlperf/training/minh_training/v6.1/slurm_logs/%x_%j.out
set -euo pipefail

HERE=/blue/rc-rse/mlperf/training/minh_training/v6.1/dlrmv4
SRC=/blue/rc-rse/mlperf/training/training/dlrmv4/preprocessed_data/dlrmv4_dataset
DEST=${1:-/blue/rc-rse/mlperf/training/minh_training/v6.1/dlrmv4_data}
HISTORY_LENGTH=4086   # = HISTORY_LENGTH and MIN_HISTORY in config_common.sh
WINDOW_SECONDS=86400  # = get_dataset.streaming_window_seconds in yambda_5b.gin

cache_src="${SRC}/processed_5b/hstu_cache_L${HISTORY_LENGTH}"
cache_dst="${DEST}/processed_5b/hstu_cache"
mkdir -p "${cache_dst}"
ln -sfn "${SRC}/shared_metadata" "${DEST}/shared_metadata"
ln -sf "${SRC}/processed_5b/item_popularity.npy" "${DEST}/processed_5b/item_popularity.npy"
# _MMAP_COLS in datasets/yambda.py, the positions file, and the cache metadata
for name in flat_uid flat_item_ids flat_timestamps flat_event_types flat_played_ratio \
            flat_is_listen_plus flat_is_like flat_is_skip flat_is_organic \
            user_start user_end unique_uids "positions_L${HISTORY_LENGTH}"; do
    [[ -s "${cache_src}/${name}.npy" ]] || { echo "missing ${cache_src}/${name}.npy" >&2; exit 1; }
    ln -sf "${cache_src}/${name}.npy" "${cache_dst}/${name}.npy"
done
ln -sf "${cache_src}/store_meta.json" "${cache_dst}/store_meta.json"
ln -sf "${cache_src}/_READY" "${cache_dst}/_READY"

module load python/3.12
date
python3 -I "${HERE}/generative_recommenders/dlrm_v4/datasets/offline_ts_indices.py" \
    --cache-dir "${cache_dst}" \
    --history-length "${HISTORY_LENGTH}" --min-history "${HISTORY_LENGTH}" \
    --window-seconds "${WINDOW_SECONDS}"
date
ls -la "${DEST}" "${DEST}/processed_5b" "${cache_dst}"
