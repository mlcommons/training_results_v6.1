# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# pyre-unsafe
'' 
                                                   

                                                                 
                                                                      
                                                                         
                                                                          
                                                                     
                                                                         
   

import logging
import mmap as _mmap_mod
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
import torch
from generative_recommenders.dlrm_v4.datasets.dataset import DLRMv4RandomDataset
from generative_recommenders.dlrm_v4.datasets.utils import xxhash_cross
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger = logging.getLogger(__name__)


def _load_npy_readonly(path: Union[str, Path]) -> np.ndarray:
                                                                              
                                                                               
                                                                         
    path = Path(path)
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version[0] == 1:
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    fd = os.open(str(path), os.O_RDONLY)
    try:
        buf = _mmap_mod.mmap(fd, 0, access=_mmap_mod.ACCESS_READ)
    finally:
        os.close(fd)
    arr = np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)
    arr.flags.writeable = False
    return arr


def _uid_unit_hash(uids: np.ndarray, salt: int) -> np.ndarray:
    ''                                                                        

                                                                             
                                                                          
                                                                            
                                                                               
                             
       
    GOLDEN = np.uint64(0x9E3779B97F4A7C15)
    M1 = np.uint64(0xBF58476D1CE4E5B9)
    M2 = np.uint64(0x94D049BB133111EB)
    s30, s27, s31 = np.uint64(30), np.uint64(27), np.uint64(31)
    with np.errstate(over="ignore"):
        z = uids.astype(np.uint64) + GOLDEN + np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> s30)) * M1
        z = (z ^ (z >> s27)) * M2
        z = z ^ (z >> s31)
                                                                               
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)

                                                                  
LISTEN_TYPE = 0
LIKE_TYPE = 1
LISTEN_PLUS_THRESHOLD = 50

                                                                         
LP_BIT = 1
LIKE_BIT = 2
SKIP_BIT = 4


class _FlatEventStore:
    ''                                                                        

                                                                            
                                                                          
                                                                         
       

                            
    _MMAP_COLS = (
        "flat_uid", "flat_item_ids", "flat_timestamps",
        "flat_event_types", "flat_played_ratio",
        "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
        "flat_is_organic",
        "user_start", "user_end", "unique_uids",
    )

    def __init__(self, sessions_df: pl.DataFrame) -> None:
        logger.info("Building flat event store from sessions...")
        sorted_sessions = sessions_df.sort(["uid", "session_id"])
        exploded = sorted_sessions.explode(
            ["item_ids", "timestamps", "event_types", "is_organic", "played_ratio_pct"]
        )

        self.flat_uid: np.ndarray = exploded["uid"].to_numpy().astype(np.int64)
        self.flat_item_ids: np.ndarray = exploded["item_ids"].to_numpy().astype(np.int64)
        self.flat_timestamps: np.ndarray = exploded["timestamps"].to_numpy().astype(np.int64)
        self.flat_event_types: np.ndarray = exploded["event_types"].to_numpy().astype(np.int64)
        self.flat_played_ratio: np.ndarray = exploded["played_ratio_pct"].to_numpy().astype(np.float32)
        self.flat_is_organic: np.ndarray = exploded["is_organic"].to_numpy().astype(np.int8)
        np.nan_to_num(self.flat_played_ratio, copy=False, nan=0.0)

        is_listen = self.flat_event_types == LISTEN_TYPE
        self.flat_is_listen_plus: np.ndarray = is_listen & (
            self.flat_played_ratio >= LISTEN_PLUS_THRESHOLD
        )
        self.flat_is_like: np.ndarray = self.flat_event_types == LIKE_TYPE
        self.flat_is_skip: np.ndarray = is_listen & (
            self.flat_played_ratio < LISTEN_PLUS_THRESHOLD
        )

        uid_changes = np.where(np.diff(self.flat_uid) != 0)[0] + 1
        starts = np.concatenate([[0], uid_changes])
        ends = np.concatenate([uid_changes, [len(self.flat_uid)]])
        uid_vals = self.flat_uid[starts]
        max_uid = int(uid_vals.max()) + 1
        self.user_start: np.ndarray = np.full(max_uid, -1, dtype=np.int64)
        self.user_end: np.ndarray = np.full(max_uid, -1, dtype=np.int64)
        self.user_start[uid_vals] = starts
        self.user_end[uid_vals] = ends
        self.unique_uids: np.ndarray = uid_vals
        self.num_users: int = len(uid_vals)
        self.total_events: int = len(self.flat_item_ids)
        logger.info(
            f"FlatEventStore: {self.total_events:,} events, {self.num_users:,} users"
        )

    @classmethod
    def load_mmap(cls, cache_dir: Union[str, Path]) -> "_FlatEventStore":
        ''                                                                 
                                                             
        import json as _json
        cache_dir = Path(cache_dir)
        with open(cache_dir / "store_meta.json") as f:
            meta = _json.load(f)
        store = object.__new__(cls)
        for name in cls._MMAP_COLS:
            setattr(store, name, _load_npy_readonly(cache_dir / f"{name}.npy"))
        store.num_users = int(meta["num_users"])
        store.total_events = int(meta["total_events"])
        logger.info(
            f"FlatEventStore mmap from {cache_dir}: "
            f"{store.total_events:,} events, {store.num_users:,} users"
        )
        return store

    def save_mmap(self, cache_dir: Union[str, Path]) -> None:
        ''                                                             
                                                                             
        import json as _json
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name in self._MMAP_COLS:
            np.save(cache_dir / f"{name}.npy", getattr(self, name))
        with open(cache_dir / "store_meta.json", "w") as f:
            _json.dump(
                {"num_users": self.num_users, "total_events": self.total_events}, f
            )
                                                                               
        (cache_dir / "_READY").touch()
        logger.info(f"FlatEventStore saved to {cache_dir}")


class DLRMv4YambdaDataset(DLRMv4RandomDataset):
    ''                                                     

         
                                                                                     
                                                                                       
                                                                           
                                                                           
                                                                               
                                               
                                                                 
                                                                                 
                                                                               
                                                                                
                                                          
                                                   
       

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        processed_dir: str,
        metadata_dir: str,
        history_length: int = 2048,
        scan_window: int = 20000,
        min_history: Optional[int] = None,
        history_strategy: str = "interleaved",
        cross_specs: Optional[Sequence[Tuple[str, Sequence[str], int, int]]] = None,
        cache_dir: Optional[str] = None,
        is_inference: bool = False,
        streaming_window_seconds: int = 86400,
        streaming_sort_within_window: bool = False,
        streaming_shuffle_fraction: float = 0.0,
        streaming_shuffle_seed: int = 0,
        train_split_percentage: float = 1.0,
        split_salt: int = 0,
        use_offline_ts_indices: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(hstu_config=hstu_config, is_inference=is_inference)
        self._processed_dir: str = processed_dir
        self._metadata_dir: str = metadata_dir
        self._history_length: int = history_length
        self._scan_window: int = scan_window
                                    
                                                                              
                                                                                 
                                                                             
                                                                       
                                                                                
                                                                              
                                                                                 
                                                                       
        if history_strategy not in ("interleaved", "last_n"):
            raise ValueError(
                f"history_strategy must be 'interleaved' or 'last_n', got "
                f"{history_strategy!r}"
            )
        self._history_strategy: str = history_strategy
                                                                               
                                                                                  
                                                                            
                                                                               
                                                                                 
                                                                           
        self._min_history: int = (
            history_length if min_history is None else int(min_history)
        )
                                                                             
                                                                             
                                                                              
                                      
        self._streaming_window_seconds: int = streaming_window_seconds
        self._streaming_sort_within_window: bool = streaming_sort_within_window
                                                                                
                                                                                   
                                                                                    
                                                                                
                                                                                    
                                                                             
                                                                                
                                                                                  
                                                                            
        self._streaming_shuffle_fraction: float = streaming_shuffle_fraction
        self._streaming_shuffle_seed: int = streaming_shuffle_seed
                                                                               
                                                                             
                                                                                
                                                                               
                                                                                
                                                                          
        self._train_split_percentage: float = train_split_percentage
        self._split_salt: int = split_salt
                                                                              
                                                                          
        self._eval_holdout_cache: Optional[np.ndarray] = None
        self._eval_holdout_cache_key: Optional[Tuple[int, int]] = None
        self._active: Optional[np.ndarray] = None
        self.is_eval: bool = False
        self._anchor_ts: Optional[np.ndarray] = None
        self._use_offline_ts_indices = use_offline_ts_indices
        self._ts_indices: Optional[np.ndarray] = None
        self._ts_offsets: Optional[np.ndarray] = None
        self._t_min: Optional[int] = None
        self._t_max: Optional[int] = None
        self._cache_dir: Optional[str] = cache_dir
        self._cross_specs: List[Tuple[str, Tuple[str, ...], int, int]] = [
            (name, tuple(keys), n, s) for (name, keys, n, s) in (cross_specs or [])
        ]
        assert hstu_config.action_weights is not None
        self._action_weights: List[int] = hstu_config.action_weights

        self._load_metadata(metadata_dir)
                                                                            
                                                                            
                                                                       
                                                                            
                                                                              
                                                   
        if cache_dir is None:
            cache_dir = os.path.join(processed_dir, f"hstu_cache_L{history_length}")
        self._cache_dir = cache_dir
        self._ensure_cache_built(cache_dir, processed_dir, history_length)
        self.store: _FlatEventStore = _FlatEventStore.load_mmap(cache_dir)
                                                                             
                                                                 
                                                                      
                                                                             
                                                              
        self._positions_name: str = self._positions_filename(
            history_length, self._min_history
        )
        self._ensure_positions_built(
            cache_dir, self._positions_name, self._min_history
        )
        if self._use_offline_ts_indices:
            from generative_recommenders.dlrm_v4.datasets.offline_ts_indices import (
                build_offline_ts_indices,
            )

            build_offline_ts_indices(
                cache_dir, history_length, self._min_history,
                window_seconds=self._streaming_window_seconds,
            )
        self._positions: np.ndarray = _load_npy_readonly(
            os.path.join(cache_dir, self._positions_name)
        )
        logger.info(
            f"Yambda dataset ready: {self.store.total_events:,} events, "
            f"{len(self._positions):,} training positions"
        )

    @staticmethod
    def _positions_filename(history_length: int, min_history: int) -> str:
        ''                                                                      
                                                                           
                                                                                 
        if min_history == history_length:
            return f"positions_L{history_length}.npy"
        return f"positions_L{history_length}_m{min_history}.npy"

    @staticmethod
    def _ensure_positions_built(
        cache_dir: str, positions_name: str, min_history: int
    ) -> None:
        ''                                                                

                                                                               
                                                                             
                                                                                
                                                                          
                                                                             
                                  
           
        import fcntl

        positions_path = os.path.join(cache_dir, positions_name)
        if os.path.exists(positions_path):
            return
        lock_path = os.path.join(cache_dir, "_positions_lock")
        with open(lock_path, "w") as lf:
            logger.info(f"Acquiring positions build lock for {positions_path}...")
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if os.path.exists(positions_path):
                    return
                flat_uid = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_uid.npy")
                )
                event_types = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_event_types.npy")
                )
                user_start = _load_npy_readonly(
                    os.path.join(cache_dir, "user_start.npy")
                )
                idx = np.arange(len(flat_uid), dtype=np.int64)
                keep = (idx - user_start[flat_uid] >= min_history) & (
                    event_types == LISTEN_TYPE
                )
                positions = np.where(keep)[0].astype(np.int64)
                tmp = positions_path + ".tmp.npy"
                np.save(tmp, positions)
                os.replace(tmp, positions_path)
                logger.info(
                    f"Wrote {positions_name}: {len(positions):,} anchors "
                    f"(min_history={min_history})"
                )
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    @staticmethod
    def _ensure_cache_built(
        cache_dir: str, processed_dir: str, history_length: int
    ) -> None:
        ''                                                          

                                                                              
                                                                           
                                                          
                                                                      
                                                                            
                                                                        
                                                                  

                                                                      
                                                      
           
        import fcntl
        import gc
        import json as _json

        ready = os.path.join(cache_dir, "_READY")
        if os.path.exists(ready):
            return
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, "_lock")
        with open(lock_path, "w") as lf:
            logger.info(f"Acquiring build lock for {cache_dir}...")
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if os.path.exists(ready):
                    return
                parquet_path = os.path.join(processed_dir, "train_sessions.parquet")
                logger.info(
                    f"Building flat-event cache from {parquet_path} "
                    f"(column-at-a-time, ~150 GB peak RAM)"
                )

                                                                              
                sessions = pl.read_parquet(parquet_path).sort(["uid", "session_id"])
                logger.info(f"Sessions sorted: {sessions.shape}")

                                                                          
                                                                           
                lengths = (
                    sessions.select(pl.col("item_ids").list.len())
                    .to_numpy()
                    .reshape(-1)
                    .astype(np.int64)
                )
                session_uids = sessions["uid"].to_numpy().astype(np.int64)
                N = int(lengths.sum())
                num_users = int(np.unique(session_uids).shape[0])
                logger.info(f"Total events: {N:,}, users: {num_users:,}")

                                                                 
                                                                  
                flat_uid = np.repeat(session_uids, lengths).astype(np.int64)
                np.save(os.path.join(cache_dir, "flat_uid.npy"), flat_uid)
                del flat_uid, session_uids, lengths
                gc.collect()
                logger.info("Wrote flat_uid.npy")

                                                                         
                                                                        
                                                  
                _list_cols = [
                    ("item_ids", "flat_item_ids", np.int64),
                    ("timestamps", "flat_timestamps", np.int64),
                    ("event_types", "flat_event_types", np.int64),
                    ("is_organic", "flat_is_organic", np.int8),
                    ("played_ratio_pct", "flat_played_ratio", np.float32),
                ]
                for src_col, dst_name, dtype in _list_cols:
                    exploded = sessions.select(pl.col(src_col).explode())
                    arr = exploded[src_col].to_numpy().astype(dtype, copy=False)
                    if dtype == np.float32:
                        np.nan_to_num(arr, copy=False, nan=0.0)
                    np.save(os.path.join(cache_dir, f"{dst_name}.npy"), arr)
                    del exploded, arr
                    gc.collect()
                    logger.info(f"Wrote {dst_name}.npy")

                                                                               
                del sessions
                gc.collect()

                                                                        
                event_types = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_event_types.npy")
                )
                played_ratio = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_played_ratio.npy")
                )
                is_listen = event_types == LISTEN_TYPE
                np.save(
                    os.path.join(cache_dir, "flat_is_listen_plus.npy"),
                    is_listen & (played_ratio >= LISTEN_PLUS_THRESHOLD),
                )
                np.save(
                    os.path.join(cache_dir, "flat_is_like.npy"),
                    event_types == LIKE_TYPE,
                )
                np.save(
                    os.path.join(cache_dir, "flat_is_skip.npy"),
                    is_listen & (played_ratio < LISTEN_PLUS_THRESHOLD),
                )
                del is_listen, played_ratio
                gc.collect()
                logger.info("Wrote flat_is_listen_plus/like/skip.npy")

                                                                         
                flat_uid = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_uid.npy")
                )
                uid_changes = np.where(np.diff(flat_uid) != 0)[0] + 1
                starts = np.concatenate([[0], uid_changes])
                ends = np.concatenate([uid_changes, [len(flat_uid)]])
                uid_vals = flat_uid[starts]
                max_uid = int(uid_vals.max()) + 1
                user_start = np.full(max_uid, -1, dtype=np.int64)
                user_end = np.full(max_uid, -1, dtype=np.int64)
                user_start[uid_vals] = starts
                user_end[uid_vals] = ends
                np.save(os.path.join(cache_dir, "user_start.npy"), user_start)
                np.save(os.path.join(cache_dir, "user_end.npy"), user_end)
                np.save(os.path.join(cache_dir, "unique_uids.npy"), uid_vals)
                logger.info("Wrote user_start/end/unique_uids.npy")

                                                                              
                                                                            
                                                                             
                user_start_per_event = user_start[flat_uid]
                idx = np.arange(len(flat_uid), dtype=np.int64)
                keep = (idx - user_start_per_event >= history_length) & (
                    event_types == LISTEN_TYPE
                )
                positions = np.where(keep)[0].astype(np.int64)
                np.save(
                    os.path.join(cache_dir, f"positions_L{history_length}.npy"),
                    positions,
                )
                logger.info(
                    f"Wrote positions_L{history_length}.npy: {len(positions):,}"
                )
                del (
                    flat_uid, event_types, user_start, user_end, uid_vals,
                    starts, ends, uid_changes, idx, user_start_per_event,
                    keep, positions,
                )
                gc.collect()

                                                                         
                with open(os.path.join(cache_dir, "store_meta.json"), "w") as f:
                    _json.dump(
                        {"num_users": num_users, "total_events": N}, f
                    )
                open(os.path.join(cache_dir, "_READY"), "w").close()
                logger.info(f"Cache build complete: {cache_dir}")
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load_metadata(self, metadata_dir: str) -> None:
        artist_map = pl.read_parquet(os.path.join(metadata_dir, "artist_item_mapping.parquet"))
        album_map = pl.read_parquet(os.path.join(metadata_dir, "album_item_mapping.parquet"))
        n_items = int(
            max(
                int(artist_map["item_id"].max()) + 1,
                int(album_map["item_id"].max()) + 1,
            )
        )
                                                                             
                                                                          
                                                                             
                                                                              
                                                            
        item_pop_path = os.path.join(self._processed_dir, "item_popularity.npy")
        if os.path.exists(item_pop_path):
            n_popularity = len(np.load(item_pop_path, mmap_mode="r"))
            if n_popularity != n_items:
                raise ValueError(
                    f"item_popularity.npy has {n_popularity} entries but the "
                    f"artist/album mappings describe {n_items} items. "
                    f"{item_pop_path} and {metadata_dir} are out of sync; "
                    "re-run preprocessing so both come from the same pass."
                )
        self.item_to_artist: np.ndarray = np.zeros(n_items, dtype=np.int64)
        valid = artist_map.filter(pl.col("item_id") < n_items)
        self.item_to_artist[valid["item_id"].to_numpy()] = valid["artist_id"].to_numpy()
        self.item_to_album: np.ndarray = np.zeros(n_items, dtype=np.int64)
        valid = album_map.filter(pl.col("item_id") < n_items)
        self.item_to_album[valid["item_id"].to_numpy()] = valid["album_id"].to_numpy()
        self.num_items: int = n_items

    def get_item_count(self) -> int:
                                                                             
                                                                           
        if self._active is not None:
            return int(len(self._active))
        return int(len(self._positions))

    def iloc(self, idx: int) -> int:
        if self._active is not None:
            return int(self._positions[self._active[idx]])
        return int(self._positions[idx])

    def _ensure_streaming_index(self) -> None:
        ''                                                                   
                                

                                                                           
                                                                            
                                                                             
                                                                        
           
        if self._use_offline_ts_indices:
            if self._ts_indices is None:
                directory = Path(self._cache_dir)
                tag = self._positions_name[len("positions_"):-4]
                tag += f"_W{self._streaming_window_seconds}"
                indices = _load_npy_readonly(directory / f"ts_indices_{tag}.npy")
                offsets = _load_npy_readonly(directory / f"ts_offsets_{tag}.npy")
                if (
                    indices.dtype != np.int64 or indices.ndim != 1
                    or len(indices) != len(self._positions)
                    or offsets.dtype != np.int64 or offsets.ndim != 1
                    or len(offsets) == 0 or offsets[0] != 0
                    or offsets[-1] != len(indices)
                    or np.any(offsets[1:] < offsets[:-1])
                ):
                    raise ValueError("Invalid offline TS indices or offsets")
                self._ts_offsets = offsets
                self._ts_indices = indices
            return
        if self._anchor_ts is not None:
            return
        import fcntl

        assert self._cache_dir is not None
                                                                              
                                                           
        anchor_path = os.path.join(
            self._cache_dir,
            self._positions_name.replace("positions_", "anchor_ts_", 1),
        )
        if not os.path.exists(anchor_path):
            lock_path = os.path.join(self._cache_dir, "_anchor_ts_lock")
            with open(lock_path, "w") as lf:
                logger.info(f"Acquiring anchor-ts build lock for {anchor_path}...")
                fcntl.flock(lf, fcntl.LOCK_EX)
                if not os.path.exists(anchor_path):
                    logger.info(
                        f"Building {anchor_path}: target ts for "
                        f"{len(self._positions):,} anchors"
                    )
                    anchor_ts = self.store.flat_timestamps[self._positions]
                    tmp = anchor_path + ".tmp.npy"
                    np.save(tmp, anchor_ts)
                    os.replace(tmp, anchor_path)
                    del anchor_ts
        self._anchor_ts = _load_npy_readonly(anchor_path)
        self._t_min = int(self._anchor_ts.min())
        self._t_max = int(self._anchor_ts.max())

    def _offline_window_bounds(self, start_ts: int, num_ts: int = 1) -> Tuple[int, int]:
        windows = len(self._ts_offsets) - 1
        start = min(windows, max(0, start_ts))
        end = min(windows, max(0, start_ts + max(0, num_ts)))
        return int(self._ts_offsets[start]), int(self._ts_offsets[end])

    def _offline_window_indices(self, start_ts: int, num_ts: int = 1) -> np.ndarray:
        lo, hi = self._offline_window_bounds(start_ts, num_ts)
        return self._ts_indices[lo:hi]

    def num_windows(self) -> int:
        ''                                                             
        self._ensure_streaming_index()
        if self._ts_offsets is not None:
            return len(self._ts_offsets) - 1
        assert self._t_min is not None and self._t_max is not None
        span = self._t_max - self._t_min + 1
        w = self._streaming_window_seconds
        return int((span + w - 1) // w)

    def window_indices(
        self, ts: int, sort_by_time: Optional[bool] = None
    ) -> np.ndarray:
        ''                                                                      
                                                               

                                                                              
                                                                               
                                                                                
                                                                  
                                         

                                                                              
                                                                            
                                                                            
                                                                             
                                                                          
           
        self._ensure_streaming_index()
        if self._ts_indices is not None:
            idx = self._offline_window_indices(ts)
            do_sort = self._streaming_sort_within_window if sort_by_time is None else sort_by_time
            if do_sort and idx.size:
                timestamps = self.store.flat_timestamps[self._positions[idx]]
                idx = idx[np.argsort(timestamps, kind="stable")]
            return idx
        assert self._anchor_ts is not None and self._t_min is not None
        w = self._streaming_window_seconds
        lo = self._t_min + ts * w
        hi = lo + w
        idx = np.where((self._anchor_ts >= lo) & (self._anchor_ts < hi))[0]
        do_sort = (
            self._streaming_sort_within_window if sort_by_time is None else sort_by_time
        )
        if do_sort and idx.size > 0:
            idx = idx[np.argsort(self._anchor_ts[idx], kind="stable")]
        logger.warning(f"window_indices({ts}): [{lo}, {hi}) -> {idx.size:,} anchors")
        return idx.astype(np.int64)

    def _eval_anchor_mask(self, anchor_idx: np.ndarray) -> np.ndarray:
        ''                                                                   

                                                                                
                                                                             
                                                                             
           
        uids = self.store.flat_uid[self._positions[anchor_idx]]
        return _uid_unit_hash(uids, self._split_salt) >= self._train_split_percentage

    def _shuffle_window(self, idx: np.ndarray, ts: int) -> np.ndarray:
        ''                                                            

                                                                              
                                                                               
                                                  

                                                                                 
                                                                            
                                                                                
                                                                               
                                                                                
                                                                                 

                                                                                  
                                                                                  
                                                                             
                                      

                                                                
                                                                             
                                                                                 
                                                                                 
                                         
           
        frac = self._streaming_shuffle_fraction
        if idx.size <= 1 or not frac or frac <= 0.0:
            return idx
        rng = np.random.default_rng(self._streaming_shuffle_seed + ts)
        if frac >= 1.0:
            return idx[rng.permutation(idx.size)]
                                                                               
                                                                                 
                                                                                  
                                                                    
        n = idx.size
        k = max(1, int(round(frac * n)))
        seg = np.arange(n, dtype=np.int64) // k
        keys = rng.random(n)
        order = np.lexsort((keys, seg))
        return idx[order]

    def train_window_indices(self, ts: int) -> np.ndarray:
        ''                                                                     
                                                                         
                                                                           
                                                                              
                                                       
        idx = self.window_indices(ts)
        if self._train_split_percentage >= 1.0:
            return self._shuffle_window(idx, ts)
        kept = idx[~self._eval_anchor_mask(idx)]
        logger.warning(
            f"train_window_indices({ts}): {idx.size:,} -> {kept.size:,} anchors "
            f"(holdout tsp={self._train_split_percentage}, salt={self._split_salt})"
        )
        return self._shuffle_window(kept, ts)

    def eval_holdout_indices(self, start_ts: int, num_windows: int = 1) -> np.ndarray:
        ''                                                     
                                                                                
                                                                                 
                                                                            
        key = (int(start_ts), int(num_windows))
        if self._eval_holdout_cache is not None and self._eval_holdout_cache_key == key:
            return self._eval_holdout_cache
        parts: List[np.ndarray] = []
        for ts in range(start_ts, start_ts + max(1, num_windows)):
            idx = self.window_indices(ts)
            if self._train_split_percentage < 1.0:
                idx = idx[self._eval_anchor_mask(idx)]
            parts.append(idx)
        holdout = (
            np.concatenate(parts).astype(np.int64)
            if parts
            else np.empty(0, dtype=np.int64)
        )
        logger.warning(
            f"eval_holdout_indices(start_ts={start_ts}, num_windows={num_windows}): "
            f"{holdout.size:,} held-out anchors (tsp={self._train_split_percentage})"
        )
        self._eval_holdout_cache = holdout
        self._eval_holdout_cache_key = key
        return holdout

    def total_train_anchors(self, start_ts: int, num_ts: int) -> int:
        ''                                                                      

                                                                               
                                                                                
                                                                                 
                                                                               
                                                                               
                                 

                                                                           
                                                                                
                                                                                
                                                                                
                                   
           
        self._ensure_streaming_index()
        if self._ts_indices is not None:
            lo, hi = self._offline_window_bounds(start_ts, num_ts)
            if self._train_split_percentage >= 1.0:
                return hi - lo
            indices = self._ts_indices[lo:hi]
            return sum(
                int(np.count_nonzero(~self._eval_anchor_mask(indices[start:start + 1_000_000])))
                for start in range(0, len(indices), 1_000_000)
            )
        assert self._anchor_ts is not None and self._t_min is not None
        if num_ts <= 0:
            return 0
        w = self._streaming_window_seconds
        lo = self._t_min + start_ts * w
        hi = self._t_min + (start_ts + num_ts) * w
        in_range = (self._anchor_ts >= lo) & (self._anchor_ts < hi)
        if self._train_split_percentage >= 1.0:
            total = int(np.count_nonzero(in_range))
        else:
            sel = np.where(in_range)[0]
            total = int(np.count_nonzero(~self._eval_anchor_mask(sel)))
        logger.warning(
            f"total_train_anchors(start_ts={start_ts}, num_ts={num_ts}): "
            f"{total:,} train anchors (tsp={self._train_split_percentage})"
        )
        return total

    def set_ts(self, ts: int, train_only: bool = False) -> None:
        ''                                                                    
                                                                               
                             

                                                                             
                                                                            
                                                                            
                                                                              
                                                                   
           
        self._active = (
            self.train_window_indices(ts) if train_only else self.window_indices(ts)
        )

    def set_active_indices(self, indices: np.ndarray) -> None:
        ''                                                                     
                                                                              
                                                                             
                          
        self._active = np.asarray(indices, dtype=np.int64)

    def load_query_samples(self, sample_list) -> None:
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        self.items_in_memory = {}
        for idx in sample_list:
            flat_pos = self.iloc(idx)
            self.items_in_memory[idx] = self._build_sample(flat_pos, max_num_candidates)
        self.last_loaded = time.time()

    def get_sample(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        if idx in self.items_in_memory:
            return self.items_in_memory[idx]
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        flat_pos = self.iloc(idx)
        return self._build_sample(flat_pos, max_num_candidates)

    @staticmethod
    def _empty_history() -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty, empty

    def _read_scan_window(
        self, flat_pos: int, user_start: int
    ) -> Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        ''                                                                  
                                                                              
                                  
        scan_start = max(int(user_start), int(flat_pos) - self._scan_window)
        scan_end = int(flat_pos)
        if scan_end <= scan_start:
            return None
        return (
            self.store.flat_item_ids[scan_start:scan_end],
            self.store.flat_timestamps[scan_start:scan_end],
            self.store.flat_is_listen_plus[scan_start:scan_end],
            self.store.flat_is_like[scan_start:scan_end],
            self.store.flat_is_skip[scan_start:scan_end],
        )

    def _materialize_history(
        self,
        keep_local: np.ndarray,
        item_ids: np.ndarray,
        timestamps: np.ndarray,
        is_lp: np.ndarray,
        is_like: np.ndarray,
        is_skip: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ''                                                                 
                                                   
        items = item_ids[keep_local]
        ts = timestamps[keep_local]
        artists = self.item_to_artist[np.clip(items, 0, self.item_to_artist.shape[0] - 1)]
        albums = self.item_to_album[np.clip(items, 0, self.item_to_album.shape[0] - 1)]
                                                                             
                                                                  
        weight = np.zeros(keep_local.shape[0], dtype=np.int64)
        weight[is_lp[keep_local]] |= LP_BIT
        weight[is_like[keep_local]] |= LIKE_BIT
        weight[is_skip[keep_local]] |= SKIP_BIT
        return items, artists, albums, ts, weight

    def _gather_history(
        self, flat_pos: int, user_start: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ''                                                         
        if self._history_strategy == "last_n":
            return self._gather_last_n_history(flat_pos, user_start)
        return self._gather_interleaved_history(flat_pos, user_start)

    def _gather_interleaved_history(
        self, flat_pos: int, user_start: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ''                                                                   
                                                                             
                                                                          
        L = self._history_length
        per_pool = max(1, L // 3)
        scan = self._read_scan_window(flat_pos, user_start)
        if scan is None:
            return self._empty_history()
        item_ids, timestamps, is_lp, is_like, is_skip = scan

                                                                            
                                                                
        idx_all = np.arange(item_ids.shape[0], dtype=np.int64)
        lp_idx = idx_all[is_lp][-per_pool:]
        like_idx = idx_all[is_like][-per_pool:]
        skip_idx = idx_all[is_skip][-per_pool:]

        keep_local = np.concatenate([lp_idx, like_idx, skip_idx])
        if keep_local.size == 0:
            return self._empty_history()

        order = np.argsort(keep_local, kind="stable")
        keep_local = keep_local[order]

        return self._materialize_history(
            keep_local, item_ids, timestamps, is_lp, is_like, is_skip
        )

    def _gather_last_n_history(
        self, flat_pos: int, user_start: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ''                                                                     
                                                                               
                                                                                
                                                                           
                                                                                 
        L = self._history_length
        scan = self._read_scan_window(flat_pos, user_start)
        if scan is None:
            return self._empty_history()
        item_ids, timestamps, is_lp, is_like, is_skip = scan

        member = is_lp | is_like | is_skip
                                                                               
                                                           
        keep_local = np.arange(item_ids.shape[0], dtype=np.int64)[member][-L:]
        if keep_local.size == 0:
            return self._empty_history()

        return self._materialize_history(
            keep_local, item_ids, timestamps, is_lp, is_like, is_skip
        )

    def _build_sample(
        self, flat_pos: int, max_num_candidates: int
    ) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        uid = int(self.store.flat_uid[flat_pos])
        user_start = int(self.store.user_start[uid])

        items, artists, albums, ts, weight = self._gather_history(
            flat_pos, user_start
        )

        target_item = int(self.store.flat_item_ids[flat_pos])
        target_artist = int(
            self.item_to_artist[target_item]
            if target_item < self.item_to_artist.shape[0]
            else 0
        )
        target_album = int(
            self.item_to_album[target_item]
            if target_item < self.item_to_album.shape[0]
            else 0
        )
        target_ts = int(self.store.flat_timestamps[flat_pos])

        played_ratio = float(self.store.flat_played_ratio[flat_pos])
        is_lp = (
            int(self.store.flat_event_types[flat_pos]) == LISTEN_TYPE
            and played_ratio >= LISTEN_PLUS_THRESHOLD
        )
                                                                             
                                                                         
        candidate_action_weight = LP_BIT if is_lp else 0

        cross_id_anchor: Dict[str, int] = {
            "uid": uid,
            "item_id": target_item,
            "artist_id": target_artist,
            "album_id": target_album,
            "hour_of_day": int((target_ts // 3600) % 24),
            "is_organic": int(self.store.flat_is_organic[flat_pos]),
        }
        cross_ids: Dict[str, int] = {
            name: xxhash_cross(cross_id_anchor, list(keys), n, salt)
            for (name, keys, n, salt) in self._cross_specs
        }

                                                          
        uih_seq_len_budget = (
            self._max_seq_len
            - max_num_candidates
            - len(self._contextual_feature_to_max_length or {})
        )
        if items.shape[0] > uih_seq_len_budget:
            items = items[-uih_seq_len_budget:]
            artists = artists[-uih_seq_len_budget:]
            albums = albums[-uih_seq_len_budget:]
            ts = ts[-uih_seq_len_budget:]
            weight = weight[-uih_seq_len_budget:]
        uih_seq_len = int(items.shape[0])
        dummy_watch_time = np.zeros(uih_seq_len, dtype=np.int64)

                                 
                                                                           
                                                                            
        contextual_features = self._contextual_feature_to_max_length or {}
        uih_contextual_values = np.empty(len(contextual_features), dtype=np.int64)
        for i, (name, length) in enumerate(contextual_features.items()):
            assert length == 1, f"yambda contextuals are length-1, got {name}={length}"
            if name == "uid":
                uih_contextual_values[i] = uid
            else:
                uih_contextual_values[i] = int(cross_ids[name])

                                                                        
                                               
                                                                                           
        n_sequential = len(self._uih_keys) - len(contextual_features)
        dummy_query_time = int(ts[-1]) if uih_seq_len > 0 else target_ts
        uih_kjt_values = np.concatenate(
            (
                uih_contextual_values,
                np.asarray(items, dtype=np.int64),
                np.asarray(artists, dtype=np.int64),
                np.asarray(albums, dtype=np.int64),
                np.asarray(weight, dtype=np.int64),
                np.asarray(ts, dtype=np.int64),
                dummy_watch_time,
                np.asarray([dummy_query_time], dtype=np.int64),
            )
        )
        uih_kjt_lengths = np.concatenate(
            (
                np.ones(len(contextual_features), dtype=np.int64),
                np.full(n_sequential, uih_seq_len, dtype=np.int64),
                np.ones(1, dtype=np.int64),
            )
        )

        uih_features_kjt = KeyedJaggedTensor(
            keys=self._uih_keys + ["dummy_query_time"],
            lengths=torch.from_numpy(uih_kjt_lengths),
            values=torch.from_numpy(uih_kjt_values),
        )

                                        
                                                                           
                                                                                 
                                                                     
        candidates_kjt_lengths = np.full(
            len(self._candidates_keys), max_num_candidates, dtype=np.int64
        )
        candidates_kjt_values = np.repeat(
            np.asarray(
                [
                    target_item,
                    target_artist,
                    target_album,
                    dummy_query_time,
                    candidate_action_weight,
                    1,                        
                ],
                dtype=np.int64,
            ),
            max_num_candidates,
        )
        candidates_features_kjt = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=torch.from_numpy(candidates_kjt_lengths),
            values=torch.from_numpy(candidates_kjt_values),
        )
        return uih_features_kjt, candidates_features_kjt
