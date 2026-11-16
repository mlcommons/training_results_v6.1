# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

             
'' 
                                                                     

                                                                                    
                                                                       
   

import gc
import logging
import os
import random
import shutil
import time
from datetime import datetime
from typing import Any, Dict, Optional, Set, Tuple

import gin
import numpy as np
import torch
from generative_recommenders.dlrm_v4.utils import (
    BinnedCumulativeAUC,
    LifetimeAUCMetricComputation,
    MetricsLogger,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.optimizer import Optimizer
from torchrec.distributed.types import ShardedTensor

logger: logging.Logger = logging.getLogger(__name__)

                                                                             
                                                                                
                                                                            
                    
WINDOW_COMPLETE: int = -1

                                                                              
                                                                        
                                                                            
                                                                   
METRICBUF_FILE_FMT: str = "metricbuf_rank{rank}.pt"


def _metric_blob_state_dict(m: torch.nn.Module) -> Dict[str, Any]:
    ''                                                                

                                                                           
                                                                                
                                                                        

                                                                               
                                                                                
                                                                                 
                                                                              

                                                                            
                                          
       
    sd = m.state_dict()
    if isinstance(m, LifetimeAUCMetricComputation):
        prefix = LifetimeAUCMetricComputation._LIFETIME_KEY_PREFIX
        sd = {k: v for k, v in sd.items() if not k.startswith(prefix)}
    elif isinstance(m, BinnedCumulativeAUC):
        sd = {
            k: (torch.zeros_like(v) if torch.is_tensor(v) else v) for k, v in sd.items()
        }
    return sd


def _collect_perrank_metric_state(
    metric_logger: "MetricsLogger",
) -> Dict[str, Dict[str, Any]]:
    ''                                                                     
                                                                     

                                                                                
                                                                               
                                                                               
                                      
                                                                               
                                                                             
                                                                    

                                                                              
                                        
       
    out: Dict[str, Dict[str, Any]] = {}
    for mode in ("train", "eval"):
        for idx, m in enumerate(metric_logger.class_metrics.get(mode, [])):
            if isinstance(m, (LifetimeAUCMetricComputation, BinnedCumulativeAUC)):
                out[f"class_metrics|{mode}|{idx}"] = m.state_dict()
    for coll in ("class_metrics", "regression_metrics"):
        for idx, m in enumerate(getattr(metric_logger, coll).get("eval_cum", [])):
            out[f"{coll}|eval_cum|{idx}"] = m.state_dict()
    return out


def _restore_perrank_metric_state(
    metric_logger: "MetricsLogger", state: Dict[str, Dict[str, Any]]
) -> None:
    for key, sd in state.items():
        coll, mode, idx_str = key.split("|")
        getattr(metric_logger, coll)[mode][int(idx_str)].load_state_dict(sd)


def _perrank_sample_counts(metric_logger: "MetricsLogger") -> Dict[str, int]:
    out: Dict[str, int] = {}

    def _count(m: torch.nn.Module) -> Optional[int]:
        if isinstance(m, LifetimeAUCMetricComputation):
            return m.lifetime_sample_count()
        if isinstance(m, BinnedCumulativeAUC):
            return m.cumulative_sample_count()
        return None

    for mode in ("train", "eval", "eval_cum"):
        for idx, m in enumerate(metric_logger.class_metrics.get(mode, [])):
            n = _count(m)
            if n is not None:
                out[f"class|{mode}|{idx}"] = n
    return out


class SparseState(Stateful):
    '' 
                                                               

                                                                               
                                                                                 

         
                                                           
                                                                                             
       

    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


def is_sparse_key(k: str, v: torch.Tensor) -> bool:
    return isinstance(v, ShardedTensor) or "embedding_collection" in k


def load_dense_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    own_state = model.state_dict()
    own_state_dense_keys = {k for k, v in own_state.items() if not is_sparse_key(k, v)}
    state_dict_dense_keys = {
        k for k, v in state_dict.items() if not is_sparse_key(k, v)
    }
    assert own_state_dense_keys == state_dict_dense_keys, (
        f"expects {own_state_dense_keys} but gets {state_dict_dense_keys}"
    )
    for name in state_dict_dense_keys:
        param = state_dict[name]
        if isinstance(param, torch.nn.Parameter):
                                                               
            param = param.data
        own_state[name].copy_(param)


def _rng_state(device: torch.device) -> Dict[str, Any]:
    ''                                                         

                                                                            
                                                                             
                                                                          
                                                                      
       
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def _restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def _list_numeric_subdirs(base_path: str) -> list[str]:
    ''                                                                            

                                                                               
                        
       
    if not os.path.isdir(base_path):
        return []
    out: list[str] = []
    for name in os.listdir(base_path):
        if name.isdigit():
            out.append(name)
    return sorted(out, key=int)


def _resolve_latest_subdir(path: str) -> str:
    ''                                                           

                                                                   
                                                                              
                                                                                                                           
                                                                               
                                                                          
                                                                              
                                                                           
                                 
       
    if not path:
        return path
    base = path.rstrip("/")
    leaf = os.path.basename(base)
    if leaf.isdigit():
        return base                                              
    subs = _list_numeric_subdirs(base)
    if not subs:
        logger.info("No checkpoint subdirs under %s — cold start (no load).", base)
        return ""                                                      
    resolved = os.path.join(base, subs[-1])
    logger.info("Auto-latest checkpoint: %s → %s", base, resolved)
    return resolved


def _prune_old_checkpoints(
    base_path: str, keep_last_n: int, just_saved_subdir: str
) -> None:
    ''                                                               

                                                                             
                                                                        
                                                                   
       
    if keep_last_n <= 0:
        return
    subs = _list_numeric_subdirs(base_path)
    if len(subs) <= keep_last_n:
        return
    to_prune = subs[:-keep_last_n]
    for name in to_prune:
        full = os.path.join(base_path, name)
        if os.path.realpath(full) == os.path.realpath(just_saved_subdir):
            continue
        try:
            shutil.rmtree(full)
            logger.info("Pruned old checkpoint: %s", full)
        except OSError as e:
            logger.warning("Failed to prune %s: %s", full, e)


def _cleanup_stale_tmps(base_path: str) -> None:
    ''                                                                     

                                                                             
                                                                              
                                                                              
       
    if not os.path.isdir(base_path):
        return
    for name in os.listdir(base_path):
        if name.endswith(".tmp") or name.endswith(".old"):
            full = os.path.join(base_path, name)
            try:
                shutil.rmtree(full)
                logger.warning("Removed stale checkpoint dir: %s", full)
            except OSError as e:
                logger.warning("Failed to remove stale dir %s: %s", full, e)


@gin.configurable
def save_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    rank: int,
    batch_idx: int,
    path: str = "",
    keep_last_n: int = 1,
    train_ts: Optional[int] = None,
    batch_idx_in_window: int = WINDOW_COMPLETE,
    device: Optional[torch.device] = None,
    split_contract: Optional[Dict[str, Any]] = None,
) -> None:
    '' 
                                                                              

                                                     
                                                                      
                                                                           

                                                                           
                                                                              
                                                                           

         
                                       
                                                             
                                                                           
                                                               
                                                                            
                                                    
                                                                                    
                                                                            
                                                                        
                                 
                                                                        
                                                                            
                                                                               
                                                                            
                                                                               
                                                      
                                                                       
                                                                           
                                  
       
    if path == "":
        return
                                                                             
                                                                            
                                                                                
                                                                           
                                                                             
                                         
    _t_ckpt_start = time.perf_counter()
    if metric_logger is not None:
        metric_logger.pause_perf("ckpt")
    base_path = path
                                                                      
    tmp_subdir = f"{base_path}/{batch_idx}.tmp"
    final_subdir = f"{base_path}/{batch_idx}"

    if rank == 0:
        _cleanup_stale_tmps(base_path)
                                                                                 
                                                                                 
                                                                                 
                                                                               
                                                                                 
                                                                                 
        shutil.rmtree(tmp_subdir, ignore_errors=True)
        os.makedirs(tmp_subdir, exist_ok=True)
        os.makedirs(f"{tmp_subdir}/sparse/", exist_ok=True)
    torch.distributed.barrier()
    sparse_path = f"{tmp_subdir}/sparse/"
    non_sparse_ckpt = f"{tmp_subdir}/non_sparse.ckpt"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if isinstance(v, ShardedTensor)
    }
    if rank == 0:
        dense_state_dict = {
            k: v
            for k, v in model.state_dict().items()
            if not isinstance(v, ShardedTensor)
        }
        class_metric_state_dict = {
            "train": [
                _metric_blob_state_dict(m) for m in metric_logger.class_metrics["train"]
            ],
            "eval": [
                _metric_blob_state_dict(m) for m in metric_logger.class_metrics["eval"]
            ],
        }
        regression_metric_state_dict = {
            "train": [
                m.state_dict() for m in metric_logger.regression_metrics["train"]
            ],
            "eval": [m.state_dict() for m in metric_logger.regression_metrics["eval"]],
        }
        torch.save(
            {
                "dense_dict": dense_state_dict,
                "optimizer_dict": optimizer.state_dict(),
                "class_metrics": class_metric_state_dict,
                "reg_metrics": regression_metric_state_dict,
                "global_step": metric_logger.global_step,
                                                                                
                                                                              
                                   
                "cumulative_train_samples": metric_logger.cumulative_train_samples,
                                                                              
                                                                                   
                "mlperf_run_started": metric_logger.mlperf_run_started,
                "sparse_tensor_keys": sparse_tensor_keys,
                                                                               
                                                                        
                "train_ts": train_ts,
                "batch_idx_in_window": batch_idx_in_window,
                                                                          
                                                                           
                                                                            
                                                                               
                                                                       
                                            
                "split_contract": split_contract,
            },
            non_sparse_ckpt,
        )

                                                                            
                                                                            
    if device is not None:
        rng_path = f"{tmp_subdir}/rng_rank{rank}.pt"
        torch.save(_rng_state(device), rng_path)

                                                                              
                                                                               
                                                                                
                                                          
    if metric_logger is not None:
        perrank_state = _collect_perrank_metric_state(metric_logger)
        if perrank_state:
            torch.save(
                perrank_state,
                f"{tmp_subdir}/{METRICBUF_FILE_FMT.format(rank=rank)}",
            )
            logger.info(
                "checkpoint save: cumulative metric state rank=%d samples=%s",
                rank,
                _perrank_sample_counts(metric_logger),
            )

    torch.distributed.barrier()
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    torch.distributed.checkpoint.save(
        sparse_dict,
        storage_writer=torch.distributed.checkpoint.FileSystemWriter(sparse_path),
    )
    torch.distributed.barrier()
                                                                               
                                             
    if rank == 0:
        if os.path.exists(final_subdir):
                                                                            
                                                                                
                                                                                 
                                                                                
                                                                                     
            old_aside = f"{final_subdir}.old"
            shutil.rmtree(old_aside, ignore_errors=True)
            os.replace(final_subdir, old_aside)
            os.replace(tmp_subdir, final_subdir)
            shutil.rmtree(old_aside, ignore_errors=True)
        else:
            os.replace(tmp_subdir, final_subdir)
        _prune_old_checkpoints(base_path, keep_last_n, final_subdir)
        logger.info(
            "checkpoint successfully saved → %s (wall-time %.2fs)",
            final_subdir,
            time.perf_counter() - _t_ckpt_start,
        )
    torch.distributed.barrier()
    if metric_logger is not None:
        metric_logger.resume_perf("ckpt")


@gin.configurable
def load_sparse_checkpoint(
    model: torch.nn.Module,
    path: str = "",
) -> None:
    if path == "":
        return
    sparse_path = f"{path}/sparse/"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if is_sparse_key(k, v)
    }
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    gc.collect()
    torch.distributed.checkpoint.load(
        sparse_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(sparse_path),
    )
    gc.collect()
    print("sparse checkpoint successfully loaded")


@gin.configurable
def load_nonsparse_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    metric_logger: Optional[MetricsLogger] = None,
    path: str = "",
    rank: int = 0,
) -> Tuple[Optional[int], int, Optional[Dict[str, Any]]]:
    '' 
                                                         

                                                                             
                                                                               
                              

            
                                                                              
                                                                            
                                                                                
                                       
       
    if path == "":
        return None, WINDOW_COMPLETE, None
    non_sparse_ckpt = f"{path}/non_sparse.ckpt"

                                                                              
                                                                               
                                                                        
    non_sparse_state_dict = torch.load(
        non_sparse_ckpt, map_location=device, weights_only=False
    )
    load_dense_state_dict(model, non_sparse_state_dict["dense_dict"])
    print("dense checkpoint successfully loaded")
    if optimizer is not None:
        optimizer.load_state_dict(non_sparse_state_dict["optimizer_dict"])
        print("optimizer checkpoint successfully loaded")
    if metric_logger is not None:
        metric_logger.global_step = non_sparse_state_dict["global_step"]
                                                                              
        metric_logger.cumulative_train_samples = non_sparse_state_dict.get(
            "cumulative_train_samples", 0
        )
                                                                            
                                                                                 
                          
        metric_logger.mlperf_run_started = non_sparse_state_dict.get(
            "mlperf_run_started", False
        )
        class_metric_state_dict = non_sparse_state_dict["class_metrics"]
        regression_metric_state_dict = non_sparse_state_dict["reg_metrics"]

                                                                            
                                                                            
                                                                    
        def _restore_metric_list(live: list, saved: Optional[list], label: str) -> None:
            saved = saved or []
            if len(live) != len(saved):
                logger.warning(
                    "metric count mismatch for %s: live=%d saved=%d; "
                    "restoring overlapping %d",
                    label,
                    len(live),
                    len(saved),
                    min(len(live), len(saved)),
                )
            for i in range(min(len(live), len(saved))):
                live[i].load_state_dict(saved[i])

        _restore_metric_list(
            metric_logger.class_metrics["train"],
            class_metric_state_dict.get("train"),
            "class/train",
        )
        _restore_metric_list(
            metric_logger.class_metrics["eval"],
            class_metric_state_dict.get("eval"),
            "class/eval",
        )
        _restore_metric_list(
            metric_logger.regression_metrics["train"],
            regression_metric_state_dict.get("train"),
            "reg/train",
        )
        _restore_metric_list(
            metric_logger.regression_metrics["eval"],
            regression_metric_state_dict.get("eval"),
            "reg/eval",
        )

                                                                               
                                                                                
                                                                          
                                                                               
                                                                             
                                                                 
        mb_path = f"{path}/{METRICBUF_FILE_FMT.format(rank=rank)}"
        if os.path.exists(mb_path):
            perrank_state = torch.load(mb_path, map_location=device, weights_only=False)
            _restore_perrank_metric_state(metric_logger, perrank_state)
            logger.info(
                "checkpoint load: cumulative metric state rank=%d samples=%s",
                rank,
                _perrank_sample_counts(metric_logger),
            )
        else:
            logger.info(
                "checkpoint load: no per-rank cumulative metric state at %s "
                "(legacy/pre-fix checkpoint); cumulative metrics will refill",
                mb_path,
            )

                                                                                
                                                                         
    rng_path = f"{path}/rng_rank{rank}.pt"
    if os.path.exists(rng_path):
                                                                            
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        _restore_rng_state(rng_state, device)
        logger.info("RNG state restored from %s", rng_path)

    train_ts = non_sparse_state_dict.get("train_ts")
    batch_idx_in_window = non_sparse_state_dict.get(
        "batch_idx_in_window", WINDOW_COMPLETE
    )
    split_contract = non_sparse_state_dict.get("split_contract")
    return train_ts, batch_idx_in_window, split_contract


@gin.configurable
def load_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    path: str = "",
    rank: int = 0,
) -> Tuple[Optional[int], int, Optional[Dict[str, Any]], bool]:
    '' 
                                                                                    

                                                                           
                                                                                
                                                                                
            

            
                                                                               
                                                                                 
                                                                            
                                                                                
                                                                               
                                                                  
       
    resolved = _resolve_latest_subdir(path)
    cold_start = resolved == ""
    load_sparse_checkpoint(model=model, path=resolved)
    train_ts, batch_idx_in_window, split_contract = load_nonsparse_checkpoint(
        model=model,
        optimizer=optimizer,
        metric_logger=metric_logger,
        path=resolved,
        device=device,
        rank=rank,
    )
    return train_ts, batch_idx_in_window, split_contract, cold_start
