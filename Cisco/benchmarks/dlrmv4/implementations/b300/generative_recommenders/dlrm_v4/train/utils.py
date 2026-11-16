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

             
import logging
import math
import os
import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import gin
import torch
import torchrec
from generative_recommenders.dlrm_v4.checkpoint import save_dmp_checkpoint, WINDOW_COMPLETE
from generative_recommenders.dlrm_v4.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v4.datasets.dataset import (
    collate_fn,
    Dataset,
)
from generative_recommenders.compacted_forward.config import (
    configure_embedding_tables,
)
from generative_recommenders.dlrm_v4.utils import get_dataset, MetricsLogger, Profiler
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import _T_co, DistributedSampler
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.distributed.types import ShardedTensor, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}

                                                                    
                                                                     
                                                                                
                                                                           
_PLACEMENT_TO_KERNEL: Dict[str, EmbeddingComputeKernel] = {
    "hbm": EmbeddingComputeKernel.FUSED,
    "uvm": EmbeddingComputeKernel.FUSED_UVM,
    "uvm_caching": EmbeddingComputeKernel.FUSED_UVM_CACHING,
}

                                                                      
                                                                                 
                                                                        
                                                                                
                                                                                  
                                                                           
_SHARDING_TO_TYPE: Dict[str, ShardingType] = {
    "row_wise": ShardingType.ROW_WISE,
    "column_wise": ShardingType.COLUMN_WISE,
    "table_wise": ShardingType.TABLE_WISE,
    "table_row_wise": ShardingType.TABLE_ROW_WISE,
    "table_column_wise": ShardingType.TABLE_COLUMN_WISE,
    "data_parallel": ShardingType.DATA_PARALLEL,
    "rw": ShardingType.ROW_WISE,
    "cw": ShardingType.COLUMN_WISE,
    "tw": ShardingType.TABLE_WISE,
    "twrw": ShardingType.TABLE_ROW_WISE,
}


@gin.configurable
def seed_everything(seed: int = -1, rank: int = 0) -> None:
    ''                                                                             

                                                                               
                                                                                  
                                                                               
                                                                                       
       
    import random

    import numpy as np

    pinned = seed >= 0
    if not pinned:
                                                                             
                                                                          
                                                                  
        if dist.is_available() and dist.is_initialized():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            drawn = int.from_bytes(os.urandom(4), "little") if rank == 0 else 0
            _seed_t = torch.tensor([drawn], dtype=torch.int64, device=device)
            dist.broadcast(_seed_t, src=0)
            seed = int(_seed_t.item())
        else:
            seed = int.from_bytes(os.urandom(4), "little")
                                                                          
    os.environ["SEED"] = str(seed)

    logger.info(
        f"[rank {rank}] seeding all RNGs with SEED={seed} "
        f"({'pinned via $SEED' if pinned else 'random per-run; set $SEED to reproduce'})"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@gin.configurable
def decorrelate_runtime_rng(rank: int = 0, enabled: bool = True) -> None:
    ''                                                                           
                                                                                  

                                                                          
                                                                        
                                                                                 
                                             
       
    if not enabled:
        logger.info(
            f"[rank {rank}] decorrelate_runtime_rng disabled; dropout masks "
            f"identical across ranks"
        )
        return
    base = int(os.environ.get("SEED", "1"))
    offset_seed = base + int(rank)
    torch.manual_seed(offset_seed)
    torch.cuda.manual_seed_all(offset_seed)
    logger.info(
        f"[rank {rank}] decorrelated runtime RNG: SEED={base} + rank={rank} "
        f"=> {offset_seed} (per-rank dropout masks)"
    )


def setup(
    rank: int,
    world_size: int,
    master_port: int,
    device: torch.device,
    master_addr: str = "localhost",
) -> dist.ProcessGroup:
                                                                          
                                                                                
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    BACKEND = dist.Backend.NCCL
                                                                           
                                                                               
                                                                               
                                                                     
    TIMEOUT = int(os.environ.get("PG_TIMEOUT_S", "1800"))

                                                                         
                                                                        
                                                              
    torch.cuda.set_device(device)

                                                                                
                                                                              
                                                                               
                                                                              
                                                

                                  
     
                                                                            
                                                                               
                                                                              
                                                                              
                                                                             
                                                                           
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=timedelta(seconds=TIMEOUT),
        )

    pg = dist.new_group(
        backend=BACKEND,
        timeout=timedelta(seconds=TIMEOUT),
    )

    return pg


def cleanup() -> None:
    dist.destroy_process_group()


def _window_boundary_barrier(
    device: torch.device, world_size: int, train_ts: int
) -> None:
    ''                                                      

                                                                              
                                                                           
                                                                               
                                                                          
                                                                          
                                                                             
                                                                            

                                                                           
                                                                           
                                                                           
                                                        
       
    if not (dist.is_available() and dist.is_initialized()) or world_size <= 1:
        return
    t0 = time.time()
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
    waited = time.time() - t0
                                                                             
                                                                            
    if waited > 5.0:
        logger.warning(
            "[window-barrier] train_ts=%d: waited %.1fs at boundary "
            "rendezvous (per-rank data-prep skew)",
            train_ts,
            waited,
        )
                                                                            
                                                                              
                                                                        
                                                                              
                                                                                
                                                                                  
    if os.environ.get("WINDOW_BARRIER_DEBUG") == "1" and dist.get_rank() == 0:
        logger.info(
            "[window-barrier] train_ts=%d rendezvous complete (waited %.3fs)",
            train_ts,
            waited,
        )


class HammerToTorchDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        self.dataset: Dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        self.dataset.load_query_samples([idx])
        sample = self.dataset.get_sample(idx)
        self.dataset.unload_query_samples([idx])
        return sample

    def __getitems__(
        self, indices: List[int]
    ) -> List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]]:
        self.dataset.load_query_samples(indices)
        samples = [self.dataset.get_sample(i) for i in indices]
        self.dataset.unload_query_samples(indices)
        return samples


class _ChainedRanges:
    ''                                                                 

                                                                          
                                                                        
                                                              
       

    def __init__(self, ranges: List[range]) -> None:
        self._ranges: List[range] = list(ranges)
        offsets = [0]
        for r in self._ranges:
            offsets.append(offsets[-1] + len(r))
        self._offsets: List[int] = offsets

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> int:
        import bisect
        if idx < 0:
            idx += self._offsets[-1]
        if idx < 0 or idx >= self._offsets[-1]:
            raise IndexError(idx)
        bucket = bisect.bisect_right(self._offsets, idx) - 1
        return self._ranges[bucket][idx - self._offsets[bucket]]


class ChunkDistributedSampler(DistributedSampler[_T_co]):
    '' 
                                                                
       

    def __init__(
        self,
        dataset: TorchDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 1,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 1001 + int(self.rank))
            indices_t = torch.randperm(self.num_samples, generator=g)
        else:
            indices_t = torch.arange(self.num_samples)
        assert self.drop_last is True, (
            "drop_last must be True for ChunkDistributedSampler"
        )
        indices_t = indices_t + (self.num_samples * int(self.rank))
        assert indices_t.numel() == self.num_samples
                                                                                
                                                                               
                                                                               
                                                                           
                                                                              
                                 
        indices_np = indices_t.numpy()
        return (int(x) for x in indices_np)

    def set_epoch(self, epoch: int) -> None:
        logger.warning(f"Setting epoch to {epoch}")
        self.epoch = epoch


@gin.configurable
def make_model(
    dataset: str,
    bf16_training: bool = False,
    hammer_kernel: Optional[str] = None,
    use_gather_select: bool = False,
    use_compacted_preprocessor: bool = True,
    max_embedding_table_rows: Optional[int] = None,
    enable_tflops_logging: bool = False,
) -> Tuple[torch.nn.Module, DlrmHSTUConfig, Dict[str, EmbeddingConfig]]:
    hstu_config = get_hstu_configs(dataset)
    table_config = get_embedding_table_config(dataset)

    if use_compacted_preprocessor:
        table_config = configure_embedding_tables(table_config, hstu_config)
    if max_embedding_table_rows is not None:
        if max_embedding_table_rows <= 0:
            raise ValueError("max_embedding_table_rows must be positive")
        for table in table_config.values():
            table.num_embeddings = max_embedding_table_rows

                                                                      
                                                                    
                                                                     
                                                   
    dlrm = DlrmHSTU(
        hstu_configs=hstu_config,
        embedding_tables=table_config,
        is_inference=False,
        bf16_training=bf16_training,
        use_gather_select=use_gather_select,
        enable_tflops_logging=enable_tflops_logging,
    )

                                                        
                                                                                      
                                                                                 
                                                                            
                                                                                  
                                                                               
                                                                               
                                                                            
    kernel_choice = (
        os.environ.get("HSTU_HAMMER_KERNEL", "").upper()
        or (hammer_kernel.upper() if hammer_kernel else "")
    )
    if kernel_choice:
        dlrm.set_hammer_kernel(HammerKernel[kernel_choice])
        logger.warning(f"HSTU hammer kernel set to: {kernel_choice}")

    model: torch.nn.Module = dlrm
    if use_compacted_preprocessor:
        from generative_recommenders.compacted_forward.dlrm_hstu import (
            CompactedForwardAdapterModule,
        )

        model = CompactedForwardAdapterModule(dlrm)

    return (
        model,
        hstu_config,
        table_config,
    )


def _collate_batch_fn(
    dataset: Dataset, use_compacted_preprocessor: bool
) -> Callable[..., Any]:
    if not use_compacted_preprocessor:
        return collate_fn

    from generative_recommenders.dlrm_v4.datasets.compacted import collate_compacted

    return partial(
        collate_compacted,
        hstu_config=dataset.hstu_config,
        is_inference=dataset._is_inference,
    )


def _forward_sample(model: torch.nn.Module, sample: Any) -> Tuple[Any, ...]:
    ''                                                               

    underlying = model.module if hasattr(model, "module") else model
    if hasattr(sample, "compacted_keys") or getattr(
        underlying, "_pipeline_mode", False
    ):
        return model.forward(sample)
    return model.forward(
        sample.uih_features_kjt,
        sample.candidates_features_kjt,
    )


def _batch_num_candidates(sample: Any) -> torch.Tensor:
    if hasattr(sample, "num_candidates"):
        return sample.num_candidates
    candidates = sample.candidates_features_kjt
    return candidates.lengths().view(len(candidates.keys()), -1)[0]


def warmup_model(
    rank: int,
    model: torch.nn.Module,
    config: DlrmHSTUConfig,
    batch_size: int,
    device: torch.device,
    num_batches: int,
    seed: int,
    embedding_value_bound: int,
) -> None:
    ''                                                               

    if num_batches < 0:
        raise ValueError(
            f"warmup num_batches must be non-negative, got {num_batches}"
        )
    if num_batches == 0:
        return

    root = model.module if hasattr(model, "module") else model
    if hasattr(root, "_content_feature_pairs"):
        from generative_recommenders.dlrm_v4.datasets.compacted import (
            collate_compacted,
        )

        warmup_collate_fn = partial(collate_compacted, hstu_config=config)
    else:
        warmup_collate_fn = collate_fn
    from generative_recommenders.dlrm_v4.datasets.dataset import (
        get_random_data,
    )

    contextual_features = list(config.contextual_feature_to_max_length)
    max_uih_len = (
        config.max_seq_len
        - sum(config.contextual_feature_to_max_length.values())
        - config.max_num_candidates
    )
    feature_value_bounds = (
        {
            config.uih_weight_feature_name: 1 << len(config.action_weights),
        }
        if config.action_weights is not None
        else None
    )

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    was_training = model.training

    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    if rank == 0:
        logger.info(
            "[run-start-warmup] starting %d synthetic forward/backward batches "
            "on a temporary model (batch_size=%d)",
            num_batches,
            batch_size,
        )

    try:
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
        model.train()

        for _ in range(num_batches):
            model.zero_grad(set_to_none=True)
            random_samples = [
                get_random_data(
                    contexual_features=contextual_features,
                    hstu_uih_keys=config.hstu_uih_feature_names,
                    hstu_candidates_keys=config.hstu_candidate_feature_names,
                    uih_max_seq_len=max_uih_len,
                    max_num_candidates=config.max_num_candidates,
                    value_bound=embedding_value_bound,
                    feature_value_bounds=feature_value_bounds,
                )
                for _ in range(batch_size)
            ]
            sample = warmup_collate_fn(random_samples)
            sample.to(device)
            outputs = _forward_sample(model, sample)
            aux_losses = outputs[2]
            if not aux_losses:
                raise RuntimeError("warmup forward returned no training losses")
            total_loss = sum(aux_losses.values())
            total_loss.backward()
            del sample, outputs, aux_losses, total_loss

        torch.cuda.synchronize(device)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            logger.info(
                "[run-start-warmup] complete total_wall_s=%.3f",
                time.perf_counter() - start,
            )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)


@gin.configurable()
def dense_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "AdamW":
        optimizer_cls = torch.optim.AdamW
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    logger.info(
        f"[dense optimizer] {optimizer_name} learning_rate={learning_rate} "
        f"(resolved from gin; override via $DENSE_LR)"
    )
    return optimizer_cls, kwargs, optimizer_factory


@gin.configurable()
def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "SGD":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "RowWiseAdagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    logger.info(
        f"[sparse optimizer] {optimizer_name} learning_rate={learning_rate} "
        f"(resolved from gin; override via $SPARSE_LR)"
    )
    return optimizer_cls, kwargs, optimizer_factory


_FBGEMM_LOWMEM_PATCHED = False


def _patch_fbgemm_lowmem_clamp_cast(enabled: bool = True, rank0: bool = False) -> None:
    ''                                                                   

                                                      
                                                          
                                                                                
                                                                            
                                                                

                                                                    
                                                                              
                                                                                
                                                                               
                                                                               
                                                                              
                                                                               
                                                                           
                                                                    

                                                                            
                                                                                
                                                                            
                                                                               
                                                                               
                                                                             
                                                                          
       
    global _FBGEMM_LOWMEM_PATCHED
    if not enabled:
        if rank0:
            logger.warning(
                "[qcomm-lowmem] DISABLED (qcomm_lowmem_clamp_cast=False / "
                "QCOMM_LOWMEM_CODEC=0) — running stock fbgemm clamp+cast, which "
                "allocates a full-size fp32 clamp temp and can OOM->hang the "
                "hottest row-wise embedding shard on skewed batches."
            )
        return
    if _FBGEMM_LOWMEM_PATCHED:
        return
    try:
        from fbgemm_gpu import quantize_comm, quantize_utils

        _HMIN = quantize_utils.TORCH_HALF_MIN
        _HMAX = quantize_utils.TORCH_HALF_MAX
        _BMIN = quantize_utils.TORCH_BFLOAT16_MIN
        _BMAX = quantize_utils.TORCH_BFLOAT16_MAX

        def _lowmem_fp16(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.half().clamp_(_HMIN, _HMAX)

        def _lowmem_bf16(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.bfloat16().clamp_(_BMIN, _BMAX)

                                                                                
                                                                                  
                                                               
        for _mod in (quantize_utils, quantize_comm):
            if hasattr(_mod, "fp32_to_fp16_with_clamp"):
                _mod.fp32_to_fp16_with_clamp = _lowmem_fp16
            if hasattr(_mod, "fp32_to_bf16_with_clamp"):
                _mod.fp32_to_bf16_with_clamp = _lowmem_bf16

        _FBGEMM_LOWMEM_PATCHED = True
        if rank0:
            logger.info(
                "[qcomm-lowmem] patched fbgemm fp32->fp16/bf16 clamp+cast to "
                "cast-then-clamp_ (drops the full-size fp32 clamp temp; avoids "
                "OOM on skewed row-wise embedding a2a)"
            )
    except Exception as e:  # noqa: BLE001 — patch is best-effort, never fatal
        if rank0:
            logger.warning(
                "[qcomm-lowmem] could not patch fbgemm clamp+cast (%s: %s); "
                "running with stock (higher-peak) quantizer",
                type(e).__name__,
                e,
            )


def _maybe_apply_qcomm_a2a(
    sharders: List[Any],
    device: torch.device,
    forward_precision: str = "fp32",
    backward_precision: str = "fp32",
    lowmem_clamp_cast: bool = True,
) -> List[Any]:
    ''                                                                         

                                                                                 
                                                                             
                                                                                 
                                                                                  
                                                                

                                                                                 
                                                                      
                                                                       
                                                                               
                                                                               
                                                                                
                               

                                                                                
                                                                               
                                                                               
       
    _COMM = {"bf16": "BF16", "fp16": "FP16", "fp32": "FP32"}
    fwd = (forward_precision or "fp32").strip().lower()
    bwd = (backward_precision or "fp32").strip().lower()
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    for name, p in (("forward", fwd), ("backward", bwd)):
        if p not in _COMM:
                                                                               
                                                                               
            raise ValueError(
                f"DLRMV4 qcomm a2a: unknown {name} precision {p!r} "
                f"(want one of fp32|bf16|fp16)"
            )
    if fwd == "fp32" and bwd == "fp32":
        return sharders
                                                                             
                                                                             
                                                                              
                                                            
    _patch_fbgemm_lowmem_clamp_cast(enabled=lowmem_clamp_cast, rank0=rank0)
    try:
        from torchrec.distributed.embedding import EmbeddingCollectionSharder
        from torchrec.distributed.fbgemm_qcomm_codec import (
            CommType,
            get_qcomm_codecs_registry,
            QCommsConfig,
        )

        qcfg = QCommsConfig(
            forward_precision=getattr(CommType, _COMM[fwd]),
            backward_precision=getattr(CommType, _COMM[bwd]),
        )
        registry = get_qcomm_codecs_registry(qcfg, device=device)
    except Exception as e:  # noqa: BLE001
                                                                                  
                                                                                  
                                                                              
                                                                                
        raise RuntimeError(
            f"DLRMV4 qcomm a2a: failed to enable configured quantization "
            f"(forward={fwd} backward={bwd}): {type(e).__name__}: {e}"
        ) from e

    new_sharders = []
    replaced = False
    for s in sharders:
        if type(s).__name__ == "EmbeddingCollectionSharder" and not replaced:
            new_sharders.append(
                EmbeddingCollectionSharder(qcomm_codecs_registry=registry)
            )
            replaced = True
        else:
            new_sharders.append(s)
    if not replaced:
                                                                                   
                                                                                   
                                                                                    
        raise RuntimeError(
            f"DLRMV4 qcomm a2a: quantization configured (forward={fwd} "
            f"backward={bwd}) but no EmbeddingCollectionSharder was found to attach "
            f"the qcomm codec registry to; refusing to run with quantization "
            f"silently disabled"
        )
    if rank0:
        logger.info(
            "DLRMV4 qcomm a2a ENABLED: forward=%s backward=%s "
            "replaced_ec_sharder=%s",
            fwd,
            bwd,
            replaced,
        )
    return new_sharders


def _maybe_apply_ec_index_dedup(
    sharders: List[Any],
    enabled: bool,
) -> List[Any]:
    ''                                                                     

                                                                                 
                                                                    
                                                                                 
                                                                          
                                                                              
                                              
                                                                          
                                                                              
                                                                                  
                                                                     

                                                                                
                                                                           
                                                                        
                                                                               
                                                                               
                                                                             
                                                                               
                                                                             
                                                                        
                                                                             
                                                                         
                                              

                                                                                 
                                                                               
                                                          
                                                                   
       
    if not enabled:
        return sharders

    from torchrec.distributed.embedding import EmbeddingCollectionSharder

    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    new_sharders = []
    replaced = False
    for s in sharders:
        if type(s).__name__ == "EmbeddingCollectionSharder" and not replaced:
            new_sharders.append(
                EmbeddingCollectionSharder(
                    fused_params=s.fused_params,
                    qcomm_codecs_registry=s.qcomm_codecs_registry,
                    use_index_dedup=True,
                )
            )
            replaced = True
        else:
            new_sharders.append(s)
    if not replaced:
                                                                           
                                                                              
                                                                       
        raise RuntimeError(
            "DLRMV4 EC index dedup configured (use_index_dedup=True) but no "
            "EmbeddingCollectionSharder was found to enable it on; refusing to "
            "run with dedup silently disabled"
        )
    if rank0:
        logger.info("DLRMV4 EC index dedup ENABLED (use_index_dedup=True)")
    return new_sharders


def _embedding_table_names(
    model: torch.nn.Module,
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]],
) -> List[str]:
    ''                                                  

                                                                               
                                                                              
                                       
       
    if embedding_table_configs:
        return list(embedding_table_configs.keys())
    names: List[str] = []
    for _, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            if isinstance(module, EmbeddingBagCollection):
                names.extend(c.name for c in module.embedding_bag_configs())
            elif isinstance(module, EmbeddingCollection):
                names.extend(c.name for c in module.embedding_configs())
    return names


def _build_placement_constraints(
    model: torch.nn.Module,
    embedding_placement: str,
    embedding_placement_overrides: Dict[str, str],
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]],
    embedding_sharding_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, ParameterConstraints]:
    ''                                                                          

                                                                            

                                               
                                                                       
                                                                                
                                                                               
                                                                                 
                                                                              
                                                                               
                                                         

                                                                                 
                                                                               
                                                      
       
    embedding_sharding_overrides = embedding_sharding_overrides or {}
    valid_place = set(_PLACEMENT_TO_KERNEL) | {"auto", ""}
    for where, val in [
        ("embedding_placement", embedding_placement),
        *[
            (f"embedding_placement_overrides[{k}]", v)
            for k, v in embedding_placement_overrides.items()
        ],
    ]:
        if val not in valid_place:
            raise ValueError(
                f"Invalid embedding placement {val!r} for {where}; "
                f"expected one of {sorted(valid_place - {''})}."
            )
    valid_shard = set(_SHARDING_TO_TYPE) | {"auto", ""}
    for k, v in embedding_sharding_overrides.items():
        if v not in valid_shard:
            raise ValueError(
                f"Invalid embedding sharding {v!r} for "
                f"embedding_sharding_overrides[{k}]; "
                f"expected one of {sorted(valid_shard - {''})}."
            )

    names = _embedding_table_names(model, embedding_table_configs)
    unknown = (
        set(embedding_placement_overrides) | set(embedding_sharding_overrides)
    ) - set(names)
    if unknown:
        logger.warning(
            "[emb-placement] override(s) for unknown table(s) %s ignored; "
            "known tables: %s",
            sorted(unknown),
            sorted(names),
        )

    constraints: Dict[str, ParameterConstraints] = {}
    resolved_place: Dict[str, str] = {}
    resolved_shard: Dict[str, str] = {}
    for name in names:
        placement = embedding_placement_overrides.get(name, embedding_placement)
        sharding = embedding_sharding_overrides.get(name, "auto")
        resolved_place[name] = placement or "auto"
        resolved_shard[name] = sharding or "auto"
        kernel = _PLACEMENT_TO_KERNEL.get(placement)
        stype = _SHARDING_TO_TYPE.get(sharding)
        kwargs: Dict[str, Any] = {}
        if kernel is not None:
            kwargs["compute_kernels"] = [kernel.value]
        if stype is not None:
            kwargs["sharding_types"] = [stype.value]
        if kwargs:
            constraints[name] = ParameterConstraints(**kwargs)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.info(
            "[emb-placement] placement(global=%r overrides=%s) sharding(overrides=%s) "
            "-> resolved_placement=%s resolved_sharding=%s "
            "(constrained=%d/%d tables; the rest are planner-auto)",
            embedding_placement,
            embedding_placement_overrides or {},
            embedding_sharding_overrides or {},
            resolved_place,
            resolved_shard,
            len(constraints),
            len(names),
        )
    return constraints


@gin.configurable
def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
    world_size: int,
    local_world_size: Optional[int] = None,
    hbm_cap_gb: int = 260,
    sparse_a2a_forward_precision: str = "fp32",
    sparse_a2a_backward_precision: str = "fp32",
    qcomm_lowmem_clamp_cast: bool = True,
    ec_index_dedup: bool = False,
    use_gather_select: bool = False,
    embedding_placement: str = "auto",
    embedding_placement_overrides: Optional[Dict[str, str]] = None,
    embedding_sharding_overrides: Optional[Dict[str, str]] = None,
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]] = None,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    dense_opt_cls, dense_opt_args, dense_opt_factory = (
        dense_optimizer_factory_and_class()
    )

    sparse_opt_cls, sparse_opt_args, sparse_opt_factory = (
        sparse_optimizer_factory_and_class()
    )
                                            
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
    sharders = get_default_sharders()
    sharders = _maybe_apply_qcomm_a2a(
        sharders,
        device,
        forward_precision=sparse_a2a_forward_precision,
        backward_precision=sparse_a2a_backward_precision,
        lowmem_clamp_cast=qcomm_lowmem_clamp_cast,
    )
                                                                             
                                                                              
                                                                            
    sharders = _maybe_apply_ec_index_dedup(
        sharders, enabled=ec_index_dedup or use_gather_select
    )
                                                                             
                                                                            
                                                               
    logger.info(
        "[hbm-cap] make_optimizer_and_shard: hbm_cap_gb=%s (planner Topology hbm_cap=%d bytes), "
        "world_size=%s local_world_size=%s",
        hbm_cap_gb,
        hbm_cap_gb * 1024 * 1024 * 1024,
        world_size,
        local_world_size or world_size,
    )
                                                                            
                                                                                   
                                                                                
                                                                                  
                                                                             
    constraints = _build_placement_constraints(
        model=model,
        embedding_placement=embedding_placement,
        embedding_placement_overrides=embedding_placement_overrides or {},
        embedding_sharding_overrides=embedding_sharding_overrides or {},
        embedding_table_configs=embedding_table_configs,
    )
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=local_world_size or world_size,
            world_size=world_size,
            compute_device="cuda",
            hbm_cap=hbm_cap_gb * 1024 * 1024 * 1024,
            ddr_cap=0,
        ),
        constraints=constraints or None,
    )
    pg = dist.GroupMember.WORLD
    env = ShardingEnv.from_process_group(pg)                   
    pg = env.process_group

    plan = planner.collective_plan(model, sharders, pg)

                                                                                 
                                                                                   
                                                                                  
                                                                                   
    if (dist.get_rank() if dist.is_initialized() else 0) == 0:
        try:
            for module_path, param_plans in plan.plan.items():
                for param_name, ps in param_plans.items():
                    logger.info(
                        "[emb-placement] plan: %s.%s -> compute_kernel=%s "
                        "sharding_type=%s",
                        module_path,
                        param_name,
                        getattr(ps, "compute_kernel", "?"),
                        getattr(ps, "sharding_type", "?"),
                    )
        except Exception as e:                                           
            logger.warning("[emb-placement] could not dump plan kernels: %s", e)

                                                                               
                                                                               
                                                                               
                                                                              
                                                                                 
                                                                             
                                                
    _emb_seed = int(os.environ.get("SEED", "1"))
    torch.manual_seed(_emb_seed)
    torch.cuda.manual_seed_all(_emb_seed)
    logger.info(f"[emb-init] re-seeded RNGs before DMP with SEED={_emb_seed}")

                 
    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
    )

                                                                                 
                                                                                
                                                                                 
                                                                           
                                                                                
                                                                              
                                     
                                                                      
                                                                            
                                                                             
                                                                                  
                                                                             
                                                                         
                      
    if os.environ.get("INIT_CHECKSUM", "0") == "1":
        import hashlib

        _rank = dist.get_rank() if dist.is_initialized() else 0
        _fps: List[str] = []
        for _name, _p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            _sharded = isinstance(_p, ShardedTensor)
            if _sharded:
                _shards = _p.local_shards()
                _loc = _shards[0].tensor if _shards else None
            else:
                _loc = _p
            if _loc is None or _loc.numel() == 0:
                _cnt, _sm, _sq = 0.0, 0.0, 0.0
            else:
                _det = _loc.detach()
                _cnt = float(_det.numel())
                _sm = _det.sum(dtype=torch.float64).item()
                _nrm = torch.linalg.vector_norm(
                    _det, ord=2, dtype=torch.float64
                ).item()
                _sq = _nrm * _nrm
            if _sharded and dist.is_initialized():
                _stat = torch.tensor(
                    [_cnt, _sm, _sq], dtype=torch.float64, device=device
                )
                dist.all_reduce(_stat, op=dist.ReduceOp.SUM)
                _cnt, _sm, _sq = _stat.tolist()
            _fps.append(f"{_name}|{int(_cnt)}|{_sm:.6f}|{_sq:.6f}")
            if _rank == 0:
                logger.info(
                    f"[init-checksum] {'sharded' if _sharded else 'dense'} "
                    f"{_name} n={int(_cnt)} sum={_sm:.6f} sumsq={_sq:.6f}"
                )
        if _rank == 0:
            _digest = hashlib.sha256("\n".join(_fps).encode()).hexdigest()[:16]
            logger.info(
                f"[init-checksum] SEED={os.environ.get('SEED', '?')} "
                f"params={len(_fps)} digest={_digest}"
            )

                            
    all_optimizers = []
    all_params = {}
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
            else:
                all_params[k] = v

    if non_fused_sparse_params:
        all_optimizers.append(
            (
                "sparse_non_fused",
                KeyedOptimizerWrapper(
                    params=non_fused_sparse_params, optim_factory=sparse_opt_factory
                ),
            )
        )

    if all_params:
        all_optimizers.append(
            (
                "dense",
                KeyedOptimizerWrapper(
                    params=all_params,
                    optim_factory=dense_opt_factory,
                ),
            )
        )
    output_optimizer = CombinedOptimizer(all_optimizers)
    output_optimizer.init_state(set(model.sparse_grad_parameter_names()))
    return model, output_optimizer


def _init_dataloader_worker(_worker_id: int, num_threads: int) -> None:
    ''                                                                     
    torch.set_num_threads(num_threads)


@gin.configurable
def make_streaming_dataloader(
    dataset: HammerToTorchDataset,
    ts: Optional[int] = None,
    batch_size: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 0,
    train_only: bool = False,
    indices: Optional["np.ndarray"] = None,
    use_compacted_preprocessor: bool = True,
    worker_num_threads: int = 4,
) -> DataLoader:
                                                                         
                                                                                
                                                                           
                                                                            
                                  
    if indices is not None:
        dataset.dataset.set_active_indices(indices)                    
    else:
        assert ts is not None, "make_streaming_dataloader needs ts or indices"
        dataset.dataset.set_ts(ts, train_only=train_only)                    
    total_items = dataset.dataset.get_item_count()
    subset = torch.utils.data.Subset(dataset, range(total_items))
                                                                           
                                                                                
                                                                             
                                                                           
                                                             
    mp_ctx = "fork" if num_workers and num_workers > 0 else None
    dataloader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch_fn(dataset.dataset, use_compacted_preprocessor),
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        sampler=DistributedSampler(subset, shuffle=False, drop_last=True),
        multiprocessing_context=mp_ctx,
        worker_init_fn=partial(_init_dataloader_worker, num_threads=worker_num_threads),
    )
    return dataloader


class StreamingWindowSampler(torch.utils.data.Sampler):
    ''                                                          

                                                                             
                                                                              
                                                                               
                                                                                
                                                                             
                                                                      

                                                                          
                                                                             
                                                                           
                                           
       

    def __init__(self, rank: int, world_size: int) -> None:
        self._rank: int = rank
        self._world_size: int = world_size
        self._indices: List[int] = []

    def set_window(self, global_indices, skip_samples: int = 0) -> None:
        ''                                                                       

                                                                                
                                                                            
                                                              
                                                                            
                                                                

                                                                              
                                                                               
                                                                
           
        n = (len(global_indices) // self._world_size) * self._world_size
        per_rank = global_indices[:n][self._rank :: self._world_size].tolist()
        if skip_samples < 0 or skip_samples > len(per_rank):
            raise ValueError(
                f"skip_samples={skip_samples} out of [0, {len(per_rank)}] "
                f"for rank={self._rank} world_size={self._world_size}"
            )
        self._indices = per_rank[skip_samples:]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


@gin.configurable
def make_persistent_streaming_dataloader(
    dataset: HammerToTorchDataset,
    sampler: StreamingWindowSampler,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    use_compacted_preprocessor: bool = True,
    worker_num_threads: int = 4,
) -> DataLoader:
    ''                                                                    
                                                                             
    use_workers = bool(num_workers and num_workers > 0)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch_fn(dataset.dataset, use_compacted_preprocessor),
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=use_workers,
        multiprocessing_context="fork" if use_workers else None,
        worker_init_fn=partial(_init_dataloader_worker, num_threads=worker_num_threads),
    )


class _PrefetchingWindowLoader:
    ''                                                                 

                                                                                
                                                                              
                                                                              
                                                                               
                                                                           
                                           

                                                                               
                                                                           
                                                           
       

    def __init__(
        self,
        dataset: "HammerToTorchDataset",
        sampler_factory,
        dl_factory,
        n_buffers: int = 2,
    ) -> None:
        self._dataset = dataset
        self._n = n_buffers
        self._samplers = [sampler_factory() for _ in range(n_buffers)]
        self._dls = [dl_factory(s) for s in self._samplers]
        self._iters: List[Optional[object]] = [None] * n_buffers

    def _prepare(self, buf: int, ts: int, skip_samples: int = 0) -> None:
                                                                            
                                                                          
                                                                             
                                                                           
                                                                           
                                                                                 
        self._samplers[buf].set_window(
            self._dataset.dataset.train_window_indices(ts), skip_samples=skip_samples
        )
        self._iters[buf] = iter(self._dls[buf])

    def stream(self, ts_list: List[int], first_skip_samples: int = 0):
        ''                                                                     
                                                                          
                                                                         
        n = len(ts_list)
        if n == 0:
            return
        threads: List[Optional[threading.Thread]] = [None] * self._n
                                                                                 
        for b in range(min(self._n, n)):
            skip = first_skip_samples if b == 0 else 0
            self._prepare(b, ts_list[b], skip_samples=skip)
        for i in range(n):
            buf = i % self._n
            if threads[buf] is not None:
                threads[buf].join()
                threads[buf] = None
            yield ts_list[i], self._iters[buf]
                                                                         
                                                                            
                                           
            j = i + self._n
            if j < n:
                th = threading.Thread(
                    target=self._prepare, args=(buf, ts_list[j]), daemon=True
                )
                th.start()
                threads[buf] = th


@gin.configurable
def make_train_test_dataloaders(
    batch_size: int,
    dataset_type: str,
    hstu_config: DlrmHSTUConfig,
    train_split_percentage: float,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    new_path_prefix: str = "",
    train_num_workers: int = 0,
    eval_num_workers: int = 0,
    num_blocks: int = 1,
    train_prefetch_factor: Optional[int] = None,
    eval_prefetch_factor: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    use_compacted_preprocessor: bool = True,
    worker_num_threads: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    dataset_class, kwargs = get_dataset(
        name=dataset_type, new_path_prefix=new_path_prefix
    )
    kwargs["embedding_config"] = embedding_table_configs

                    
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
    total_items = dataset.dataset.get_item_count()
    items_per_block = total_items // num_blocks
    train_size_per_block = round(train_split_percentage * items_per_block)
                                                                               
                                                                           
                                                                                
                                                              
    if num_blocks == 1:
        train_inds = range(0, train_size_per_block)
        test_inds = range(train_size_per_block, items_per_block)
    else:
        train_inds = _ChainedRanges([
            range(i * items_per_block, i * items_per_block + train_size_per_block)
            for i in range(num_blocks)
        ])
        test_inds = _ChainedRanges([
            range(i * items_per_block + train_size_per_block, (i + 1) * items_per_block)
            for i in range(num_blocks)
        ])
    train_set = torch.utils.data.Subset(dataset, train_inds)
    test_set = torch.utils.data.Subset(dataset, test_inds)

                                                                                   
                                                                                 
                                                                                
                                                                                
                                                                             
                                                
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch_fn(dataset.dataset, use_compacted_preprocessor),
        drop_last=True,
        num_workers=train_num_workers,
        prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
        pin_memory=True,
        sampler=ChunkDistributedSampler(train_set, drop_last=True, shuffle=True),
        multiprocessing_context="fork" if train_num_workers > 0 else None,
        worker_init_fn=partial(_init_dataloader_worker, num_threads=worker_num_threads),
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size if eval_batch_size is not None else batch_size,
        shuffle=False,
        collate_fn=_collate_batch_fn(dataset.dataset, use_compacted_preprocessor),
        drop_last=True,
        num_workers=eval_num_workers,
        prefetch_factor=eval_prefetch_factor if eval_num_workers > 0 else None,
        pin_memory=True,
        sampler=ChunkDistributedSampler(test_set, drop_last=True, shuffle=True),
        multiprocessing_context="fork" if eval_num_workers > 0 else None,
        worker_init_fn=partial(_init_dataloader_worker, num_threads=worker_num_threads),
    )
    return train_dataloader, test_dataloader


                                                                             
                                                                                 
                                                                    
_EMB_DIAG_ACC: Dict[str, Dict[str, Any]] = {}
_EMB_DIAG_NBATCH: int = 0
                                                                            
                                                                            
                                                                              
                     
_EMB_DIAG_GLOBAL_CAP: int = 1 << 17          


def _log_unique_embedding_diag(
    sample, rank: int, step: int, max_steps: int = 100, log_every: int = 50
) -> None:
    ''                                                                           

                                                                                
                                                                                 
                                                                                  
                                                                           
                                                                                

                                                                               
                                                                           
                                                                          
       
    if hasattr(sample, "compacted_keys"):
        return
    if rank != 0:
        return
    global _EMB_DIAG_NBATCH
    try:
        from generative_recommenders.dlrm_v4.configs import YAMBDA_5B_CROSS_SPECS

        cross_caps = {name: n for (name, _k, n, _s) in YAMBDA_5B_CROSS_SPECS}

        def _table_of(key: str):
                                                                                  
                                                                           
            if key in cross_caps:
                return key, cross_caps[key]
            if key == "uid" or key.endswith("_uid") or key.endswith(".uid"):
                return "uid", 0
            if "artist" in key:
                return "artist_id", 0
            if "album" in key:
                return "album_id", 0
            if "item" in key and key.endswith("id"):
                return "item_id", 0
            return None, 0

        for tag, kjt in (
            ("uih", sample.uih_features_kjt),
            ("cand", sample.candidates_features_kjt),
        ):
            for key in kjt.keys():
                table, cap = _table_of(key)
                if table is None:
                    continue
                vals = kjt[key].values()
                total = int(vals.numel())
                if total == 0:
                    continue
                u, counts = torch.unique(vals, return_counts=True)
                uniq = int(u.numel())
                hot1 = int(counts.max().item())
                k = min(10, uniq)
                topk = int(torch.topk(counts, k).values.sum().item())

                slot = _EMB_DIAG_ACC.setdefault(
                    f"{tag}.{key}",
                    {
                        "table": table,
                        "cap": cap,
                        "n": 0,
                        "tot": 0,
                        "uniq": 0,
                        "upct": 0.0,
                        "upct_min": 100.0,
                        "upct_max": 0.0,
                        "hot1pct": 0.0,
                        "topkpct": 0.0,
                        "glob": None,                                                  
                    },
                )
                upct = 100.0 * uniq / total
                slot["n"] += 1
                slot["tot"] += total
                slot["uniq"] += uniq
                slot["upct"] += upct
                slot["upct_min"] = min(slot["upct_min"], upct)
                slot["upct_max"] = max(slot["upct_max"], upct)
                slot["hot1pct"] += 100.0 * hot1 / total
                slot["topkpct"] += 100.0 * topk / total
                if total <= _EMB_DIAG_GLOBAL_CAP:
                    prev = slot["glob"]
                    merged = u if prev is None else torch.cat([prev, u])
                    slot["glob"] = torch.unique(merged)

        _EMB_DIAG_NBATCH += 1
        n = _EMB_DIAG_NBATCH
        if n % log_every == 0 or n >= max_steps:
            lines = [f"emb-diag AGGREGATE over {n} batches (step<= {step}):"]
            for name in sorted(_EMB_DIAG_ACC):
                s = _EMB_DIAG_ACC[name]
                c = max(1, s["n"])
                cap_s = f" cap={s['cap']/1e6:.0f}M" if s["cap"] else ""
                glob_s = ""
                if s["glob"] is not None:
                    g = int(s["glob"].numel())
                    glob_s = (
                        f" | global_uniq={g} over {s['tot']} seen "
                        f"({s['tot']/max(1,g):.1f}x reuse)"
                    )
                lines.append(
                    f"  {name}[{s['table']}]{cap_s}: "
                    f"avg_tot={s['tot']/c:.0f} "
                    f"avg_uniq%={s['upct']/c:.1f} "
                    f"(min={s['upct_min']:.1f} max={s['upct_max']:.1f}) "
                    f"avg_hot1%={s['hot1pct']/c:.1f} "
                    f"avg_top10%={s['topkpct']/c:.1f}"
                    f"{glob_s}"
                )
            logger.info("\n".join(lines))
    except Exception as e:                                        
        logger.warning(f"emb-diag failed: {e}")


@gin.configurable
def train_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    start_batch_idx: int = 0,
    streaming_diag_unique_emb: bool = False,
                                          
) -> None:
    model.train()
    batch_idx: int = start_batch_idx
    profiler = Profiler(rank) if output_trace else None

    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)                    
        for sample in dataloader:
            if streaming_diag_unique_emb and batch_idx < int(
                os.environ.get("DIAG_EMB_STEPS", "100")
            ):
                _log_unique_embedding_diag(
                    sample,
                    rank,
                    batch_idx,
                    max_steps=int(os.environ.get("DIAG_EMB_STEPS", "100")),
                    log_every=metric_log_frequency,
                )
            optimizer.zero_grad()
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = _forward_sample(model, sample)
                         
            sum(aux_losses.values()).backward()
            optimizer.step()
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=_batch_num_candidates(sample),
            )
            if batch_idx % metric_log_frequency != 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if batch_idx % checkpoint_frequency == 0 and batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=batch_idx,
                )
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
        if num_batches is not None and batch_idx >= num_batches:
            break


@gin.configurable
def eval_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metric_logger: MetricsLogger,
    device: torch.device,
    metric_log_frequency: int = 1,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
                                          
) -> None:
    model.eval()
                                                                                    
    metric_logger.pause_perf("eval")
    batch_idx: int = 0
    profiler = Profiler(rank) if output_trace else None
    metric_logger.reset(mode="eval")
    with torch.no_grad():
        for sample in dataloader:
            sample.to(device)
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = _forward_sample(model, sample)
            metric_logger.update(
                mode="eval",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=_batch_num_candidates(sample),
            )
            if batch_idx % metric_log_frequency != 0:
                metric_logger.compute_and_log(mode="eval")
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
    metric_logger.compute_and_log(mode="eval")
    for k, v in metric_logger.compute(mode="eval").items():
        print(f"{k}: {v}")
    metric_logger.resume_perf("eval")


class _PipelineModelWrapper(torch.nn.Module):
    ''                                                                   
                                        

                                                                            
                                                                            
                                                                               
                                   
       

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(self, batch: Any) -> Tuple[torch.Tensor, Tuple[Any, ...]]:
                                                                             
                                                                               
                                                                               
                                                                              
                          
        (
            _,
            _,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        ) = self._model(batch)
        loss = sum(aux_losses.values())
        num_candidates = _batch_num_candidates(batch)
        output = (
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
            num_candidates,
        )
        return loss, output


class _ClippingTrainPipelineSparseDist(TrainPipelineSparseDist):
    ''                                                                     

                                                                           
                                                                         
                                                                      
       

    def __init__(self, *args: Any, grad_clip_norm: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._grad_clip_norm = grad_clip_norm

    def _optimizer_step(self) -> None:
        if self._grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), max_norm=self._grad_clip_norm
            )
        super()._optimizer_step()


def build_train_pipeline(
    model: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> TrainPipelineSparseDist:
    ''                                                                    

                                                                               
                                                                               
                                                                               
                                                                    
       
                                                                                 
                                                                         
                                                                                
                                                                                   
    underlying = model.module if hasattr(model, "module") else model
    underlying._pipeline_mode = True

    return _ClippingTrainPipelineSparseDist(
        model=model,
        optimizer=optimizer,
        device=device,
        execute_all_batches=True,
        custom_model_fwd=_PipelineModelWrapper(model),
        grad_clip_norm=grad_clip_norm or 0.0,
    )


class _TrainWindowFeed:
    ''                                                                       

                                                                           
                                                                          
                                                                             
                                                                     
                                                                           
                                                                         
       

    def __init__(self, iterator: Iterator[Any]) -> None:
        self._iterator = iterator
        self.exhausted = False

    def segment(self, num_batches: Optional[int]) -> Iterator[Any]:
        taken = 0
        while num_batches is None or taken < num_batches:
            try:
                batch = next(self._iterator)
            except StopIteration:
                self.exhausted = True
                return
            taken += 1
            yield batch


@gin.configurable
def train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    eval_frequency: int = 1,
    start_train_batch_idx: int = 0,
    start_eval_batch_idx: int = 0,
    use_pipeline: bool = False,
                                          
) -> None:
    train_batch_idx: int = start_train_batch_idx
    eval_batch_idx: int = start_eval_batch_idx
    profiler = Profiler(rank) if output_trace else None
    assert train_dataloader is not None and eval_dataloader is not None

    eval_data_iterator = iter(eval_dataloader)
    train_data_iterator = iter(train_dataloader)

                                                                            
                                                                          
                                                                                
    train_pipeline = (
        build_train_pipeline(model, optimizer, device) if use_pipeline else None
    )

    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)                    
        while True:
            model.train()
            if train_pipeline is not None:
                try:
                    (
                        aux_losses,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                        num_candidates,
                    ) = train_pipeline.progress(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    break
            else:
                try:
                    sample = next(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    break
                optimizer.zero_grad()
                sample.to(device)
                (
                    _,
                    _,
                    aux_losses,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = _forward_sample(model, sample)
                             
                sum(aux_losses.values()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                num_candidates = _batch_num_candidates(sample)
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=num_candidates,
            )
            if train_batch_idx % metric_log_frequency == 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if train_batch_idx % checkpoint_frequency == 0 and train_batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=train_batch_idx,
                )
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if train_batch_idx % eval_frequency == 0:
                model.eval()
                eval_batch_idx: int = 0
                with torch.no_grad():
                    while True:
                        try:
                            sample = next(eval_data_iterator)
                        except StopIteration:
                            eval_data_iterator = iter(eval_dataloader)
                            sample = next(eval_data_iterator)
                        sample.to(device)
                        (
                            _,
                            _,
                            _,
                            mt_target_preds,
                            mt_target_labels,
                            mt_target_weights,
                        ) = _forward_sample(model, sample)
                        metric_logger.update(
                            mode="eval",
                            predictions=mt_target_preds,
                            labels=mt_target_labels,
                            weights=mt_target_weights,
                            num_candidates=_batch_num_candidates(sample),
                        )
                        eval_batch_idx += 1
                        if output_trace:
                            assert profiler is not None
                            profiler.step()
                        if eval_batch_idx % metric_log_frequency == 0:
                            metric_logger.compute_and_log(mode="eval")
                        if (
                            num_eval_batches is not None
                            and eval_batch_idx >= num_eval_batches
                        ):
                            break
                    for k, v in metric_logger.compute(mode="eval").items():
                        print(f"{k}: {v}")
                model.train()
                                                                                
                                                                              
                                               
            if num_train_batches and train_batch_idx >= num_train_batches:
                break


def select_in_window_checkpoint_reason(
    *,
    train_batch_idx: int,
    global_step: int,
    elapsed_since_last_save: float,
    in_window_checkpoint_frequency: int,
    checkpoint_step_frequency: int,
    checkpoint_time_interval_s: float,
) -> Optional[str]:
    ''                                                                     

                                                                            
                                                                               
                                                                                

                                                                           
                                                                           
                                                                         
                                  

                                                                    
                                                                           
                                                      
       
    if (
        in_window_checkpoint_frequency > 0
        and train_batch_idx % in_window_checkpoint_frequency == 0
    ):
        return "in_window_batch"
    if (
        checkpoint_step_frequency > 0
        and global_step > 0
        and global_step % checkpoint_step_frequency == 0
    ):
        return "global_step"
    if (
        checkpoint_time_interval_s > 0
        and elapsed_since_last_save >= checkpoint_time_interval_s
    ):
        return "time_interval"
    return None


def data_pct_to_global_steps(
    *,
    data_pct: float,
    total_train_samples: int,
    global_batch_size: int,
) -> int:
    ''                                                                         
                                                                            

                                                                       
                                                                            
                                                                             
                                                                        
                                                                              
                                                                                
                                                                               
                                                                               
                                                                          
       
    if data_pct <= 0 or total_train_samples <= 0 or global_batch_size <= 0:
        return 0
    return int(
        math.ceil(data_pct * float(total_train_samples) / float(global_batch_size))
    )


def resolve_skip_eval_until_step(
    *,
    skip_eval_epoch_pct: float,
    total_train_samples: int,
    global_batch_size: int,
) -> int:
    ''                                                                

                                                                            
                                                                           
                                                                               
                                                                     

                                                                             
                                                                         
                                                                               
                                                                            
                                                                             
                                                                                
                                                                            
                                                                                

                                                                               
              
       
    return data_pct_to_global_steps(
        data_pct=skip_eval_epoch_pct,
        total_train_samples=total_train_samples,
        global_batch_size=global_batch_size,
    )


def _validate_split_contract(
    saved: Optional[Dict[str, Any]],
    live: Dict[str, Any],
    rank: int,
) -> None:
    ''                                                                      
                                                    

                                                                                
                                                                            
                                                                                
                                                                     
                                                                            
                                                       
       
    allow = os.environ.get("ALLOW_SPLIT_MISMATCH", "0") == "1"
    if saved is None:
                                                                          
                                                                             
                                                    
        if live.get("train_split_percentage", 1.0) < 1.0 and not allow:
            raise RuntimeError(
                "Resuming a checkpoint with NO saved split contract into a "
                f"user-holdout run (train_split_percentage="
                f"{live['train_split_percentage']}). The earlier run's split "
                "cannot be verified, so held-out eval users may have been "
                "trained. Set ALLOW_SPLIT_MISMATCH=1 to override."
            )
        return
    mismatches = {
        k: (saved.get(k), live.get(k))
        for k in live
        if saved.get(k) != live.get(k)
    }
    if mismatches:
        msg = (
            "Split/resume contract mismatch between checkpoint and current run: "
            + ", ".join(
                f"{k}: checkpoint={s!r} current={c!r}" for k, (s, c) in mismatches.items()
            )
            + ". Resuming would desync the skip offset and/or leak held-out "
            "users into training."
        )
        if allow:
            if rank == 0:
                logger.warning("%s ALLOW_SPLIT_MISMATCH=1 set — continuing anyway.", msg)
        else:
            raise RuntimeError(msg + " Set ALLOW_SPLIT_MISMATCH=1 to override.")
    elif rank == 0:
        logger.info("Split/resume contract verified against checkpoint: %s", live)


@gin.configurable
def streaming_train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_train_ts: int,
    hstu_config: DlrmHSTUConfig,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    train_prefetch_factor: int = 2,
    eval_prefetch_factor: int = 8,
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
                                                                              
                                                                       
                                                                                
                                                                               
                                                                             
                                                                                
    eval_interim_metrics: int = 0,
                                                                              
                                                                     
                                                                       
                                          
    mlperf_train_loss_log_frequency: int = 0,
    checkpoint_frequency: int = 100,
    start_ts: int = 0,
    persistent_loader: bool = False,
    eval_every_n_windows: int = 1,
                                                                                
                                                                               
                                                                         
                                                      
    eval_every_data_pct: float = 0.0,
                                                                              
                                                                 
                                                                           
                                                                           
                                                                                
                                                           
                                                                       
                                                                                
                                                                            
                                                                   
    skip_eval_epoch_pct: float = 0.0,
    double_buffer: bool = False,
                                         
                                                                       
                                                                               
    eval_holdout_ts: Optional[int] = None,
    eval_holdout_num_windows: int = 1,
                                                  
    resume_train_ts: Optional[int] = None,
    resume_batch_idx_in_window: int = WINDOW_COMPLETE,
                                                                         
                                                                              
                                                                    
    resume_split_contract: Optional[Dict[str, Any]] = None,
                                                                            
                                                                                
                                                                                
    resume_cold_start: bool = False,
    in_window_checkpoint_frequency: int = 0,
                                                          
    checkpoint_step_frequency: int = 0,
    checkpoint_time_interval_s: float = 0.0,
                                                                            
                                                                                    
    grad_clip_norm: float = 0.0,
                                                                               
                                                                             
                                                                           
                                                                          
                                                
    use_pipeline: bool = False,
                                                                               
                                                                              
                                                                          
                                                            
    lr_warmup_steps: int = 0,
                                                                               
                                                                                
                                                                                 
                                                                                  
                                                                                
                                                                                    
                                                                
                                          
    lr_warmup_start_lr: float = 0.0,
                                                                        
    streaming_diag_unique_emb: bool = False,
                                              
    die_at_step: int = -1,
                                                                            
    mlperf_logger: Optional[Any] = None,
    train_num_workers: int = 0,
    eval_num_workers: int = 0,
) -> None:
    ''                                                                      
                

                                                                               
                                                                 
                                                                          
                                                                           
                                                                            
                                                                           
                                                                                   
                                                                                   

                                                                          
                                                                            
                                                                            
                                                                             
                                                                       
                                                                            
                                                                             
                                                                         
                                                                      
                                                                           
                                                                           
                                                                             
                                                             
                                                               

                                                             
                                                                            
                                                                              
                                                              
                                                                             
                                                                           

                                                                                
                                                                                
                                                                              
                             
       
                                                                                
                                                                         
                                                                                 
                                                                 
    if (eval_every_data_pct and eval_every_data_pct > 0) and eval_every_n_windows > 0:
        raise ValueError(
            "Conflicting eval cadences: eval_every_data_pct="
            f"{eval_every_data_pct} (>0) AND eval_every_n_windows="
            f"{eval_every_n_windows} (>0). They are mutually exclusive. To use "
            "the data-fraction cadence set EVAL_EVERY_N_WINDOWS=0; to use the "
            "per-window cadence set EVAL_EVERY_DATA_PCT=0."
        )
                                                                            
                                                                               
                                               
    mlperf_loss_every = (
        mlperf_train_loss_log_frequency
        if mlperf_train_loss_log_frequency and mlperf_train_loss_log_frequency > 0
        else metric_log_frequency
    )
    profiler = Profiler(rank) if output_trace else None
                                                                                
                                                                                
                                                                               
                                                                               
    if num_eval_batches is not None and num_eval_batches <= 0:
        num_eval_batches = None
    if num_train_batches is not None and num_train_batches <= 0:
        num_train_batches = None
    dataset_class, kwargs = get_dataset()
    kwargs["embedding_config"] = embedding_table_configs
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
                                                                              
                                                                            
                                                                         
                                                                         
    window_sampler: Optional[StreamingWindowSampler] = None
    persistent_dl: Optional[DataLoader] = None
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    if persistent_loader:
        window_sampler = StreamingWindowSampler(rank=rank, world_size=world_size)
        persistent_dl = make_persistent_streaming_dataloader(
            dataset=dataset,
            sampler=window_sampler,
            num_workers=train_num_workers,
            prefetch_factor=train_prefetch_factor,
        )

                                                                            
                                                                          
                                           
    supports_holdout = hasattr(dataset.dataset, "eval_holdout_indices")

                                                                             
                                                                               
                                                                             
                                                                          
    requested_end_ts = start_ts + num_train_ts
                                                                               
                                                                            
                                                                             
                                                                                 
                                                                             
                                                                
    eval_anchor_ts = start_ts
                                                                              
                                                                
    eval_holdout_ts_resolved = (
        eval_holdout_ts
        if (eval_holdout_ts is not None and eval_holdout_ts >= 0)
        else requested_end_ts
    )

                                                                         
                                                                            
                                                                              
                                                                              
                                                              
    bs = (
        persistent_dl.batch_size
        if persistent_dl is not None
        else int(os.environ.get("BATCH_SIZE", "1024"))
    )
    global_batch_size = bs * world_size

                                                                           
                                                                        
                                                                      
                                                                       
                                                                             
                                                                          
    eval_interval_steps = 0
    if eval_every_data_pct and eval_every_data_pct > 0:
        if hasattr(dataset.dataset, "total_train_anchors"):
                                                                              
                                                                       
                                                                              
                                                                               
                                                                                
                                                                           
                                                                              
                                                                            
                                                                            
            if world_size > 1 and torch.distributed.is_initialized():
                _tta = (
                    dataset.dataset.total_train_anchors(                   
                        eval_anchor_ts, requested_end_ts - eval_anchor_ts
                    )
                    if rank == 0
                    else 0
                )
                _tta_t = torch.tensor([_tta], dtype=torch.int64, device=device)
                torch.distributed.broadcast(_tta_t, src=0)
                total_train_anchors = int(_tta_t.item())
            else:
                total_train_anchors = dataset.dataset.total_train_anchors(                   
                    eval_anchor_ts, requested_end_ts - eval_anchor_ts
                )
                                                                              
                                                                            
                                                                                 
                                                                          
                                                                                 
                                                                                 
            total_train_steps = total_train_anchors // max(1, global_batch_size)
            eval_interval_steps = max(
                1,
                data_pct_to_global_steps(
                    data_pct=eval_every_data_pct,
                    total_train_samples=total_train_anchors,
                    global_batch_size=global_batch_size,
                ),
            )
            if rank == 0:
                logger.info(
                    "[data-pct-eval] eval_every_data_pct=%.6g -> "
                    "eval_interval_steps=%d (total_train_anchors=%d bs=%d "
                    "world_size=%d total_train_steps=%d over windows [%d, %d))",
                    eval_every_data_pct,
                    eval_interval_steps,
                    total_train_anchors,
                    bs,
                    world_size,
                    total_train_steps,
                    eval_anchor_ts,
                    requested_end_ts,
                )
        elif rank == 0:
            logger.warning(
                "[data-pct-eval] dataset %s has no total_train_anchors(); "
                "data-fraction eval is DISABLED (no per-window eval either, "
                "since EVAL_EVERY_N_WINDOWS must be 0 to reach here) — only the "
                "final eval will run.",
                type(dataset.dataset).__name__,
            )

                                                                           
                                                                               
                                                                            
                                                                                
                                                                              
    live_split_contract: Optional[Dict[str, Any]] = None
    if supports_holdout:
        live_split_contract = {
            "train_split_percentage": dataset.dataset._train_split_percentage,                   
            "split_salt": dataset.dataset._split_salt,                   
            "eval_holdout_ts": eval_holdout_ts_resolved,
            "eval_holdout_num_windows": eval_holdout_num_windows,
            "batch_size": persistent_dl.batch_size if persistent_dl is not None else None,
            "world_size": world_size,
        }
                                                                                
                                                                                 
                                                                                 
                                                                               
                                                            
        if not resume_cold_start:
            _validate_split_contract(resume_split_contract, live_split_contract, rank)

                                                                            
                                                                         
                                                                           
                                                                             
                                                                    
    first_skip_samples = 0
    if resume_train_ts is not None:
        original_end_ts = start_ts + num_train_ts
        if resume_batch_idx_in_window == WINDOW_COMPLETE:
            new_start = resume_train_ts + 1
            if rank == 0:
                logger.info(
                    "Resuming from completed train_ts=%d → start_ts=%d "
                    "(num_train_ts %d → %d)",
                    resume_train_ts, new_start,
                    num_train_ts, max(0, original_end_ts - new_start),
                )
            start_ts = new_start
        else:
            if rank == 0:
                logger.info(
                    "Resuming mid-window at train_ts=%d batch_idx_in_window=%d "
                    "(skipping batches already trained)",
                    resume_train_ts,
                    resume_batch_idx_in_window,
                )
            start_ts = resume_train_ts
                                                                              
                                                                         
                                                                         
                                                                           
                                                                       
            assert persistent_dl is not None, (
                "Mid-window resume requires persistent_loader=True"
            )
            first_skip_samples = resume_batch_idx_in_window * persistent_dl.batch_size
        num_train_ts = max(0, original_end_ts - start_ts)
        if num_train_ts == 0 and rank == 0:
            logger.info(
                "Resume target already reached (end_ts=%d, start_ts=%d) — "
                "no further training windows; skipping straight to final eval.",
                original_end_ts, start_ts,
            )

    if rank == 0:
        logger.info(
            "[grad-clip] streaming path gradient clipping %s (max_norm=%.4g via $GRAD_CLIP_NORM)",
            "ENABLED" if (grad_clip_norm and grad_clip_norm > 0) else "OFF",
            grad_clip_norm,
        )

    train_pipeline: Optional[TrainPipelineSparseDist] = None
    if use_pipeline:
                                                                              
                                                              
        unsupported = {
            "in_window_checkpoint_frequency": in_window_checkpoint_frequency > 0,
            "checkpoint_step_frequency": checkpoint_step_frequency > 0,
            "checkpoint_time_interval_s": checkpoint_time_interval_s > 0,
            "streaming_diag_unique_emb": streaming_diag_unique_emb,
            "die_at_step": die_at_step >= 0,
        }
        enabled = [name for name, on in unsupported.items() if on]
        if enabled:
            raise ValueError(
                f"use_pipeline=True does not support {', '.join(enabled)}"
            )
        train_pipeline = build_train_pipeline(
            model, optimizer, device, grad_clip_norm=grad_clip_norm
        )
        if rank == 0:
            logger.info("[train-pipeline] TrainPipelineSparseDist enabled for streaming training")

    def _detach_train_pipeline() -> None:
                                                                                  
        if train_pipeline is None:
            return
        assert not train_pipeline.batches, "train pipeline has batches in flight"
        train_pipeline.detach()

    def _steps_to_drain(
        gstep: int, train_batch_idx: int, data_pct_eval: bool
    ) -> Optional[int]:
                                                                             
                                                                     
                                                                     
        limits = []
        if data_pct_eval and eval_interval_steps > 0:
            k = max(
                gstep // eval_interval_steps + 1,
                -(-skip_eval_until_step // eval_interval_steps),
                1,
            )
            limits.append(k * eval_interval_steps - gstep)
        if num_train_batches:
            limits.append(num_train_batches - train_batch_idx)
        return min(limits) if limits else None

    def _window_iter(ts: int, skip_samples: int = 0):
                                                                            
                                                                              
                                          
        if persistent_loader:
            assert window_sampler is not None and persistent_dl is not None
            window_sampler.set_window(
                dataset.dataset.train_window_indices(ts),                    
                skip_samples=skip_samples,
            )
            return iter(persistent_dl)
        if skip_samples != 0:
            raise NotImplementedError(
                "skip_samples>0 requires persistent_loader=True"
            )
        return iter(
            make_streaming_dataloader(
                dataset=dataset,
                ts=ts,
                train_only=True,
                num_workers=train_num_workers,
                prefetch_factor=train_prefetch_factor,
            )
        )
                                                                                
                                                                                
                                                                            
                                                                              
                                                        
    if hasattr(dataset.dataset, "num_windows"):
        available = dataset.dataset.num_windows()                    
        max_count = max(0, available - 1 - start_ts)
        if num_train_ts > max_count:
            logger.warning(
                f"start_ts={start_ts} + num_train_ts={num_train_ts} exceeds "
                f"available windows ({available}); clamping num_train_ts to {max_count}."
            )
            num_train_ts = max_count
                                                                            
                                                                           
                                                                      
    last_ckpt_time = [time.time()]

    def _broadcast_elapsed() -> float:
        ''                                                                  
                                                                                
                                                                              
                                                                                 
                                                                           
        elapsed = time.time() - last_ckpt_time[0]
        if torch.distributed.is_initialized() and world_size > 1:
            t = torch.tensor([elapsed], device=device, dtype=torch.float64)
            torch.distributed.broadcast(t, src=0)
            elapsed = float(t.item())
        return elapsed

    def _save_mid_window(train_ts: int, batch_idx_in_window: int) -> None:
        ''                                                             
                                                                        
                                                                   
                                                                           
                                                                       
                                                                        
        save_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metric_logger,
            rank=rank,
            batch_idx=train_ts,
            train_ts=train_ts,
            batch_idx_in_window=batch_idx_in_window,
            device=device,
            split_contract=live_split_contract,
        )

                                                                               
                                                                             
                                                                          
                                                                              
                                                                              
                                                                         
                                                                             
                                                                               
                                                                                
                                                                    
    _warmup_groups = list(optimizer.param_groups)
    _warmup_fused_opt = getattr(model, "fused_optimizer", None)
    if _warmup_fused_opt is not None:
        _warmup_groups += list(_warmup_fused_opt.param_groups)
    _warmup_base_lrs = [pg["lr"] for pg in _warmup_groups]
                                                                              
                                                                                
                                                                                
    _warmup_start_lrs = [min(float(lr_warmup_start_lr), b) for b in _warmup_base_lrs]
    if lr_warmup_steps and lr_warmup_steps > 0 and rank == 0:
        logger.info(
            "[lr-warmup] linear warmup over %d global steps across %d param "
            "group(s); start LRs: %s -> base LRs: %s",
            lr_warmup_steps,
            len(_warmup_groups),
            ", ".join(f"{s:.3e}" for s in _warmup_start_lrs),
            ", ".join(f"{b:.3e}" for b in _warmup_base_lrs),
        )

    def _apply_lr_warmup(gstep: int) -> None:
                                                                               
                                                                               
        if not (lr_warmup_steps and lr_warmup_steps > 0) or gstep > lr_warmup_steps:
            return
        progress = min(1.0, float(gstep) / float(lr_warmup_steps))
                                                                             
                                                                              
        for pg, base, start in zip(
            _warmup_groups, _warmup_base_lrs, _warmup_start_lrs
        ):
            pg["lr"] = start + (base - start) * progress
                                                                                 
                                              
                                                                        
                                                                                
                                                                                 
                                                                                
                                                                                 
                                                                                
                                                                     
        if _warmup_fused_opt is not None:
            _warmup_fused_opt.step()
        if rank == 0 and gstep % max(1, metric_log_frequency) == 0:
            _fused_lr = ""
            if _warmup_fused_opt is not None:
                _fused_lr = (
                    f" fused_pg_lr={_warmup_fused_opt.param_groups[0]['lr']:.3e}"
                )
                                                                         
                                                                            
                                                                                
                                                                               
                                                                     
                def _first_tbe_kernel_lr(opt, depth=0):
                    if depth > 6 or opt is None:
                        return None
                    _em = getattr(opt, "_emb_module", None)
                    if _em is not None and hasattr(_em, "get_learning_rate"):
                        return float(_em.get_learning_rate())
                    for _sub in getattr(opt, "_optims", []):
                        _child = _sub[1] if isinstance(_sub, tuple) else _sub
                        _lr = _first_tbe_kernel_lr(_child, depth + 1)
                        if _lr is not None:
                            return _lr
                    _wrapped = getattr(opt, "_optimizer", None)
                    if _wrapped is not None:
                        return _first_tbe_kernel_lr(_wrapped, depth + 1)
                    return None

                try:
                    _klr = _first_tbe_kernel_lr(_warmup_fused_opt)
                    if _klr is not None:
                        _fused_lr += f" tbe_kernel_lr={_klr:.3e}"
                except Exception:
                    pass
            logger.info(
                "[lr-warmup] gstep=%d progress=%.4f dense_lr=%.3e%s",
                gstep,
                progress,
                _warmup_groups[0]["lr"],
                _fused_lr,
            )

    def _sequential_train_step(sample: Any, train_batch_idx: int) -> Tuple[Any, ...]:
        if streaming_diag_unique_emb and train_batch_idx < int(
            os.environ.get("DIAG_EMB_STEPS", "100")
        ):
            _log_unique_embedding_diag(
                sample,
                rank,
                train_batch_idx,
                max_steps=int(os.environ.get("DIAG_EMB_STEPS", "100")),
                log_every=metric_log_frequency,
            )
                                                                           
                                                                     
                                                                         
                                                                             
                                       
        _apply_lr_warmup(metric_logger.global_step["train"])
        optimizer.zero_grad()
        sample.to(device, non_blocking=True)
        (
            _,
            _,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        ) = _forward_sample(model, sample)
                     
        sum(aux_losses.values()).backward()
                                                                           
                                                                           
                                                                           
                                                                           
                                                        
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm
            )
        optimizer.step()
        return (
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
            _batch_num_candidates(sample),
        )

    def _run_train_window(
        train_data_iterator,
        train_ts: int,
        start_batch_idx: int = 0,
        label: Optional[str] = None,
        do_eval: Optional[Callable[[int, int], None]] = None,
    ) -> None:
                                                                           
                                                                             
                                                                          
                                                                          
                                                              
        train_batch_idx = start_batch_idx
        first_wait: Optional[float] = None
        feed = (
            _TrainWindowFeed(train_data_iterator)
            if train_pipeline is not None
            else None
        )
        segment: Optional[Iterator[Any]] = None
        segment_steps_left: Optional[int] = None
        while True:
            model.train()
            _t_next = time.perf_counter() if (label and rank == 0) else None
            if train_pipeline is not None:
                assert feed is not None
                if segment is None:
                    if feed.exhausted:
                        break
                    segment_steps_left = _steps_to_drain(
                        metric_logger.global_step["train"],
                        train_batch_idx,
                        data_pct_eval=do_eval is not None,
                    )
                    segment = feed.segment(segment_steps_left)
                                                                           
                                                                                 
                _apply_lr_warmup(metric_logger.global_step["train"])
                try:
                    (
                        aux_losses,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                        num_candidates,
                    ) = train_pipeline.progress(segment)
                except StopIteration:
                                                                 
                    break
                if segment_steps_left is not None:
                    segment_steps_left -= 1
                    if segment_steps_left == 0:
                        segment = None
                if _t_next is not None and first_wait is None:
                    first_wait = time.perf_counter() - _t_next
            else:
                try:
                    sample = next(train_data_iterator)
                except StopIteration:
                    break
                if _t_next is not None and first_wait is None:
                    first_wait = time.perf_counter() - _t_next
                (
                    aux_losses,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                    num_candidates,
                ) = _sequential_train_step(sample, train_batch_idx)
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=num_candidates,
            )
            gstep = metric_logger.global_step["train"]
                                                                            
                                                                                 
                                                                               
                                                                     
            metric_logger.maybe_log_mlperf_train_loss(aux_losses, every=mlperf_loss_every)
                                                                               
                                                                              
            if gstep % metric_log_frequency == 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
                                                                   
                                                                            
                                                                          
                                                                            
                                                                           
                                                                             
                                                           
                                                                             
                                                                        
            elapsed = (
                _broadcast_elapsed() if checkpoint_time_interval_s > 0 else 0.0
            )
            save_reason = select_in_window_checkpoint_reason(
                train_batch_idx=train_batch_idx,
                global_step=gstep,
                elapsed_since_last_save=elapsed,
                in_window_checkpoint_frequency=in_window_checkpoint_frequency,
                checkpoint_step_frequency=checkpoint_step_frequency,
                checkpoint_time_interval_s=checkpoint_time_interval_s,
            )
            if save_reason is not None:
                if rank == 0:
                    logger.info(
                        "checkpoint trigger=%s train_ts=%d batch=%d global_step=%d",
                        save_reason,
                        train_ts,
                        train_batch_idx,
                        gstep,
                    )
                _save_mid_window(train_ts, train_batch_idx)
                                                                          
                                                                      
                last_ckpt_time[0] = time.time()
                                                                                
                                                                             
                                                                              
                                                                             
                                                                                
                                                                                
                                          
            if (
                do_eval is not None
                and eval_interval_steps > 0
                and gstep > 0
                and gstep % eval_interval_steps == 0
            ):
                                                                               
                                                                           
                                                                      
                                                                   
                                                                               
                                                  
                if gstep < skip_eval_until_step:
                    if rank == 0:
                        logger.info(
                            "[data-pct-eval] SKIP eval train_ts=%d global_step=%d "
                            "< skip_eval_until_step=%d "
                            "(SKIP_EVAL_EPOCH_PCT=%.6g)",
                            train_ts,
                            gstep,
                            skip_eval_until_step,
                            skip_eval_epoch_pct,
                        )
                else:
                    if rank == 0:
                        logger.info(
                            "[data-pct-eval] trigger eval train_ts=%d global_step=%d "
                            "(interval=%d)",
                            train_ts,
                            gstep,
                            eval_interval_steps,
                        )
                    do_eval(train_ts, gstep)
                    model.train()
                    dataset.dataset.is_eval = False                    
                                                                                    
                                                                                     
                                                                                  
                                               
                    if mlt.run_stopped:
                        break
                                                                            
                                                                         
                                                                              
            if (
                die_at_step >= 0
                and metric_logger.global_step["train"] >= die_at_step
            ):
                if rank == 0:
                    logger.warning(
                        "die_at_step=%d hit at train_ts=%d batch=%d global_step=%d "
                        "→ sys.exit(42)",
                        die_at_step,
                        train_ts,
                        train_batch_idx,
                        metric_logger.global_step["train"],
                    )
                                                                            
                                                          
                torch.distributed.barrier()
                import sys
                sys.exit(42)
                                                                                
                                                                              
                                               
            if num_train_batches and train_batch_idx >= num_train_batches:
                break
        if label and rank == 0 and first_wait is not None:
            logger.info(
                f"[boundary] {label} train first-batch data-wait={first_wait * 1000:.1f}ms"
            )

    def _run_eval_window(
        eval_data_iterator, label: Optional[str] = None
    ) -> Dict[str, float]:
                                                                                 
                                                                             
                                                                              
                                                                              
                                                                             
                                                              
         
                                                                               
                                                                                 
                                                                               
                                                                      
        metric_logger.pause_perf("eval")
        _detach_train_pipeline()
        model.eval()
                                                                               
                                                                            
                                                                               
                                                                            
                                                                                
                                                                               
        metric_logger.reset(mode="eval")
        eval_batch_idx = 0
        first_wait: Optional[float] = None
        _t_enter = time.perf_counter() if (label and rank == 0) else None
        with torch.no_grad():
            while True:
                _t_next = time.perf_counter() if (label and rank == 0) else None
                try:
                    sample = next(eval_data_iterator)
                except StopIteration:
                    break
                if _t_next is not None and first_wait is None:
                    first_wait = time.perf_counter() - _t_next
                sample.to(device)
                (
                    _,
                    _,
                    _,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = _forward_sample(model, sample)
                metric_logger.update(
                    mode="eval",
                    predictions=mt_target_preds,
                    labels=mt_target_labels,
                    weights=mt_target_weights,
                    num_candidates=_batch_num_candidates(sample),
                )
                eval_batch_idx += 1
                if output_trace:
                    assert profiler is not None
                    profiler.step()
                if eval_interim_metrics and eval_batch_idx % metric_log_frequency == 0:
                    metric_logger.compute_and_log(mode="eval")
                if num_eval_batches is not None and eval_batch_idx >= num_eval_batches:
                    break
            _eval_metrics = metric_logger.compute(mode="eval")
            for k, v in _eval_metrics.items():
                print(f"{k}: {v}")
        if label and rank == 0 and _t_enter is not None:
            _eval_total = time.perf_counter() - _t_enter
            _fw = (first_wait * 1000) if first_wait is not None else float("nan")
            logger.info(
                f"[boundary] {label} eval first-batch data-wait={_fw:.1f}ms "
                f"total_eval={_eval_total * 1000:.1f}ms batches={eval_batch_idx}"
            )
                                                                              
                                                                                   
                                                                                 
                                                                                 
                                                                          
                                                                           
            import json
            import re as _re

            _log = os.environ.get("LOG")
            if _log:
                _base = _log[:-4] if _log.endswith(".log") else _log
                _metrics_path = f"{_base}.metrics.jsonl"
            else:
                _metrics_path = "streaming_eval_metrics.jsonl"
            _ts_m = _re.search(r"train_ts=(\d+)", label)
            _rec = {
                "label": label,
                "train_ts": int(_ts_m.group(1)) if _ts_m else None,
                "global_step": int(metric_logger.global_step.get("train", -1)),
                "eval_batches": eval_batch_idx,
                "total_eval_ms": round(_eval_total * 1000, 1),
                "wall_time": time.time(),
            }
            for _k, _v in _eval_metrics.items():
                try:
                    _rec[_k] = float(_v)
                except (TypeError, ValueError):
                    pass
            try:
                with open(_metrics_path, "a") as _f:
                    _f.write(json.dumps(_rec) + "\n")
            except OSError as _e:
                logger.warning("failed to write metrics sink %s: %s", _metrics_path, _e)
        metric_logger.resume_perf("eval")
                                                                                   
        return _eval_metrics

    def _maybe_checkpoint(train_ts: int) -> None:
        if (
            train_ts % checkpoint_frequency == 0 and train_ts > 0
        ) or train_ts == start_ts + num_train_ts - 1:
                                                                               
                                                                       
                                                          
            save_dmp_checkpoint(
                model=model,
                optimizer=optimizer,
                metric_logger=metric_logger,
                rank=rank,
                batch_idx=train_ts,
                train_ts=train_ts,
                batch_idx_in_window=WINDOW_COMPLETE,
                device=device,
                split_contract=live_split_contract,
            )
            last_ckpt_time[0] = time.time()

                                                                                
                                                                             
                                                                             
    train_ts_list = list(range(start_ts, start_ts + num_train_ts))
    n_train = len(train_ts_list)

    def _should_eval(i: int) -> bool:
        ''                                                                      

                                                   
                                                                                
                                                                    
                                                   
                                                                              
                                                                         
                                                                            
                                                                            
                                                                       
                                                                          
                                                          
           
        if eval_every_n_windows <= 0:
            return False
        if eval_every_n_windows == 1:
            return True
        return (train_ts_list[i] - eval_anchor_ts) % eval_every_n_windows == 0 or i == n_train - 1

    def _skip_early_window_eval(i: int) -> bool:
        ''                                                                   
                                                                                
                                                                            
                                                                 
                                                                          
                                                                            
                                                                 
           
        if skip_eval_until_step <= 0 or i == n_train - 1:
            return False
        return metric_logger.global_step["train"] < skip_eval_until_step

                                                                              
                                                                            
                                                                      
                                                                               
                                                                         
                                                                             
    eval_global_indices: Optional["np.ndarray"] = None
    if supports_holdout:
        eval_global_indices = dataset.dataset.eval_holdout_indices(                    
            eval_holdout_ts_resolved, eval_holdout_num_windows
        )
        if rank == 0:
            logger.info(
                "Fixed eval holdout: ts=[%d, %d) -> %d anchors (train_split_percentage=%s)",
                eval_holdout_ts_resolved,
                eval_holdout_ts_resolved + eval_holdout_num_windows,
                len(eval_global_indices),
                dataset.dataset._train_split_percentage,                   
            )

                                                                                
                                                                             
                                                                            
                                                                                
                     
    total_train_samples = 0
    if mlperf_logger is not None or skip_eval_epoch_pct > 0:
        if eval_interval_steps > 0:
                                                                            
                                                                            
                                                                           
                                                                            
                                                
            total_train_samples = total_train_anchors
        else:
            _idx_fn = getattr(
                dataset.dataset, "train_window_indices", None
            ) or getattr(dataset.dataset, "window_indices", None)
            if _idx_fn is not None:
                for _ts in train_ts_list:
                    total_train_samples += int(_idx_fn(_ts).size)
        if rank == 0:
            logger.info(
                "MLPerf: total_train_samples=%d over %d windows",
                total_train_samples,
                n_train,
            )
                                                                            
                                                                               
                                                                               
                                        
    skip_eval_until_step = resolve_skip_eval_until_step(
        skip_eval_epoch_pct=skip_eval_epoch_pct,
        total_train_samples=total_train_samples,
        global_batch_size=global_batch_size,
    )
    if rank == 0 and skip_eval_epoch_pct > 0:
        logger.info(
            "[skip-eval] skip_eval_epoch_pct=%.6g -> skip_eval_until_step=%d "
            "(total_train_samples=%d global_batch_size=%d -> "
            "skip_until_samples=%d)",
            skip_eval_epoch_pct,
            skip_eval_until_step,
            total_train_samples,
            global_batch_size,
            int(skip_eval_epoch_pct * total_train_samples),
        )

    if mlperf_logger is not None:
                                                                                
        metric_logger.mlperf_logger = mlperf_logger

        def _current_lr() -> float:
            return float(optimizer.param_groups[0]["lr"])

        metric_logger.lr_getter = _current_lr

                                                                                 
                                                                              
                                                                              
                                                
    from generative_recommenders.dlrm_v4.train.mlperf_logging_utils import (
        MLPerfRunTracker,
    )

    mlt = MLPerfRunTracker(
        logger=mlperf_logger,
        metric_logger=metric_logger,
        total_train_samples=total_train_samples,
        rank=rank,
        device=device,
    )
    mlt.log_dataset_sizes(
        eval_samples=eval_global_indices.size
        if eval_global_indices is not None
        else None
    )

    if persistent_loader and double_buffer:
                                                                            
                                                                             
                                                                                
        prefetcher = _PrefetchingWindowLoader(
            dataset=dataset,
            sampler_factory=lambda: StreamingWindowSampler(rank, world_size),
            dl_factory=lambda s: make_persistent_streaming_dataloader(
                dataset=dataset,
                sampler=s,
                num_workers=train_num_workers,
                prefetch_factor=train_prefetch_factor,
            ),
        )
        eval_sampler: Optional[StreamingWindowSampler] = None
        eval_dl: Optional[DataLoader] = None
                                                                                
                                                                            
                                                                                
                                                                            
                                           
        eval_iter: Optional[Iterator] = None
                                                                               
                                                                       
                                                                              
                                                         
        if (eval_every_n_windows > 0 or eval_interval_steps > 0) and len(
            train_ts_list
        ) > 0:
            eval_sampler = StreamingWindowSampler(rank, world_size)
            eval_dl = make_persistent_streaming_dataloader(
                dataset=dataset,
                sampler=eval_sampler,
                num_workers=eval_num_workers,
                prefetch_factor=eval_prefetch_factor,
            )
                                                                           
                                                                           
                                                                               
                                                                                 
                                                                                
                                                                            
                                                                                 
                                                                                 
                                                                                
                                                                              
                                                                             
            eval_sampler.set_window(eval_global_indices)
            eval_iter = iter(eval_dl)

                                                                               
                                                                            
                                                                               
                                                          
        def _do_eval_db(train_ts: int, gstep: int) -> None:
            nonlocal eval_iter
                                                                                
                                                                                  
                                                                               
                                                                               
                                                                  
            dataset.dataset.is_eval = True                    
            assert eval_dl is not None and eval_iter is not None
            mlt.block_stop()
            mlt.eval_start()
            eval_metrics = _run_eval_window(
                eval_iter,
                label=f"eval_holdout@train_ts={train_ts}@step={gstep}",
            )
                                                                         
                                                                     
            mlt.eval_stop(eval_metrics)
            if not mlt.run_stopped:
                eval_iter = iter(eval_dl)
                mlt.block_start()

        _db_do_eval = _do_eval_db if eval_interval_steps > 0 else None

                                                                         
                                                                             
                                                                             
                                                                             
                                                                        
                                         
        _per_window_blocks = eval_every_n_windows > 0
        if not _per_window_blocks:
            mlt.block_start()

        for i, (train_ts, train_data_iterator) in enumerate(
                                                                            
                                                                              
                                                                                 
                                                                         
            prefetcher.stream(train_ts_list, first_skip_samples=first_skip_samples)
        ):
            dataset.dataset.is_eval = False                    
                                                               
                                                                               
                                                                  
            start_batch = (
                resume_batch_idx_in_window
                if i == 0 and resume_batch_idx_in_window > 0
                else 0
            )
                                                                          
                                                                              
                                                                              
                                                                              
                                                                            
                                                                             
                                                                             
                                                                             
                                                                           
                                                                            
                                                                  
            _window_boundary_barrier(device, world_size, train_ts)
            if _per_window_blocks:
                mlt.block_start()
            _run_train_window(
                train_data_iterator,
                train_ts=train_ts,
                start_batch_idx=start_batch,
                label=f"train_ts={train_ts}",
                do_eval=_db_do_eval,
            )
            if _per_window_blocks:
                mlt.block_stop()
            should_stop = False
            if _should_eval(i) and not _skip_early_window_eval(i):
                dataset.dataset.is_eval = True                    
                assert eval_sampler is not None and eval_dl is not None
                mlt.eval_start()
                eval_metrics = _run_eval_window(
                    eval_iter, label=f"eval_holdout@train_ts={train_ts}"
                )
                should_stop = mlt.eval_stop(eval_metrics)
                                                                              
                                                                              
                                                                          
                                                                             
                next_eval_i = next(
                    (j for j in range(i + 1, n_train) if _should_eval(j)), None
                )
                if next_eval_i is not None:
                    eval_iter = iter(eval_dl)
            elif _should_eval(i) and rank == 0:
                logger.info(
                    "[per-window-eval] SKIP eval train_ts=%d global_step=%d "
                    "< skip_eval_until_step=%d (SKIP_EVAL_EPOCH_PCT=%.6g)",
                    train_ts,
                    metric_logger.global_step["train"],
                    skip_eval_until_step,
                    skip_eval_epoch_pct,
                )
            _maybe_checkpoint(train_ts)
                                                                   
                                                                                   
            if should_stop or mlt.run_stopped:
                                                                                 
                break

                                                                            
                                                                               
                                                                         
        if not _per_window_blocks:
            mlt.block_stop()
    else:
                                                                              
                                                                            
                                                                    
        def _do_eval_nb(train_ts: int, gstep: int) -> None:
                                                                                    
                                                                              
                                                                           
                                                                             
            dataset.dataset.is_eval = True                    
            mlt.block_stop()
            mlt.eval_start()
            if eval_global_indices is not None:
                eval_metrics = _run_eval_window(
                    iter(
                        make_streaming_dataloader(
                            dataset=dataset,
                            indices=eval_global_indices,
                            num_workers=eval_num_workers,
                            prefetch_factor=eval_prefetch_factor,
                        )
                    ),
                    label=f"eval_holdout@train_ts={train_ts}@step={gstep}",
                )
            else:
                eval_metrics = _run_eval_window(
                    iter(
                        make_streaming_dataloader(
                            dataset=dataset,
                            ts=train_ts + 1,
                            num_workers=eval_num_workers,
                            prefetch_factor=eval_prefetch_factor,
                        )
                    ),
                    label=f"eval@train_ts={train_ts}@step={gstep}",
                )
            mlt.eval_stop(eval_metrics)
            if not mlt.run_stopped:
                mlt.block_start()

        _nb_do_eval = _do_eval_nb if eval_interval_steps > 0 else None

                                                                            
                                                                               
                                    
        _per_window_blocks = eval_every_n_windows > 0
        if not _per_window_blocks:
            mlt.block_start()

        for i, train_ts in enumerate(train_ts_list):
            dataset.dataset.is_eval = False                    
            skip = first_skip_samples if i == 0 else 0
            start_batch = (
                resume_batch_idx_in_window
                if i == 0 and resume_batch_idx_in_window > 0
                else 0
            )
                                                                            
                                                                          
                                                                        
            _window_boundary_barrier(device, world_size, train_ts)
            if _per_window_blocks:
                mlt.block_start()
            _run_train_window(
                _window_iter(train_ts, skip_samples=skip),
                train_ts=train_ts,
                start_batch_idx=start_batch,
                do_eval=_nb_do_eval,
            )
            if _per_window_blocks:
                mlt.block_stop()
            should_stop = False
            if _should_eval(i) and not _skip_early_window_eval(i):
                dataset.dataset.is_eval = True                    
                mlt.eval_start()
                if eval_global_indices is not None:
                    eval_metrics = _run_eval_window(
                        iter(
                            make_streaming_dataloader(
                                dataset=dataset,
                                indices=eval_global_indices,
                                num_workers=eval_num_workers,
                                prefetch_factor=eval_prefetch_factor,
                            )
                        ),
                        label=f"eval_holdout@train_ts={train_ts}",
                    )
                else:
                                                                             
                    eval_metrics = _run_eval_window(
                        iter(
                            make_streaming_dataloader(
                                dataset=dataset,
                                ts=train_ts + 1,
                                num_workers=eval_num_workers,
                                prefetch_factor=eval_prefetch_factor,
                            )
                        )
                    )
                should_stop = mlt.eval_stop(eval_metrics)
            elif _should_eval(i) and rank == 0:
                logger.info(
                    "[per-window-eval] SKIP eval train_ts=%d global_step=%d "
                    "< skip_eval_until_step=%d (SKIP_EVAL_EPOCH_PCT=%.6g)",
                    train_ts,
                    metric_logger.global_step["train"],
                    skip_eval_until_step,
                    skip_eval_epoch_pct,
                )
            _maybe_checkpoint(train_ts)
                                                                   
                                                                                   
            if should_stop or mlt.run_stopped:
                                                                                 
                break

                                                                           
                                                                                  
        if not _per_window_blocks:
            mlt.block_stop()

                                                                          
                                                                               
    if not mlt.run_stopped:
        dataset.dataset.is_eval = True                    
        mlt.eval_start()
        if eval_global_indices is not None:
            final_metrics = _run_eval_window(
                iter(
                    make_streaming_dataloader(
                        dataset=dataset,
                        indices=eval_global_indices,
                        num_workers=eval_num_workers,
                        prefetch_factor=eval_prefetch_factor,
                    )
                ),
                label="eval_holdout@final",
            )
        else:
            final_metrics = _run_eval_window(
                iter(
                    make_streaming_dataloader(
                        dataset=dataset,
                        ts=num_train_ts,
                        num_workers=eval_num_workers,
                        prefetch_factor=eval_prefetch_factor,
                    )
                ),
                label="eval@final",
            )
        mlt.eval_stop(final_metrics)
        if rank == 0:
            for k, v in final_metrics.items():
                print(f"{k}: {v}")
                                                                                
        mlt.finalize(final_metrics)
