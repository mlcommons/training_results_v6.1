# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import logging
import math
from typing import Any, Callable

import torch
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core.utils import get_model_config

logger = logging.getLogger(__name__)


def _get_effective_eval_dp(state: Any, pg_collection: Any) -> tuple[int, int]:
    ''                                                                            
    from callback_eval_cp import EvalCPCallback

    eval_cp_callback = EvalCPCallback.get_active()
    if eval_cp_callback is not None and eval_cp_callback.eval_dp_group is not None:
        eval_dp_group = eval_cp_callback.eval_dp_group
        return (
            torch.distributed.get_world_size(eval_dp_group),
            torch.distributed.get_rank(eval_dp_group),
        )
    return state.cfg.data_parallel_size, pg_collection.dp.rank()


class EvalMinimalSchedulePaddingCallback(Callback):
    ''                                                                        

    _validate_patched = False
    _evaluate_patched = False

    def __init__(self) -> None:
        if not EvalMinimalSchedulePaddingCallback._validate_patched:
            self._patch_config_validate()
            EvalMinimalSchedulePaddingCallback._validate_patched = True
        if not EvalMinimalSchedulePaddingCallback._evaluate_patched:
            self._patch_evaluate()
            EvalMinimalSchedulePaddingCallback._evaluate_patched = True

    @staticmethod
    def _patch_config_validate() -> None:
        ''                                                                                    
        from megatron.bridge.training import config as config_module

        original = config_module.ConfigContainer.validate

        @functools.wraps(original)
        def patched(config) -> None:
            configured_eval_gbs = config.validation.eval_global_batch_size
            configured_eval_mbs = config.validation.eval_micro_batch_size
            eval_gbs = configured_eval_gbs or config.train.global_batch_size
            eval_mbs = configured_eval_mbs or config.train.micro_batch_size

                                                                                          
                                                                                           
                                                                                     
            validation_gbs = (
                eval_mbs
                * config.data_parallel_size
                * (config.model.context_parallel_size or 1)
            )
            if eval_gbs != validation_gbs:
                logger.info(
                    "[EvalMinimalSchedulePaddingCallback] validating configured "
                    "eval_global_batch_size=%s with temporary divisible value=%s; "
                    "runtime masking preserves the configured sample count",
                    eval_gbs,
                    validation_gbs,
                )
            config.validation.eval_micro_batch_size = eval_mbs
            config.validation.eval_global_batch_size = validation_gbs
            try:
                original(config)
            finally:
                                                                                
                config.validation.eval_micro_batch_size = eval_mbs
                config.validation.eval_global_batch_size = eval_gbs

        config_module.ConfigContainer.validate = patched

    @staticmethod
    def _patch_evaluate() -> None:
        ''                                                                         
        from megatron.bridge.training import eval as eval_module

        original = eval_module.evaluate
        signature = inspect.signature(original)

        @functools.wraps(original)
        def patched(*args: Any, **kwargs: Any):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            state = bound.arguments["state"]
            model = bound.arguments["model"]
            data_iterator = bound.arguments["data_iterator"]
            pg_collection = bound.arguments["pg_collection"] or get_pg_collection(model)
            model_config = get_model_config(model[0])

            eval_gbs = state.cfg.validation.eval_global_batch_size
            eval_mbs = state.cfg.validation.eval_micro_batch_size
            if eval_mbs is None:
                eval_mbs = state.cfg.train.micro_batch_size
            if eval_mbs is None or eval_mbs <= 0:
                raise RuntimeError(
                    "EvalMinimalSchedulePaddingCallback requires a positive "
                    "eval_micro_batch_size or train micro_batch_size"
                )
            pp_size = pg_collection.pp.size()
                                                                            
                                                                        
                                                                         
                                                                       
            eval_dp, eval_dp_rank = _get_effective_eval_dp(state, pg_collection)

            natural_microbatches = max(1, math.ceil(eval_gbs / (eval_mbs * eval_dp)))
            scheduled_microbatches, eval_group_size = _minimal_legal_eval_schedule(
                natural_microbatches, pp_size
            )
            natural_slots = natural_microbatches * eval_mbs * eval_dp
            scheduled_gbs = scheduled_microbatches * eval_mbs * eval_dp
            padding_microbatches = scheduled_microbatches - natural_microbatches
            partial_wave_padding_slots = natural_slots - eval_gbs
            temporal_padding_slots = padding_microbatches * eval_mbs * eval_dp
            extra_work_pct = 100.0 * padding_microbatches / natural_microbatches

            if torch.distributed.get_rank() == 0:
                logger.info(
                    "[EvalMinimalSchedulePaddingCallback] "
                    f"eval_global_batch_size={eval_gbs}, eval_micro_batch_size={eval_mbs}, "
                    f"data_parallel_size={eval_dp}, pp_size={pp_size}, "
                    f"microbatch_group_size={eval_group_size}, "
                    f"natural_microbatches={natural_microbatches}, "
                    f"scheduled_microbatches={scheduled_microbatches}, "
                    f"zero_weight_padding_microbatches={padding_microbatches}, "
                    f"configured_samples={eval_gbs}, "
                    f"natural_slots={natural_slots}, "
                    f"partial_wave_padding_slots={partial_wave_padding_slots}, "
                    f"temporal_padding_slots={temporal_padding_slots}, "
                    f"scheduled_slots={scheduled_gbs}, "
                    f"extra_schedule_work_vs_natural={extra_work_pct:.1f}%"
                )

            bound.arguments["data_iterator"] = _wrap_eval_iterator(
                data_iterator,
                natural_microbatches=natural_microbatches,
                scheduled_microbatches=scheduled_microbatches,
                eval_gbs=eval_gbs,
                eval_mbs=eval_mbs,
                eval_dp=eval_dp,
                eval_dp_rank=eval_dp_rank,
            )

            consumed_before = state.train_state.consumed_valid_samples
            train_group_size = model_config.microbatch_group_size_per_vp_stage
            state.cfg.validation.eval_global_batch_size = scheduled_gbs
            model_config.microbatch_group_size_per_vp_stage = eval_group_size
            try:
                return original(*bound.args, **bound.kwargs)
            finally:
                                                                                               
                                                                                              
                                                                                   
                consumed_delta = state.train_state.consumed_valid_samples - consumed_before
                completed_iters = consumed_delta // scheduled_gbs
                state.train_state.consumed_valid_samples = (
                    consumed_before + completed_iters * eval_gbs
                )
                model_config.microbatch_group_size_per_vp_stage = train_group_size
                state.cfg.validation.eval_global_batch_size = eval_gbs

        eval_module.evaluate = patched


def _minimal_legal_eval_schedule(
    natural_microbatches: int, pp_size: int
) -> tuple[int, int]:
    ''                                                                         
    if natural_microbatches < 1:
        raise ValueError("natural_microbatches must be positive")
    if pp_size < 1:
        raise ValueError("pp_size must be positive")
    if natural_microbatches >= pp_size:
                                                                              
                                                  
        return natural_microbatches, natural_microbatches
    return pp_size, pp_size


def _map_loss_masks(
    batch: Any, transform: Callable[[Any], Any], operation: str
) -> tuple[Any, bool]:
    ''                                                                        
    if isinstance(batch, dict):
        loss_mask = batch.get("loss_mask")
        if loss_mask is None:
            raise ValueError(f"{operation} requires a non-null loss_mask in the eval batch")
        batch = dict(batch)
        batch["loss_mask"] = transform(loss_mask)
        return batch, True
    if isinstance(batch, (list, tuple)):
        transformed_items = [_map_loss_masks(item, transform, operation) for item in batch]
        transformed_batch = type(batch)(item for item, _ in transformed_items)
        return transformed_batch, any(found for _, found in transformed_items)
    return batch, False


def _require_mapped_loss_mask(
    batch: Any, transform: Callable[[Any], Any], operation: str
) -> Any:
    batch, found = _map_loss_masks(batch, transform, operation)
    if not found:
        raise ValueError(f"{operation} requires a non-null loss_mask in the eval batch")
    return batch


def _zero_loss_mask(batch: Any) -> Any:
    return _require_mapped_loss_mask(
        batch, torch.zeros_like, "zero-weight eval schedule padding"
    )


def _mask_loss_mask_rows(batch: Any, real_rows: int, eval_mbs: int) -> Any:
    ''                                                                        
    if real_rows >= eval_mbs:
        return batch
    if real_rows <= 0:
        return _zero_loss_mask(batch)

    def mask_partial_rows(loss_mask):
        if loss_mask.ndim < 2 or loss_mask.shape[0] != eval_mbs:
            raise ValueError(
                "partial eval-DP masking requires loss_mask's first dimension "
                f"to equal eval_micro_batch_size ({eval_mbs}); got {tuple(loss_mask.shape)}"
            )
        loss_mask = loss_mask.clone()
        loss_mask[real_rows:] = 0
        return loss_mask

    return _require_mapped_loss_mask(
        batch, mask_partial_rows, "partial eval-DP masking"
    )


def _real_rows_in_partial_wave(
    eval_gbs: int,
    eval_mbs: int,
    eval_dp: int,
    eval_dp_rank: int,
    natural_microbatches: int,
) -> int:
    ''                                                                              
    final_wave_samples = eval_gbs - (natural_microbatches - 1) * eval_mbs * eval_dp
    return sum(
        row * eval_dp + eval_dp_rank < final_wave_samples
        for row in range(eval_mbs)
    )


def _wrap_eval_iterator(
    data_iterator: Any,
    *,
    natural_microbatches: int,
    scheduled_microbatches: int,
    eval_gbs: int,
    eval_mbs: int,
    eval_dp: int,
    eval_dp_rank: int,
) -> Any:
    if data_iterator is None:
        return data_iterator
    if isinstance(data_iterator, list):
        return [
            _wrap_eval_iterator(
                iterator,
                natural_microbatches=natural_microbatches,
                scheduled_microbatches=scheduled_microbatches,
                eval_gbs=eval_gbs,
                eval_mbs=eval_mbs,
                eval_dp=eval_dp,
                eval_dp_rank=eval_dp_rank,
            )
            for iterator in data_iterator
        ]

    def generator():
        index = 0
        padding_template = None
        real_rows_in_final_wave = _real_rows_in_partial_wave(
            eval_gbs, eval_mbs, eval_dp, eval_dp_rank, natural_microbatches
        )
        while True:
            schedule_index = index % scheduled_microbatches
            index += 1
            if schedule_index >= natural_microbatches:
                if padding_template is None:
                    raise RuntimeError("temporal eval padding requires a cached real batch")
                                                                                            
                                                                                           
                yield _zero_loss_mask(padding_template)
                continue

            batch = next(data_iterator)
            if padding_template is None:
                padding_template = batch
            if schedule_index == natural_microbatches - 1:
                batch = _mask_loss_mask_rows(batch, real_rows_in_final_wave, eval_mbs)
            yield batch

    return generator()
