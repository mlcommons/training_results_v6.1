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
from typing import Any

import torch
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core.utils import get_model_config

logger = logging.getLogger(__name__)


class EvalMinimalSchedulePaddingCallback(Callback):
    """Run eval with the minimal legal PP schedule and zero-weight padding."""

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
        """Make the stock eval divisibility check harmless while this feature is active."""
        from megatron.bridge.training import config as config_module

        original = config_module.ConfigContainer.validate

        @functools.wraps(original)
        def patched(config) -> None:
            configured_eval_gbs = config.validation.eval_global_batch_size
            configured_eval_mbs = config.validation.eval_micro_batch_size
            eval_gbs = configured_eval_gbs or config.train.global_batch_size
            eval_mbs = configured_eval_mbs or config.train.micro_batch_size

            # This is divisible by train DP and by any eval DP obtained by
            # repartitioning train DP*CP through EvalCPCallback.
            validation_gbs = (
                eval_mbs
                * config.data_parallel_size
                * (config.model.context_parallel_size or 1)
            )
            config.validation.eval_micro_batch_size = eval_mbs
            config.validation.eval_global_batch_size = validation_gbs
            try:
                original(config)
            finally:
                # Preserve the defaults that stock validate() would materialize.
                config.validation.eval_micro_batch_size = eval_mbs
                config.validation.eval_global_batch_size = eval_gbs

        config_module.ConfigContainer.validate = patched

    @staticmethod
    def _patch_evaluate() -> None:
        """Wrap mBridge evaluate(), including WarmupCallback's direct call path."""
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
            eval_dp = state.cfg.data_parallel_size
            pp_size = pg_collection.pp.size()
            eval_group_size = pp_size

            natural_microbatches = max(1, math.ceil(eval_gbs / (eval_mbs * eval_dp)))
            scheduled_microbatches = _minimal_legal_microbatches(
                natural_microbatches, eval_group_size, pp_size
            )
            scheduled_gbs = scheduled_microbatches * eval_mbs * eval_dp
            padding_microbatches = scheduled_microbatches - natural_microbatches
            padding_slots = padding_microbatches * eval_mbs * eval_dp
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
                    f"natural_slots={natural_microbatches * eval_mbs * eval_dp}, "
                    f"scheduled_slots={scheduled_gbs}, "
                    f"zero_weight_padding_slots={padding_slots}, "
                    f"extra_schedule_work_vs_natural={extra_work_pct:.1f}%"
                )

            bound.arguments["data_iterator"] = _mask_padded_microbatches(
                data_iterator, natural_microbatches, scheduled_microbatches
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


def _minimal_legal_microbatches(
    natural_microbatches: int, group_size: int, pipeline_parallel_size: int
) -> int:
    """Return the smallest microbatch count accepted by the interleaved schedule."""
    scheduled = max(group_size, natural_microbatches)
    remainder = scheduled % group_size
    if remainder == 0 or remainder >= pipeline_parallel_size:
        return scheduled
    return scheduled + pipeline_parallel_size - remainder


def _zero_loss_mask(batch: Any) -> Any:
    if isinstance(batch, dict):
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            batch = dict(batch)
            batch["loss_mask"] = torch.zeros_like(loss_mask)
        return batch
    if isinstance(batch, (list, tuple)):
        return type(batch)(_zero_loss_mask(item) for item in batch)
    return batch


def _mask_padded_microbatches(
    data_iterator: Any, natural_microbatches: int, scheduled_microbatches: int
) -> Any:
    if data_iterator is None or natural_microbatches >= scheduled_microbatches:
        return data_iterator
    if isinstance(data_iterator, list):
        return [
            _mask_padded_microbatches(iterator, natural_microbatches, scheduled_microbatches)
            for iterator in data_iterator
        ]

    def generator():
        index = 0
        while True:
            batch = next(data_iterator)
            if index % scheduled_microbatches >= natural_microbatches:
                batch = _zero_loss_mask(batch)
            index += 1
            yield batch

    return generator()
