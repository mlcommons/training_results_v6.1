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

import logging
import math
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from megatron.bridge.data.iterator_utils import make_data_iterator_list
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel.random import _fork_rng
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import reduce_aux_losses_tracker_across_ranks
from megatron.core.transformer.moe.paged_stash import PagedStashManager
from megatron.core.utils import get_attr_wrapped_model
from modelopt.torch.distill.plugins.megatron import (
    get_tensor_shapes_adjust_fn_for_distillation,
)
from utils import get_rank, startup_timer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MockCompilePlan:
    mode: str
    train_num_microbatches: int
    train_num_samples: int
    train_micro_batch_size: int
    train_dp_rank: int
    train_dp_size: int
    validation_num_microbatches: int | None
    validation_num_samples: int
    validation_micro_batch_size: int
    validation_dp_rank: int
    validation_dp_size: int


class _CompletedWork:
    """Completed request implementing the schedule's wait/completion contract."""

    def wait(self):
        return True

    def is_completed(self):
        return True


class MockP2PCommunicator(P2PCommunicator):
    """P2P communicator that synthesizes receives instead of communicating."""

    def __init__(self, pp_group, config):
        if config.variable_seq_lengths or config.mtp_standalone:
            raise ValueError(
                "Mock pipeline compile requires fixed tensor shapes and does not "
                "support standalone MTP"
            )
        super().__init__(pp_group=pp_group, config=config)
        self._generator = None

    def _make_tensor(self, tensor_shape):
        device = torch.cuda.current_device()
        if self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(1234 + self.current_stage)
        return torch.randn(
            tuple(tensor_shape),
            device=device,
            dtype=self.config.pipeline_dtype,
            generator=self._generator,
            requires_grad=True,
        )

    def _communicate(
        self,
        *,
        tensor_send_next,
        tensor_send_prev,
        recv_prev,
        recv_next,
        tensor_shape,
        wait_on_reqs=True,
    ):
        if (recv_prev or recv_next) and self.config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided for mocked receives")
        if (recv_prev or recv_next) and tensor_shape is None:
            raise RuntimeError("tensor_shape must be provided for mocked receives")

        tensor_recv_prev = self._make_tensor(tensor_shape) if recv_prev else None
        tensor_recv_next = self._make_tensor(tensor_shape) if recv_next else None
        if wait_on_reqs:
            requests = None
        else:
            requests = {}
            if tensor_send_next is not None:
                requests["send_next"] = _CompletedWork()
            if recv_prev:
                requests["recv_prev"] = _CompletedWork()
            if tensor_send_prev is not None:
                requests["send_prev"] = _CompletedWork()
            if recv_next:
                requests["recv_next"] = _CompletedWork()
        return tensor_recv_prev, tensor_recv_next, requests


def get_reduced_num_microbatches(
    *,
    full_num_microbatches,
    num_model_chunks,
    pipeline_parallel_size,
    microbatch_group_size_per_vp_stage,
):
    """Return the smallest legal count for the configured pipeline schedule."""
    if num_model_chunks <= 1:
        return 1

    if microbatch_group_size_per_vp_stage is None:
        group_size = pipeline_parallel_size
        if group_size > full_num_microbatches:
            raise ValueError(
                f"pipeline_parallel_size ({pipeline_parallel_size}) cannot exceed "
                f"the full microbatch count ({full_num_microbatches}) for an "
                "interleaved reduced compile"
            )
    else:
        group_size = microbatch_group_size_per_vp_stage
    if not pipeline_parallel_size <= group_size <= full_num_microbatches:
        raise ValueError(
            "microbatch_group_size_per_vp_stage must be between the pipeline "
            f"parallel size ({pipeline_parallel_size}) and the full microbatch "
            f"count ({full_num_microbatches}); got {group_size}"
        )
    return group_size


def _legalize_interleaved_num_microbatches(
    base_num_microbatches,
    *,
    pipeline_parallel_size,
    microbatch_group_size,
):
    replication = max(
        1,
        math.ceil(microbatch_group_size / base_num_microbatches),
    )
    while True:
        candidate = base_num_microbatches * replication
        remainder = candidate % microbatch_group_size
        if remainder == 0 or remainder >= pipeline_parallel_size:
            return candidate
        replication += 1


def _build_mock_compile_plan(
    *,
    mode,
    validation_steps,
    model_config,
    num_model_chunks,
    pipeline_parallel_size,
    train_num_microbatches,
    train_micro_batch_size,
    train_dp_rank,
    train_dp_size,
    validation_global_batch_size,
    validation_micro_batch_size,
    validation_dp_rank,
    validation_dp_size,
):
    if mode == "off":
        return None

    if mode == "reduced":
        compile_train_num_microbatches = get_reduced_num_microbatches(
            full_num_microbatches=train_num_microbatches,
            num_model_chunks=num_model_chunks,
            pipeline_parallel_size=pipeline_parallel_size,
            microbatch_group_size_per_vp_stage=getattr(
                model_config,
                "microbatch_group_size_per_vp_stage",
                None,
            ),
        )
    else:
        compile_train_num_microbatches = train_num_microbatches

    compile_validation_num_microbatches = None
    if validation_steps > 0:
        validation_dp_product = (
            validation_micro_batch_size * validation_dp_size
        )
        base_validation_num_microbatches = max(
            1,
            math.ceil(
                validation_global_batch_size / validation_dp_product
            ),
        )
        if num_model_chunks > 1:
            microbatch_group_size = (
                getattr(
                    model_config,
                    "microbatch_group_size_per_vp_stage",
                    pipeline_parallel_size,
                )
                or pipeline_parallel_size
            )
            compile_validation_num_microbatches = (
                _legalize_interleaved_num_microbatches(
                    base_validation_num_microbatches,
                    pipeline_parallel_size=pipeline_parallel_size,
                    microbatch_group_size=microbatch_group_size,
                )
            )
        else:
            compile_validation_num_microbatches = (
                base_validation_num_microbatches
            )
    return MockCompilePlan(
        mode=mode,
        train_num_microbatches=compile_train_num_microbatches,
        train_num_samples=(
            compile_train_num_microbatches
            * train_micro_batch_size
            * train_dp_size
        ),
        train_micro_batch_size=train_micro_batch_size,
        train_dp_rank=train_dp_rank,
        train_dp_size=train_dp_size,
        validation_num_microbatches=compile_validation_num_microbatches,
        validation_num_samples=(
            (compile_validation_num_microbatches or 0)
            * validation_micro_batch_size
            * validation_dp_size
        ),
        validation_micro_batch_size=validation_micro_batch_size,
        validation_dp_rank=validation_dp_rank,
        validation_dp_size=validation_dp_size,
    )


def _prepare_forward_backward_inputs(
    context,
    data_iterator,
    model_config,
    micro_batch_size,
):
    seq_length = getattr(
        model_config,
        "seq_length",
        context.state.cfg.model.seq_length,
    )
    forward_backward_data_iterator = data_iterator
    if len(context.model) > 1:
        forward_backward_data_iterator = make_data_iterator_list(
            model=context.model,
            data_iterator=forward_backward_data_iterator,
        )

    if not context.state.cfg.dist.use_decentralized_pg:
        adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(
            context.model,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=seq_length,
        )
    else:
        adjust_tensor_shapes_fn = None

    return forward_backward_data_iterator, seq_length, adjust_tensor_shapes_fn


def _clear_grads(context):
    for model_chunk in context.model:
        model_chunk.zero_grad_buffer()
    context.optimizer.zero_grad()


def _iter_unique_modules(model):
    seen_modules = set()
    for model_chunk in model:
        for module in model_chunk.modules():
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            yield module


def _snapshot_training_modes(model):
    return [(module, module.training) for module in _iter_unique_modules(model)]


def _restore_training_modes(state):
    for module, training in state:
        module.training = training


def _snapshot_ddp_first_batch_state(model):
    state = []
    seen_bucket_groups = set()
    for model_chunk in model:
        bucket_groups = getattr(model_chunk, "bucket_groups", [])
        expert_bucket_groups = getattr(
            model_chunk,
            "expert_parallel_bucket_groups",
            [],
        )
        for bucket_group in bucket_groups + expert_bucket_groups:
            if (
                id(bucket_group) in seen_bucket_groups
                or not hasattr(bucket_group, "is_first_batch")
            ):
                continue
            seen_bucket_groups.add(id(bucket_group))
            state.append(
                (
                    bucket_group,
                    bucket_group.is_first_batch,
                    bucket_group.golden_per_param_grad_ready_counts.copy(),
                    bucket_group.per_param_grad_ready_counts.copy(),
                )
            )
    return state


def _restore_ddp_first_batch_state(state):
    for bucket_group, is_first_batch, golden_counts, ready_counts in state:
        bucket_group.is_first_batch = is_first_batch
        bucket_group.golden_per_param_grad_ready_counts = golden_counts
        bucket_group.per_param_grad_ready_counts = ready_counts


def _reset_token_dispatcher_state(model):
    """Discard HybridEP overflow flags accumulated by the synthetic pass."""
    seen_dispatchers = set()
    for module in _iter_unique_modules(model):
        token_dispatcher = getattr(module, "token_dispatcher", None)
        reset_over_budget = getattr(token_dispatcher, "reset_over_budget", None)
        if not callable(reset_over_budget) or id(token_dispatcher) in seen_dispatchers:
            continue
        seen_dispatchers.add(id(token_dispatcher))
        reset_over_budget()


def _snapshot_router_state(model):
    state = []
    for module in _iter_unique_modules(model):
        expert_bias = getattr(module, "expert_bias", None)
        if expert_bias is None:
            continue
        local_tokens = getattr(module, "local_tokens_per_expert", None)
        state.append(
            (
                module,
                expert_bias.detach().clone(),
                local_tokens.detach().clone() if local_tokens is not None else None,
            )
        )
    return state


def _restore_router_state(state):
    with torch.no_grad():
        for router, expert_bias, local_tokens in state:
            router.expert_bias.copy_(expert_bias)
            if local_tokens is not None:
                router.local_tokens_per_expert.copy_(local_tokens)


def _clear_pipeline_inputs(model):
    for model_chunk in model:
        set_input_tensor = get_attr_wrapped_model(model_chunk, "set_input_tensor")
        set_input_tensor(None)


@contextmanager
def preserve_mock_rng_state():
    """Prevent mock iterators and kernels from advancing application RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with _fork_rng():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def run_mock_pipeline_compile_pass(
    *,
    label,
    context,
    data_iterator,
    forward_step_func,
    forward_backward_func,
    forward_only,
    pg_collection,
    model_config,
    micro_batch_size,
    num_microbatches=None,
):
    """Compile JITs by driving the real pipeline schedule with mocked P2P.

    ``num_microbatches`` defaults to the full ``get_num_microbatches()`` count.
    A reduced count compiles common schedule and model kernels at lower cost, but
    does not guarantee full 1F1B or overlap-path coverage. Mock passes keep
    paged stash enabled for its capture-phase memory savings, but use a
    temporary manager so synthetic inputs cannot advance the real manager's
    persistent lifecycle or size its long-lived buffers.
    """
    mock_p2p_communicator = MockP2PCommunicator(
        pp_group=pg_collection.pp,
        config=model_config,
    )
    forward_backward_data_iterator, seq_length, adjust_tensor_shapes_fn = (
        _prepare_forward_backward_inputs(
            context,
            data_iterator,
            model_config,
            micro_batch_size,
        )
    )
    effective_num_microbatches = (
        get_num_microbatches() if num_microbatches is None else num_microbatches
    )
    paged_stash_context = _temporarily_isolate_paged_stash_manager(model_config)
    original_training_states = (
        _snapshot_training_modes(context.model) if forward_only else []
    )
    # Keep forward-only passes state-neutral even if router implementations
    # update running bias/count buffers under no_grad in a future MCore version.
    router_state = _snapshot_router_state(context.model)
    ddp_first_batch_state = _snapshot_ddp_first_batch_state(context.model)

    with startup_timer(label, microbatches=effective_num_microbatches):
        if get_rank() == 0:
            logger.info(f"Starting {label} ({effective_num_microbatches} microbatches)")
        if forward_only:
            for model_module in context.model:
                model_module.eval()
        try:
            _clear_grads(context)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            grad_context = torch.no_grad() if forward_only else nullcontext()
            with paged_stash_context, grad_context:
                forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=forward_backward_data_iterator,
                    model=context.model,
                    num_microbatches=effective_num_microbatches,
                    seq_length=seq_length,
                    micro_batch_size=micro_batch_size,
                    decoder_seq_length=seq_length,
                    forward_only=forward_only,
                    adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
                    p2p_communicator=mock_p2p_communicator,
                    pg_collection=pg_collection,
                )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            if not forward_only:
                reduce_aux_losses_tracker_across_ranks(pg_collection=pg_collection)
        finally:
            get_moe_metrics_tracker().clear()
            _clear_grads(context)
            _restore_ddp_first_batch_state(ddp_first_batch_state)
            _reset_token_dispatcher_state(context.model)
            _restore_router_state(router_state)
            _clear_pipeline_inputs(context.model)
            if forward_only:
                _restore_training_modes(original_training_states)
        if get_rank() == 0:
            logger.info(f"Finished {label}")


def run_mock_compile(
    context,
    *,
    mode,
    validation_steps,
    data_iterator_builder,
    prepared_forward_step_func,
    raw_forward_backward_func,
    pg_collection,
    model_config,
    validation_dp_rank,
    validation_dp_size,
    validation_context=None,
):
    validation_global_batch_size = validation_micro_batch_size = 0
    if validation_steps > 0:
        validation_global_batch_size = (
            context.state.cfg.validation.eval_global_batch_size
        )
        validation_micro_batch_size = (
            context.state.cfg.validation.eval_micro_batch_size
        )
    plan = _build_mock_compile_plan(
        mode=mode,
        validation_steps=validation_steps,
        model_config=model_config,
        num_model_chunks=len(context.model),
        pipeline_parallel_size=pg_collection.pp.size(),
        train_num_microbatches=get_num_microbatches(),
        train_micro_batch_size=context.state.cfg.train.micro_batch_size,
        train_dp_rank=parallel_state.get_data_parallel_rank(),
        train_dp_size=parallel_state.get_data_parallel_world_size(),
        validation_global_batch_size=validation_global_batch_size,
        validation_micro_batch_size=validation_micro_batch_size,
        validation_dp_rank=validation_dp_rank,
        validation_dp_size=validation_dp_size,
    )
    if plan is None:
        return

    with preserve_mock_rng_state():
        train_iterator = data_iterator_builder(
            num_samples=plan.train_num_samples,
            dp_rank=plan.train_dp_rank,
            dp_size_=plan.train_dp_size,
            iterator_micro_batch_size=plan.train_micro_batch_size,
        )
        run_mock_pipeline_compile_pass(
            label="warmup_mock_train_compile",
            context=context,
            data_iterator=train_iterator,
            forward_step_func=prepared_forward_step_func,
            forward_backward_func=raw_forward_backward_func,
            forward_only=False,
            num_microbatches=plan.train_num_microbatches,
            micro_batch_size=plan.train_micro_batch_size,
            pg_collection=pg_collection,
            model_config=model_config,
        )

        if plan.validation_num_microbatches is None:
            return
        validation_iterator = data_iterator_builder(
            num_samples=plan.validation_num_samples,
            dp_rank=plan.validation_dp_rank,
            dp_size_=plan.validation_dp_size,
            iterator_micro_batch_size=plan.validation_micro_batch_size,
        )
        eval_context = (
            validation_context if validation_context is not None else nullcontext()
        )
        with eval_context:
            run_mock_pipeline_compile_pass(
                label="warmup_mock_validation_compile",
                context=context,
                data_iterator=validation_iterator,
                forward_step_func=prepared_forward_step_func,
                forward_backward_func=raw_forward_backward_func,
                forward_only=True,
                num_microbatches=plan.validation_num_microbatches,
                micro_batch_size=plan.validation_micro_batch_size,
                pg_collection=pg_collection,
                model_config=model_config,
            )


@contextmanager
def _temporarily_isolate_paged_stash_manager(model_config):
    """Run a mock pass with paged stash enabled on a disposable manager."""
    if not getattr(model_config, "moe_paged_stash", False):
        yield
        return

    original_manager = PagedStashManager.STASH_MGR
    mock_manager = PagedStashManager()
    original_status = getattr(original_manager, "status", None)
    if get_rank() == 0:
        logger.info(
            "Mock paged-stash isolation enter: real_status=%s mock_status=%s",
            original_status,
            mock_manager.status,
        )
    PagedStashManager.STASH_MGR = mock_manager
    try:
        yield
    finally:
        PagedStashManager.STASH_MGR = original_manager
        mock_status = getattr(mock_manager, "status", None)
        if get_rank() == 0:
            logger.info(
                "Mock paged-stash isolation exit: mock_status=%s restored_real_status=%s",
                mock_status,
                getattr(original_manager, "status", None),
            )
