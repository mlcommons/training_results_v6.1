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
import time
from contextlib import nullcontext
from pprint import pprint

import torch
from callback_eval_cp import EvalCPCallback
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.training import eval as eval_module
from megatron.bridge.training import train as train_module
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import prepare_forward_step_func
from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph
from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import reduce_aux_losses_tracker_across_ranks
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import MockGPTDataset
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.utils import get_model_config

try:
    from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
except ImportError:
    OptimizerCudaGraphWrapper = None  # type: ignore[misc, assignment]
from utils import get_rank

logger = logging.getLogger(__name__)


def _reset_param_gather_dispatched(model):
    for model_chunk in model:
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            bucket_group.param_gather_dispatched = False


def _copy_main_params_to_param_buffer(optimizer):
    for optim_instance in optimizer.chained_optimizers:
        if isinstance(optim_instance, DistributedOptimizer):
            optim_instance._copy_main_params_to_param_buffer()


def _zero_and_copy_main_params_to_param_buffer(model, optimizer):
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    _copy_main_params_to_param_buffer(optimizer)


def _handle_mxfp8_param_buffer_copy_skip_training_cg(
    optimizer,
    model,
    reuse_grad_buf_for_mxfp8_param_ag,
    overlap_param_gather,
):
    """Replaces train_module._handle_mxfp8_param_buffer_copy.

    When skip_training_cg_param_ag is on, the param AG is omitted at CG
    capture, so the buffer doesn't need re-populating once the graph is
    captured — eval's force-sync keeps param.data correct. Behavior
    matches upstream when the flag is off.
    """
    if not (reuse_grad_buf_for_mxfp8_param_ag and overlap_param_gather):
        return

    forward_pre_hook_enabled = len(model[0].remove_forward_pre_hook_handles) > 0
    full_cg_captured = FullCudaGraphWrapper.cuda_graph.get("training") is not None
    model_config = get_model_config(model[0])
    skip_training_cg_param_ag = bool(
        getattr(model_config, "skip_training_cg_param_ag", False)
    )

    if skip_training_cg_param_ag:
        should_copy_main_params = forward_pre_hook_enabled and not full_cg_captured
    else:
        should_copy_main_params = forward_pre_hook_enabled or full_cg_captured

    if should_copy_main_params:
        _copy_main_params_to_param_buffer(optimizer)


_handle_mxfp8_param_buffer_copy_skip_training_cg._skip_training_cg_param_ag_patch = True
if not getattr(
    train_module._handle_mxfp8_param_buffer_copy,
    "_skip_training_cg_param_ag_patch",
    False,
):
    train_module._handle_mxfp8_param_buffer_copy = (
        _handle_mxfp8_param_buffer_copy_skip_training_cg
    )


# Skip deletion for Bridge's uncaptured TE helper; WarmupCallback owns the captured graphs.
_orig_te_delete_cuda_graphs = TECudaGraphHelper.delete_cuda_graphs


def _guarded_te_delete_cuda_graphs(self):
    if not self._graphs_created:
        return
    return _orig_te_delete_cuda_graphs(self)


TECudaGraphHelper.delete_cuda_graphs = _guarded_te_delete_cuda_graphs


def create_mock_dataset_config(config):
    from megatron.bridge.training.config import MockGPTDatasetConfig

    vocab_size = 129280  # DeepSeek-V3 tokenizer vocab size
    tokenizer = build_tokenizer(
        config=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size),
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )

    mock_config = MockGPTDatasetConfig(
        seq_length=config.model.encoder_seq_length,
        random_seed=config.model.seed,
        dataloader_type="single",
        num_workers=config.data.num_workers,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        path_to_cache=None,
        split="900,50,50",
        data_sharding=True,
        create_attention_mask=False,
    )
    mock_config.tokenizer = tokenizer

    return mock_config


def capture_te_cuda_graph(context, model_config):
    """Capture TE CUDA graphs.

    The caller is responsible for managing forward pre-hook and param_sync_func
    state around this call — hooks should be disabled and param_sync_func should
    be None so that collectives are not baked into the captured graph.

    Returns the TECudaGraphHelper so the caller can set manual hooks after
    re-enabling forward pre-hooks.
    """
    cuda_graph_helper = TECudaGraphHelper(
        model=context.model,
        config=model_config,
        seq_length=context.state.cfg.model.seq_length,
        micro_batch_size=context.state.cfg.train.micro_batch_size,
        optimizers=[context.optimizer],
    )

    cuda_graph_helper.create_cudagraphs()

    # Prevent train.py from ever re-capturing
    model_config.cuda_graph_warmup_steps = -1
    return cuda_graph_helper


class WarmupCallback(Callback):
    def __init__(self, cfg, forward_step_func=None):
        self.cfg = cfg
        self.forward_step_func = forward_step_func
        self.train_steps = cfg.trainer.warmup_train_steps
        self.val_steps = cfg.trainer.warmup_validation_steps
        self.cg_warmup_steps = cfg.model.cuda_graph_warmup_steps
        self.optimizer_cuda_graph = cfg.optim.optimizer_cuda_graph
        self.cuda_graph_helper = None

    def on_data_init_start(self, context: CallbackContext):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if get_rank() == 0:
            print("\nMCore config:", flush=True)
            pprint(context.model[0].config)

        torch.distributed.barrier()
        pg_collection = get_pg_collection(context.model)
        model_config = get_model_config(context.model[0])

        forward_backward_func = get_forward_backward_func(
            pp_size=pg_collection.pp.size(),
            vp_size=context.state.cfg.model.virtual_pipeline_model_parallel_size,
        )

        if is_full_iteration_cuda_graph(model_config):
            forward_backward_func = FullCudaGraphWrapper(
                forward_backward_func,
                cuda_graph_warmup_steps=self.cg_warmup_steps,
            )
            skip_training_cg_param_ag = (
                bool(self.cfg.optim.skip_training_cg_param_ag)
                and context.state.cfg.validation.eval_interval == 1
            )
        else:
            skip_training_cg_param_ag = False
        model_config.skip_training_cg_param_ag = skip_training_cg_param_ag

        prepared_forward_step_func = prepare_forward_step_func(self.forward_step_func, context.state)

        mock_config = create_mock_dataset_config(self.cfg)
        mock_config.finalize()

        micro_batch_size = self.cfg.model.micro_batch_size
        num_microbatches = get_num_microbatches()
        dp_size = parallel_state.get_data_parallel_world_size()

        train_samples = self.train_steps * num_microbatches * micro_batch_size * dp_size
        # Validation runs at the eval-DP topology when EvalCPCallback is active
        # (otherwise train DP). Sample count and sampler indices must match.
        eval_cp_cb = EvalCPCallback.get_active()
        if eval_cp_cb is not None and eval_cp_cb.eval_dp_group is not None:
            val_dp_rank = torch.distributed.get_rank(eval_cp_cb.eval_dp_group)
            val_dp_size = torch.distributed.get_world_size(eval_cp_cb.eval_dp_group)
        else:
            val_dp_rank = parallel_state.get_data_parallel_rank()
            val_dp_size = dp_size
        # Use training num_microbatches for validation dataset sizing: the pipeline
        # scheduler consumes data at the training rate regardless of eval_batch_size.
        val_samples = self.val_steps * num_microbatches * micro_batch_size * val_dp_size

        def _build_iterator(num_samples, dp_rank, dp_size_):
            ds, _, _ = BlendedMegatronDatasetBuilder(
                MockGPTDataset,
                [num_samples, 0, 0],
                lambda: True,  # is_dataset_built_on_rank - always True for mock
                mock_config,
            ).build()
            return iter(build_pretraining_data_loader(
                dataset=ds,
                consumed_samples=0,
                dataloader_type=mock_config.dataloader_type or "single",
                micro_batch_size=micro_batch_size,
                num_workers=mock_config.num_workers,
                data_sharding=mock_config.data_sharding,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size_,
            ))

        data_iterator = _build_iterator(train_samples, parallel_state.get_data_parallel_rank(), dp_size)
        val_data_iterator = (
            _build_iterator(val_samples, val_dp_rank, val_dp_size) if self.val_steps > 0 else None
        )

        pp_group = pg_collection.pp
        p2p_communicator = P2PCommunicator(pp_group=pp_group, config=model_config)
        torch.distributed.barrier(pp_group)

        # Mirror train.py: disable forward pre-hooks during warmup to prevent
        # stale data from propagating through the first all-gather.
        should_toggle_fwd_pre_hook = train_module.should_disable_forward_pre_hook(
            context.state.cfg.ddp.use_megatron_fsdp,
            context.state.cfg.optimizer.use_distributed_optimizer,
            context.state.cfg.ddp.overlap_param_gather,
        )
        if should_toggle_fwd_pre_hook:
            train_module.disable_forward_pre_hook(context.model, param_sync=False)
            param_sync_func = model_config.param_sync_func
            model_config.param_sync_func = None

        for group in context.optimizer.param_groups:
            group["betas_"] = group["betas"]
            group["bias_correction_"] = group["bias_correction"]
            group["lr_"] = group["lr"]
            group["weight_decay_"] = group["weight_decay"]
            group["betas"] = [1.0, 1.0]
            group["bias_correction"] = False
            group["lr"] = 0.0 if not self.optimizer_cuda_graph else torch.zeros_like(group["lr_"])
            group["weight_decay"] = 0.0

        scheduler_state = context.scheduler.state_dict()

        # Defer optimizer graph capture until after this callback's extra optimizer steps.
        if OptimizerCudaGraphWrapper is not None and isinstance(
            context.optimizer.step, OptimizerCudaGraphWrapper
        ):
            wrapped_optimizer_step = context.optimizer.step
            prev_cuda_graph_warmup_steps = wrapped_optimizer_step.cuda_graph_warmup_steps
            wrapped_optimizer_step.cuda_graph_warmup_steps = max(
                prev_cuda_graph_warmup_steps,
                self.train_steps + 1,
            )
            if torch.distributed.get_rank() == 0:
                logger.info(
                    f"Extended optimizer CUDA graph warmup steps from "
                    f"{prev_cuda_graph_warmup_steps} to "
                    f"{wrapped_optimizer_step.cuda_graph_warmup_steps}"
                )

        if torch.distributed.get_rank() == 0:
            logger.info("Starting training warmup")
        start = time.time()
        for step_idx in range(self.train_steps):
            if torch.distributed.get_rank() == 0:
                logger.info(f"    Starting warmup step {step_idx}")
                step_timer = time.time()

            capture_training_cg_this_step = (
                skip_training_cg_param_ag
                and should_toggle_fwd_pre_hook
                and FullCudaGraphWrapper.cuda_graph.get("training") is None
                and FullCudaGraphWrapper.curr_iteration.get("training", 0) == self.cg_warmup_steps
            )
            disabled_forward_pre_hook_for_capture = False
            if capture_training_cg_this_step:
                # Disable the param AG so it isn't baked into the captured
                # graph — eval's AG'ed weights are reused at replay.
                if len(context.model[0].remove_forward_pre_hook_handles) > 0:
                    train_module.disable_forward_pre_hook(context.model, param_sync=False)
                    disabled_forward_pre_hook_for_capture = True
                model_config.param_sync_func = None

            torch.cuda.synchronize()
            torch.distributed.barrier()
            train_module.train_step(
                forward_step_func=prepared_forward_step_func,
                data_iterator=data_iterator,
                model=context.model,
                optimizer=context.optimizer,
                scheduler=context.scheduler,
                global_state=context.state,
                pg_collection=pg_collection,
                forward_backward_func=forward_backward_func,
                p2p_communicator=p2p_communicator,
            )
            # Initialize NCCL comms for MoE aux-loss reduce groups, same as training_log does.
            reduce_aux_losses_tracker_across_ranks(pg_collection=pg_collection)

            if step_idx == 0:
                # Capture TE graphs before enabling hooks, with param_sync_func=None.
                if context.state.cfg.model.cuda_graph_impl == "transformer_engine":
                    self.cuda_graph_helper = capture_te_cuda_graph(context, model_config)
                # Enable forward pre-hooks after the first warmup step, matching
                # the normal train.py behavior. Local full-iteration CUDA graph
                # capture disables parameter gather only for the capture step.
                if should_toggle_fwd_pre_hook:
                    train_module.enable_forward_pre_hook(context.model)
                    model_config.param_sync_func = param_sync_func
                    if self.cuda_graph_helper is not None:
                        self.cuda_graph_helper.cuda_graph_set_manual_hooks()

            if capture_training_cg_this_step:
                if disabled_forward_pre_hook_for_capture:
                    train_module.enable_forward_pre_hook(context.model)
                model_config.param_sync_func = param_sync_func

            torch.cuda.synchronize()
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                logger.info(f"    Finished warmup step {step_idx}, takes {time.time() - step_timer} s")

        if hasattr(context.scheduler, "num_steps"):
            context.scheduler.num_steps = 0
        context.scheduler.load_state_dict(scheduler_state)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            logger.info(f"Finished training warmup: {time.time() - start} s. ")

        for group in context.optimizer.param_groups:
            group["betas"] = group["betas_"]
            group["bias_correction"] = group["bias_correction_"]
            group["lr"] = group["lr_"]
            group["weight_decay"] = group["weight_decay_"]
            del group["betas_"]
            del group["bias_correction_"]
            del group["lr_"]
            del group["weight_decay_"]
            if "step" in group:
                # MegatronFusedAdam
                if isinstance(group["step"], torch.Tensor):
                    group["step"].fill_(1)
                else:
                    group["step"] = 1

        # If no training steps ran, hooks are still disabled — enable now.
        if should_toggle_fwd_pre_hook and self.train_steps == 0:
            train_module.enable_forward_pre_hook(context.model)
            model_config.param_sync_func = param_sync_func

        if self.val_steps > 0:
            start_val_time = time.time()
            if torch.distributed.get_rank() == 0:
                logger.info("Starting validation warmups")
            original_eval_iters = context.state.cfg.validation.eval_iters
            context.state.cfg.validation.eval_iters = self.val_steps

            # Repopulate the shared buffer with parameter shards so validation
            # all-gathers weights, not stale training gradients.
            if (
                context.state.cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag
                and context.state.cfg.ddp.overlap_param_gather
            ):
                _zero_and_copy_main_params_to_param_buffer(context.model, context.optimizer)

            if should_toggle_fwd_pre_hook:
                train_module.disable_forward_pre_hook(context.model)

            torch.cuda.synchronize()
            torch.distributed.barrier()
            # When EvalCPCallback is active, swap CP topology around the bare
            # evaluate() call (it bypasses on_eval_start/end via callback hooks).
            eval_ctx = (
                eval_cp_cb.eval_topology(context.model) if eval_cp_cb is not None else nullcontext()
            )
            with eval_ctx:
                eval_module.evaluate(
                    context.state,
                    prepared_forward_step_func,
                    val_data_iterator,
                    context.model,
                    None,
                    context.state.cfg,
                    False,
                )

            context.state.train_state.consumed_valid_samples = 0
            context.state.cfg.validation.eval_iters = original_eval_iters

            if should_toggle_fwd_pre_hook:
                train_module.enable_forward_pre_hook(context.model)

            torch.cuda.synchronize()
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                logger.info(f"Finished validation warmup: {time.time() - start_val_time} s. ")

        if (
            is_full_iteration_cuda_graph(model_config)
            and FullCudaGraphWrapper.cuda_graph.get("training") is not None
        ):
            # Use the unreachable capture step -1 to keep train/eval on replay
            # after warmup has captured the full CUDA graph.
            model_config.cuda_graph_warmup_steps = -1

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            logger.info(f"Time spent in run_training_warmup: {time.time() - start}s")

    def on_train_end(self, context: CallbackContext):
        # Let the helper clear layer.cuda_graphs; deleting it first breaks cleanup.
        if self.cuda_graph_helper is None:
            return
        self.cuda_graph_helper.delete_cuda_graphs()
        self.cuda_graph_helper = None

    def on_eval_start(self, context: CallbackContext):
        # train.py already populated param_data and forced eval param sync before
        # eval callbacks run. Only clear the stale dispatch flag left by that
        # force-sync so any later hook-driven path can dispatch normally.
        if (
            context.state.cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag
            and context.state.cfg.ddp.overlap_param_gather
        ):
            _reset_param_gather_dispatched(context.model)
