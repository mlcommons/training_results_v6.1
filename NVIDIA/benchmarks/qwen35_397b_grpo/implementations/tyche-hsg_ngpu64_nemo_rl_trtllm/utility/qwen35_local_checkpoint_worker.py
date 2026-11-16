# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Bridge local checkpoints behind NeMo-RL's policy-worker extension."""

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor

import ray
import torch
from megatron.bridge.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedObject
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.core import (
    AsyncCaller,
    AsyncCallsQueue,
)
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)

from .qwen35_local_checkpoint import (
    fetch_rank_checkpoint,
    local_endpoint_root,
    publish_rank_location,
)


def _collective_io(operation):
    """Keep every rank out of Bridge collectives if a local filesystem operation fails."""
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f"rank {torch.distributed.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(
            "Local checkpoint I/O failed: " + "; ".join(filter(None, errors))
        )
    return result


def _start_timer():
    torch.cuda.synchronize()
    return time.perf_counter()


def _snapshot_metadata(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_snapshot_metadata(item) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    return copy.deepcopy(value)


def _emit_timing(phase, weights_path, directory, started, include_size=True):
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    local_bytes = (
        _collective_io(lambda: sum(path.stat().st_size for path in directory.glob("*.pt")))
        if include_size
        else None
    )
    print(
        "QWEN35_LOCAL_CHECKPOINT_TIMING "
        + json.dumps(
            {
                "phase": phase,
                "rank": torch.distributed.get_rank(),
                "checkpoint": str(weights_path),
                "seconds": elapsed,
                "local_bytes": local_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )


class _CPUThreadCaller(AsyncCaller):
    """Run only CPU writes in the background; finalize on the policy actor."""

    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def schedule_async_call(self, request):
        if self.future is not None or request.preload_fn is not None:
            raise RuntimeError("Local async writes require one already-staged request")
        self.future = self.executor.submit(
            request.async_fn, *request.async_fn_args, **(request.async_fn_kwargs or {})
        )

    def is_current_async_call_done(self, blocking=False, no_dist=False):
        pending = self.future is not None and not self.future.done()
        if not blocking:
            done = not pending if no_dist else self.sync_all_async_calls(int(pending))
            if not done:
                return False
        if self.future is not None:
            result = self.future.result
            # All peers must observe write failures before native finalize barriers.
            if no_dist:
                result()
            else:
                _collective_io(result)
            self.future = None
        return True

    def close(self, abort=False):
        self.executor.shutdown(wait=not abort, cancel_futures=abort)

    def __del__(self):
        self.close(abort=True)


class _LocalAsyncCallsQueue(AsyncCallsQueue):
    def _get_async_caller(self):
        if self.persistent_caller is None:
            self.persistent_caller = _CPUThreadCaller()
        return self.persistent_caller

    def __del__(self):
        # Exceptions are propagated by explicit finalization, never by destruction.
        self.close(abort=True)


class _LocalCheckpointManager(LocalCheckpointManager):
    def save(self, state_dict, iteration, is_async=False):
        self._write_context = {"rank": self.rank, "checkpoint": str(self.local_ckpt_dir)}
        if is_async:
            # Model shards are independently copied by native D2H staging. Common
            # counters/extra state must not retain references to live training data.
            state_dict.common = _snapshot_metadata(state_dict.common)
            for shard in nested_values(state_dict.sharded_state_dict):
                if isinstance(shard, ShardedObject):
                    shard.data = _snapshot_metadata(shard.data)
        request = super().save(state_dict, iteration, is_async=is_async)
        if request is not None:
            request = request._replace(async_fn=self._write_cpu)
            request.add_finalize_fn(self.on_complete)
        return request

    def _write_cpu(self, state_dicts):
        # Native async save has completed and synchronized D2H before scheduling.
        # Never call the synchronous override: it contains CUDA and collectives.
        started = time.perf_counter()
        super()._save_fn(state_dicts)
        print(
            "QWEN35_LOCAL_CHECKPOINT_TIMING "
            + json.dumps(
                {
                    **self._write_context,
                    "phase": "local_write",
                    "seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Native synchronous save/load enters collectives immediately after local I/O.
    # Report an I/O error to peers before they reach those collectives.
    def _save_fn(self, state_dicts):
        def write():
            # NVRx also stages D2H nonblocking in its synchronous save path.
            torch.cuda.synchronize()
            return super(_LocalCheckpointManager, self)._save_fn(state_dicts)

        return _collective_io(write)

    def _load_fn(self, checkpoint_id):
        return _collective_io(
            lambda: super(_LocalCheckpointManager, self)._load_fn(checkpoint_id)
        )


class LocalCheckpointPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    """Save weight shards locally; restore each logical rank before the usual refit.

    Async local writes use a CPU thread with native staging and finalization,
    without forking a live Ray/CUDA actor or creating another CUDA process.
    """

    def __init__(self, config, tokenizer, weights_path=None, **kwargs):
        if config.get("quant_cfg"):
            raise ValueError(
                "Native local checkpoints do not support ModelOpt quantization"
            )
        # An explicit path always means a deferred endpoint in this opt-in worker.
        # Missing rank metadata must fail restore, never select the base model.
        local_restore = weights_path is not None
        config = copy.deepcopy(config)
        checkpoint_config = config["megatron_cfg"].setdefault("checkpoint", {})
        is_async = bool(checkpoint_config.get("async_save")) and not local_restore
        # Prevent Bridge from prewarming a multiprocessing/CUDA writer.
        checkpoint_config["async_save"] = False
        if local_restore:
            kwargs["skip_weight_load"] = True
        super().__init__(config, tokenizer, weights_path=weights_path, **kwargs)
        self.cfg["megatron_cfg"]["checkpoint"]["async_save"] = is_async
        self.mcore_state.cfg.checkpoint.async_save = is_async
        if is_async:
            self.mcore_state._async_calls_queue = _LocalAsyncCallsQueue()
        if local_restore:
            self.load_checkpoint(weights_path)

    def _checkpoint_manager(self, weights_path):
        manager = _LocalCheckpointManager(str(local_endpoint_root(weights_path)))
        ckpt = self.mcore_state.cfg.checkpoint
        ckpt.non_persistent_ckpt_type = "local"
        ckpt.non_persistent_local_ckpt_algo = "fully_parallel"
        ckpt.save_rng = False
        ckpt.save_optim = False
        ckpt.save = str(manager.root_local_ckpt_dir)
        self.checkpointing_context["local_checkpoint_manager"] = manager
        return manager

    def save_checkpoint(self, weights_path, optimizer_path=None, **kwargs):
        if optimizer_path is not None:
            raise ValueError("Deferred local checkpoints save model weights only")
        self.finalize_async_save()
        started = _start_timer()
        manager = _collective_io(lambda: self._checkpoint_manager(weights_path))
        is_async = self.mcore_state.cfg.checkpoint.async_save
        step = int(self.mcore_state.train_state.step)

        def complete():
            _collective_io(
                lambda: publish_rank_location(
                    weights_path,
                    torch.distributed.get_rank(),
                    manager.local_ckpt_dir,
                    step,
                )
            )
            _emit_timing(
                "local_save_complete" if is_async else "local_save",
                weights_path,
                manager.local_ckpt_dir,
                started,
            )

        manager.on_complete = complete
        _collective_io(
            lambda: manager.local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        )
        self.model = self.move_model(
            self.model, "cuda", move_params=True, move_grads=False
        )
        torch.cuda.synchronize()
        restore_hook = (
            self.should_disable_forward_pre_hook and self._forward_pre_hook_enabled()
        )
        if restore_hook:
            self.disable_forward_pre_hook()
        try:
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=None,
                opt_param_scheduler=None,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
                non_persistent_ckpt=True,
            )
        finally:
            if restore_hook:
                self.enable_forward_pre_hook()
        if is_async:
            _emit_timing(
                "local_save_staging",
                weights_path,
                manager.local_ckpt_dir,
                started,
                include_size=False,
            )
        else:
            complete()

    def load_checkpoint(self, weights_path, optimizer_path=None):
        if optimizer_path is not None or self.optimizer is not None:
            raise ValueError(
                "Deferred local checkpoint loading requires an inference-only policy"
            )
        manager = _collective_io(lambda: self._checkpoint_manager(weights_path))
        started = _start_timer()
        iteration = _collective_io(
            lambda: fetch_rank_checkpoint(
                weights_path, torch.distributed.get_rank(), manager.local_ckpt_dir
            )
        )
        _emit_timing("fetch_remap", weights_path, manager.local_ckpt_dir, started)
        started = _start_timer()
        if manager.find_latest() != iteration:
            raise RuntimeError("Local checkpoint is incomplete after rank remapping")
        ckpt = self.mcore_state.cfg.checkpoint
        ckpt.load = str(manager.root_local_ckpt_dir)
        ckpt.pretrained_checkpoint = None
        ckpt.ckpt_step = None
        ckpt.finetune = True
        ckpt.load_optim = False
        ckpt.load_rng = False
        self.model = self.move_model(
            self.model, "cuda", move_params=True, move_grads=False
        )
        load_checkpoint(
            self.mcore_state,
            [self.model],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=self.checkpointing_context,
        )
        _emit_timing("native_load", weights_path, manager.local_ckpt_dir, started)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class LocalCheckpointPolicyWorker(LocalCheckpointPolicyWorkerImpl):
    pass
