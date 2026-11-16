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
import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core.utils import get_model_config

logger = logging.getLogger(__name__)


_PG_FIELDS = ("cp", "dp", "dp_cp", "tp_cp", "tp_dp_cp", "intra_dp_cp", "hcp")
_MODULE_ATTRS = {
    "cp_group": "cp",
    "tp_cp_group": "tp_cp",
    "tp_dp_cp_group": "tp_dp_cp",
    "dp_cp_group": "dp_cp",
}
_MCORE_PARALLEL_STATE_GLOBALS = {
    "_CONTEXT_PARALLEL_GROUP": "cp",
    "_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS": "hcp",
    "_DATA_PARALLEL_GROUP_WITH_CP": "dp_cp",
    "_TENSOR_AND_CONTEXT_PARALLEL_GROUP": "tp_cp",
    "_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP": "tp_dp_cp",
}
_YARN_KEYS = ("max_seq_len_cached", "offset_cached", "dtype_cached", "packed_seq_cached")
_YARN_BUFFERS = ("cos_cached", "sin_cached")


class EvalCPCallback(Callback):
    """Swap CP topology around every evaluation pass."""

    _active: "Optional[EvalCPCallback]" = None

    def __init__(self, eval_cp_size: int) -> None:
        self._eval_cp = eval_cp_size
        self._eval_pg: Any = None
        self._cfg_container: Any = None
        self._restore: list[Callable[[], None]] = []
        self._eval_yarn_cache: dict[int, dict[str, Any]] = {}
        EvalCPCallback._active = self

    @classmethod
    def get_active(cls) -> "Optional[EvalCPCallback]":
        return cls._active

    @property
    def eval_dp_group(self) -> Any:
        return None if self._eval_pg is None else self._eval_pg.dp

    @contextmanager
    def eval_topology(self, model: Any) -> Iterator[Any]:
        """Apply the eval CP swap for the duration of the with-block.

        Used by code paths that bypass on_eval_start/end hooks (e.g. warmup).
        """
        if self._eval_pg is None:
            yield None
            return
        self._apply(model)
        try:
            yield self._eval_pg
        finally:
            self._undo()

    def on_data_init_start(self, ctx: CallbackContext) -> None:
        """Build and publish eval groups before warmup and native data setup."""
        self._cfg_container = ctx.state.cfg
        if self._eval_pg is not None:
            ctx.state._eval_pgs = self._eval_pg
            return
        native_eval_pg = getattr(ctx.state, "_eval_pgs", None)
        if native_eval_pg is not None:
            if native_eval_pg.cp.size() != self._eval_cp:
                raise RuntimeError(
                    "MBridge provided eval process groups with context parallel size "
                    f"{native_eval_pg.cp.size()}, expected {self._eval_cp}"
                )
            self._eval_pg = native_eval_pg
        else:
            self._eval_pg = _build_eval_pg(ctx.state.cfg, self._eval_cp)
        # MBridge's native setup_data_iterators() reads state._eval_pgs.dp and
        # uses it only for validation/test samplers, while retaining the model's
        # training DP group for the training sampler.
        ctx.state._eval_pgs = self._eval_pg

    def on_eval_start(self, ctx: CallbackContext) -> None:
        if self._eval_pg is not None:
            self._apply(ctx.model)

    def on_eval_end(self, ctx: CallbackContext) -> None:
        self._undo()

    on_test_start = on_eval_start
    on_test_end = on_eval_end

    def _save_set(self, obj: Any, attr: str, new: Any) -> None:
        old = getattr(obj, attr)
        self._restore.append(lambda: setattr(obj, attr, old))
        setattr(obj, attr, new)

    def _undo(self) -> None:
        for cb in reversed(self._restore):
            cb()
        self._restore.clear()

    def _apply(self, model: Any) -> None:
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        if self._eval_pg is None or self._cfg_container is None:
            raise RuntimeError(
                "EvalCPCallback requires on_data_init_start to complete before applying "
                "the evaluation topology"
            )

        chunks = model if isinstance(model, list) else [model]
        modules = [m for c in chunks for m in c.modules()]
        train_pg = get_pg_collection(model)
        # Bridge passes the same model_cfg object to every chunk's
        # model_provider_func, so chunks share one config instance.
        model_cfg = get_model_config(chunks[0])
        eval_pg = self._eval_pg
        eval_cp = eval_pg.cp
        eval_size = eval_cp.size()
        eval_dp_size = eval_pg.dp.size()

        if torch.distributed.get_rank() == 0:
            logger.info(
                f"[EvalCPCallback] applying swap: train_cp={train_pg.cp.size()} -> eval_cp={eval_size}, "
                f"train_dp={train_pg.dp.size()} -> eval_dp={eval_dp_size}"
            )

        # Snapshot TE attention state BEFORE swap. m.cp_group is None when
        # train_cp==1 (TE.__init__ gates the CP setup on context_parallel_size>1),
        # but we still need to wire eval CP. Driving both apply and restore from
        # this snapshot keeps that case correct.
        cfg_cp_comm_type = getattr(model_cfg, "cp_comm_type", None)
        if not isinstance(cfg_cp_comm_type, str):
            cfg_cp_comm_type = None
        te_train_state: list = []
        for m in modules:
            if not isinstance(m, TEDotProductAttention):
                continue
            train_group = m.cp_group
            train_ranks = (
                dist.get_process_group_ranks(train_group)
                if train_group is not None
                else getattr(m, "cp_global_ranks", None)
            )
            comm_type = cfg_cp_comm_type or getattr(m, "cp_comm_type", None) or "p2p"
            te_train_state.append((m, train_group, train_ranks, comm_type))

        # 1. model_config + train pg_collection fields + ConfigContainer.
        self._save_set(model_cfg, "context_parallel_size", eval_size)
        for f in _PG_FIELDS:
            new = getattr(eval_pg, f, None)
            if new is not None:
                self._save_set(train_pg, f, new)
        # Read by evaluate() to compute eval_num_microbatches.
        self._save_set(self._cfg_container, "data_parallel_size", eval_dp_size)

        # 2. mcore parallel_state module-level globals (read directly by code
        # that bypasses pg_collection).
        try:
            from megatron.core import parallel_state as mpu
            for global_name, pg_field in _MCORE_PARALLEL_STATE_GLOBALS.items():
                new_value = getattr(eval_pg, pg_field, None)
                if new_value is not None and hasattr(mpu, global_name):
                    self._save_set(mpu, global_name, new_value)
        except ImportError:
            pass

        # 3. Per-module pg_collection + cached cp_group refs.
        for m in modules:
            if hasattr(m, "pg_collection"):
                self._save_set(m, "pg_collection", eval_pg)
            for attr, pg_field in _MODULE_ATTRS.items():
                if hasattr(m, attr):
                    self._save_set(m, attr, getattr(eval_pg, pg_field, None))
            if isinstance(getattr(m, "cp_size", None), int):
                self._save_set(m, "cp_size", eval_size)

        # 4. YarnRotaryEmbedding cos/sin buffers. The wrapper installed here
        # makes cp_group part of forward()'s lru_cache key — full-iteration
        # CUDA graphs may have captured cached tensor pointers, so do NOT
        # cache_clear; instead let train/eval coexist as separate entries.
        YarnRotaryEmbedding = _wrap_yarn_forward()
        if YarnRotaryEmbedding is not None:
            for m in modules:
                if not isinstance(m, YarnRotaryEmbedding) or not hasattr(m, "cos_cached"):
                    continue
                train_snap = _snapshot_yarn(m)
                self._restore.append(lambda m=m, s=train_snap: self._restore_yarn(m, s))
                eval_snap = self._eval_yarn_cache.get(id(m))
                if eval_snap is None:
                    m._set_cos_sin_cache(*[train_snap[f] for f in _YARN_KEYS])
                    self._eval_yarn_cache[id(m)] = _snapshot_yarn(m)
                else:
                    _set_yarn_state(m, eval_snap)

        # 5. TE attention: runtime equivalent of construction-time CP setup.
        if getattr(TEDotProductAttention, "cp_stream", None) is None:
            TEDotProductAttention.cp_stream = torch.cuda.Stream()
        stream = TEDotProductAttention.cp_stream
        eval_ranks = dist.get_process_group_ranks(eval_cp)
        for m, train_group, train_ranks, comm_type in te_train_state:
            restore_stream = stream if train_group is not None else None
            self._restore.append(
                lambda m=m, g=train_group, r=train_ranks, s=restore_stream, t=comm_type:
                    m.set_context_parallel_group(g, r, s, t)
            )
            m.set_context_parallel_group(eval_cp, eval_ranks, stream, comm_type)

    def _restore_yarn(self, m: Any, train_snap: dict) -> None:
        # Eval forward may grow/retype Yarn cos/sin buffers. Keep the final
        # eval tensors alive for validation CUDA-graph replay, then put train
        # tensors back on the module.
        self._eval_yarn_cache[id(m)] = _snapshot_yarn(m)
        _set_yarn_state(m, train_snap)


def _wrap_yarn_forward() -> Optional[type]:
    """Inject self.cp_group into YarnRotaryEmbedding.forward's lru_cache key.

    Idempotent — flag stored on the class's own __dict__ (not via setattr-and-
    getattr) so MRO inheritance does not skip a future subclass.
    """
    try:
        from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
    except ImportError:
        return None

    if YarnRotaryEmbedding.__dict__.get("_eval_cp_injected", False):
        return YarnRotaryEmbedding
    original = YarnRotaryEmbedding.forward
    if not hasattr(original, "cache_clear"):
        return YarnRotaryEmbedding

    @functools.wraps(original)
    def wrapped(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        if cp_group is None:
            cp_group = self.cp_group
        return original(self, max_seq_len, offset, packed_seq, cp_group)

    YarnRotaryEmbedding.forward = wrapped
    YarnRotaryEmbedding._eval_cp_injected = True  # type: ignore[attr-defined]
    return YarnRotaryEmbedding


def _snapshot_yarn(m: Any) -> dict:
    return {f: getattr(m, f) for f in _YARN_KEYS + _YARN_BUFFERS}


def _set_yarn_state(m: Any, snap: dict) -> None:
    for f in _YARN_KEYS:
        setattr(m, f, snap[f])
    for f in _YARN_BUFFERS:
        m.register_buffer(f, snap[f], persistent=False)


def _build_eval_pg(cfg: Any, eval_cp_size: int) -> Any:
    """pg_collection sized for eval_cp_size, sharing TP/PP with the train pg."""
    from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
    from megatron.bridge.models.mamba.mamba_builder import MambaModelConfig
    from megatron.bridge.training.initialize import _create_pg_collection

    mc = (
        cfg.model.transformer
        if isinstance(cfg.model, (GPTModelConfig, MambaModelConfig))
        else cfg.model
    )
    saved = mc.context_parallel_size
    mc.context_parallel_size = eval_cp_size
    try:
        return _create_pg_collection(
            mc, num_distributed_optimizer_instances=cfg.ddp.num_distributed_optimizer_instances
        )
    finally:
        mc.context_parallel_size = saved
