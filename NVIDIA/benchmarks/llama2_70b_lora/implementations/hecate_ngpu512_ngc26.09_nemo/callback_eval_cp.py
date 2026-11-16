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


class EvalCPCallback(Callback):
    """Swap CP topology around every evaluation pass."""

    _patched = False
    _validate_patched = False
    _active: "Optional[EvalCPCallback]" = None

    def __init__(self, eval_cp_size: int) -> None:
        self._eval_cp = eval_cp_size
        self._eval_pg: Any = None
        self._cfg_container: Any = None
        self._restore: list = []
        EvalCPCallback._active = self
        if not EvalCPCallback._validate_patched:
            EvalCPCallback._validate_patched = True
            self._install_validate_patch()

    def _install_validate_patch(self) -> None:
        from megatron.bridge.training import config as _cfg_mod

        cls = _cfg_mod.ConfigContainer
        original = cls.validate
        eval_cp = self._eval_cp

        def patched(cc_self) -> None:
            train_cp = cc_self.model.context_parallel_size
            dp_total = cc_self.data_parallel_size * train_cp
            do_override = eval_cp > 1 and dp_total % eval_cp == 0
            saved_dp = cc_self.data_parallel_size
            if do_override:
                cc_self.data_parallel_size = dp_total // eval_cp
            try:
                original(cc_self)
            finally:
                if do_override:
                    cc_self.data_parallel_size = saved_dp

        cls.validate = patched

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

    def _install_patch(self) -> None:
        from megatron.bridge.training import setup as sm
        original = sm.setup_data_iterators
        outer = self

        def patched(**kw: Any) -> tuple:
            train_iter, _, _ = original(**kw)
            cfg = kw["cfg"]
            outer._cfg_container = cfg
            eval_dp_size = outer._eval_pg.dp.size()

            saved_gbs = cfg.train.global_batch_size
            divisor = cfg.train.micro_batch_size * eval_dp_size
            if saved_gbs % divisor != 0:
                cfg.train.global_batch_size = divisor
            try:
                kw["dp_group"] = outer._eval_pg.dp
                _, eval_v, eval_t = original(**kw)
            finally:
                cfg.train.global_batch_size = saved_gbs
            return train_iter, eval_v, eval_t

        sm.setup_data_iterators = patched

    def on_data_init_start(self, ctx: CallbackContext) -> None:
        """Build eval pg before warmup runs (warmup's eval_topology needs it)."""
        if self._eval_pg is not None:
            return
        self._cfg_container = ctx.state.cfg
        self._eval_pg = _build_eval_pg(ctx.state.cfg, self._eval_cp)
        _install_rope_cp_group_injectors()
        if not EvalCPCallback._patched:
            EvalCPCallback._patched = True
            self._install_patch()

    def on_eval_start(self, ctx: CallbackContext) -> None:
        if self._eval_pg is not None:
            self._apply(ctx.model)

    def on_eval_end(self, ctx: CallbackContext) -> None:
        self._undo()

    on_test_start = on_eval_start
    on_test_end = on_eval_end

    def _undo(self) -> None:
        for cb in reversed(self._restore):
            cb()
        self._restore.clear()

    def _save_set(self, obj: Any, attr: str, new: Any) -> None:
        old = getattr(obj, attr)
        self._restore.append(lambda: setattr(obj, attr, old))
        setattr(obj, attr, new)

    def _apply(self, model: Any) -> None:
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        chunks = model if isinstance(model, list) else [model]
        train_pg = get_pg_collection(model)
        chunk_cfgs: list = []
        seen: set[int] = set()
        for c in chunks:
            cfg = get_model_config(c)
            if id(cfg) not in seen:
                seen.add(id(cfg))
                chunk_cfgs.append(cfg)
        model_cfg = chunk_cfgs[0]
        eval_pg = self._eval_pg
        eval_cp = eval_pg.cp
        eval_size = eval_cp.size()
        eval_dp_size = eval_pg.dp.size()
        modules = [m for c in chunks for m in c.modules()]

        if torch.distributed.get_rank() == 0:
            logger.info(
                f"[EvalCPCallback] applying swap: train_cp={train_pg.cp.size()} -> eval_cp={eval_size}, "
                f"train_dp={train_pg.dp.size()} -> eval_dp={eval_dp_size}"
            )

        # 1. pg_collection fields + per-chunk model_config + ConfigContainer.
        for cfg in chunk_cfgs:
            self._save_set(cfg, "context_parallel_size", eval_size)
        for f in _PG_FIELDS:
            new = getattr(eval_pg, f, None)
            if new is not None:
                self._save_set(train_pg, f, new)
        if self._cfg_container is not None:
            # Read by evaluate() to compute eval_num_microbatches.
            self._save_set(self._cfg_container, "data_parallel_size", eval_dp_size)

        # 2. mcore parallel_state module-level globals.
        try:
            from megatron.core import parallel_state as mpu
        except ImportError:
            mpu = None
        if mpu is not None:
            for global_name, pg_field in _MCORE_PARALLEL_STATE_GLOBALS.items():
                new_value = getattr(eval_pg, pg_field, None)
                if new_value is not None and hasattr(mpu, global_name):
                    self._save_set(mpu, global_name, new_value)

        # 3. Per-module attrs.
        for m in modules:
            if hasattr(m, "pg_collection"):
                self._save_set(m, "pg_collection", eval_pg)
            for attr, pg_field in _MODULE_ATTRS.items():
                if hasattr(m, attr):
                    self._save_set(m, attr, getattr(eval_pg, pg_field, None))
            if isinstance(getattr(m, "cp_size", None), int):
                self._save_set(m, "cp_size", eval_size)

        # 4. TE attention cp_group swap.
        if getattr(TEDotProductAttention, "cp_stream", None) is None:
            TEDotProductAttention.cp_stream = torch.cuda.Stream()
        stream = TEDotProductAttention.cp_stream
        eval_ranks = dist.get_process_group_ranks(eval_cp)
        cfg_cp_comm_type = getattr(model_cfg, "cp_comm_type", None)
        if not isinstance(cfg_cp_comm_type, str):
            cfg_cp_comm_type = None
        for m in modules:
            if not isinstance(m, TEDotProductAttention):
                continue
            train_group = m.cp_group  # may be None when train_cp == 1
            train_ranks = (
                dist.get_process_group_ranks(train_group)
                if train_group is not None
                else getattr(m, "cp_global_ranks", None)
            )
            comm_type = cfg_cp_comm_type or getattr(m, "cp_comm_type", None) or "p2p"
            train_stream = stream if train_group is not None else None
            self._restore.append(
                lambda m=m, g=train_group, r=train_ranks, s=train_stream, t=comm_type:
                    m.set_context_parallel_group(g, r, s, t)
            )
            m.set_context_parallel_group(eval_cp, eval_ranks, stream, comm_type)


def _install_rope_cp_group_injectors() -> None:
    """Make RoPE lru_cache entries topology-specific.

    RotaryEmbedding.forward is cached by the arguments passed to it. Some call
    sites omit cp_group and rely on self.cp_group inside the function body, which
    would otherwise reuse a train-CP cache entry during eval-CP. Injecting
    self.cp_group into the key lets train and eval CUDA graphs capture different
    stable tensors without invalidating either cache entry.
    """
    try:
        from megatron.core.models.common.embeddings import rotary_pos_embedding as rope_mod
    except ImportError:
        return

    def _make_wrapper(original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapped(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
            if cp_group is None:
                cp_group = self.cp_group
            return original(self, max_seq_len, offset, packed_seq, cp_group)

        return wrapped

    for name in ("RotaryEmbedding", "MultimodalRotaryEmbedding"):
        cls = getattr(rope_mod, name, None)
        if cls is None or getattr(cls, "_eval_cp_injected", False):
            continue
        original = cls.forward
        if not hasattr(original, "cache_clear"):
            continue
        cls.forward = _make_wrapper(original)
        cls._eval_cp_injected = True  # type: ignore[attr-defined]


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
