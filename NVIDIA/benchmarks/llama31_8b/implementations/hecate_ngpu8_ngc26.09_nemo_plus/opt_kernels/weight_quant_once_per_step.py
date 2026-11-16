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
#
# copy_main_params_to_model_params is adapted from Megatron-Core's
# DistributedOptimizer._copy_main_params_to_model_params, and the quantize_weight call
# reproduces the keyword set of Transformer Engine's module/base.py (both Apache-2.0,
# Copyright (c) NVIDIA CORPORATION & AFFILIATES).


from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
import triton
import triton.language as tl


_BLOCK = 1024
_UNROLL = 4
_NUM_WARPS = 4
_SLOT_STRIDE = 8


@triton.jit(do_not_specialize=["slot"])
def _copy_f32_to_bf16_amax_kernel(
    src_ptr,
    dst_ptr,
    amax_ptr,
    slot,
    n_elements,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
    HAS_AMAX: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * (BLOCK * UNROLL)
    m = tl.zeros([BLOCK], dtype=tl.float32)
    for u in tl.static_range(UNROLL):
        offs = base + u * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(src_ptr + offs, mask=mask, other=0.0)
        y = x.to(tl.bfloat16)
        tl.store(dst_ptr + offs, y, mask=mask)
        if HAS_AMAX:
            m = tl.maximum(m, tl.abs(y.to(tl.float32)))
    if HAS_AMAX:
        tl.atomic_max(amax_ptr + slot, tl.max(m, axis=0))


def copy_f32_to_bf16_amax(
    src: torch.Tensor, dst: torch.Tensor, amax: torch.Tensor, slot: int
) -> None:
    n = src.numel()
    if n == 0:
        return
    if not (
        src.dtype == torch.float32
        and dst.dtype == torch.bfloat16
        and src.is_contiguous()
        and dst.is_contiguous()
        and dst.numel() == n
        and n < 2**31
    ):
        dst.copy_(src)
        if slot >= 0:
            off = slot * _SLOT_STRIDE
            torch.maximum(amax[off], dst.abs().amax().float(), out=amax[off])
        return
    grid = (triton.cdiv(n, _BLOCK * _UNROLL),)
    _copy_f32_to_bf16_amax_kernel[grid](
        src,
        dst,
        amax,
        max(slot, 0) * _SLOT_STRIDE,
        n,
        BLOCK=_BLOCK,
        UNROLL=_UNROLL,
        HAS_AMAX=slot >= 0,
        num_warps=_NUM_WARPS,
    )


class _Entry:
    __slots__ = ("param", "slot", "order", "quantizer", "ws", "event", "pending", "n_prequant", "n_consumed", "n_fallback")

    def __init__(self, param: torch.nn.Parameter, slot: int):
        self.param = param
        self.slot = slot
        self.order = 1 << 30
        self.quantizer = None
        self.ws = None
        self.event = None
        self.pending = False
        self.n_prequant = 0
        self.n_consumed = 0
        self.n_fallback = 0


class _State:
    enabled = False
    installed = False
    debug = False
    by_param: Dict[int, _Entry] = {}
    by_ptr: Dict[int, _Entry] = {}
    slots = 0
    amax: Optional[torch.Tensor] = None
    amax_ready = False
    side: Optional[torch.cuda.Stream] = None
    learn_counter = 0
    opt_steps = 0
    reported = False


_S = _State()


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def _log(msg: str, all_ranks: bool = False) -> None:
    if all_ranks or _rank() == 0:
        sys.stderr.write(f"[opt wq rank {_rank()}] {msg}\n")
        sys.stderr.flush()


def _register_param(p: torch.nn.Parameter) -> None:
    if id(p) in _S.by_param:
        return
    if p.dim() != 2 or p.dtype not in (torch.bfloat16, torch.float16):
        return
    if p.shape[0] % 128 != 0 or p.shape[1] % 128 != 0:
        return
    _S.by_param[id(p)] = _Entry(p, _S.slots)
    _S.slots += 1


def _refresh_ptr_map() -> None:
    _S.by_ptr = {e.param.data_ptr(): e for e in _S.by_param.values()}


def _lookup(tensor: torch.Tensor) -> Optional[_Entry]:
    if tensor.dim() != 2:
        return None
    e = _S.by_ptr.get(tensor.data_ptr())
    if e is None or e.param.shape != tensor.shape:
        return None
    return e


def _amax_buffer() -> torch.Tensor:
    if _S.amax is None or _S.amax.numel() < max(_S.slots, 1) * _SLOT_STRIDE:
        _S.amax = torch.zeros(max(_S.slots, 1) * _SLOT_STRIDE, dtype=torch.float32, device="cuda")
    return _S.amax


def _side_stream() -> torch.cuda.Stream:
    if _S.side is None:
        _S.side = torch.cuda.Stream()
    return _S.side


def _learn(e: _Entry, quantizer) -> None:
    if e.quantizer is not None:
        return
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if not isinstance(quantizer, NVFP4Quantizer):
        return
    if quantizer.with_rht or quantizer.stochastic_rounding or getattr(quantizer, "row_scaled_nvfp4", False):
        return
    if getattr(quantizer, "with_amax_reduction", False):
        return
    q = quantizer.copy()
    q.set_usage(rowwise=True, columnwise=True)
    e.quantizer = q
    e.order = _S.learn_counter
    _S.learn_counter += 1
    if e.event is None:
        e.event = torch.cuda.Event()


def _consume(e: _Entry):
    torch.cuda.current_stream().wait_event(e.event)
    e.pending = False
    e.n_consumed += 1
    return e.ws


def _prequantize(e: _Entry):
    import transformer_engine_torch as tex

    off = e.slot * _SLOT_STRIDE
    amax_view = _S.amax[off : off + 1]
    return tex.nvfp4_quantize_with_amax(e.param.detach(), e.quantizer, amax_view, amax_view)


def _cmp(name: str, a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> str:
    if a is None and b is None:
        return f"{name}=both-none"
    if a is None or b is None:
        return f"{name}=ONE-NONE"
    if a.shape != b.shape:
        return f"{name}=SHAPE{tuple(a.shape)}vs{tuple(b.shape)}"
    nbad = int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item()) if a.dtype == torch.uint8 else int((a != b).sum().item())
    return f"{name}={'ok' if nbad == 0 else 'BAD:' + str(nbad)}"


def _debug_compare(e: _Entry, ref, tag: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    ws = e.ws
    parts = [
        _cmp("rowwise_data", getattr(ws, "_rowwise_data", None), getattr(ref, "_rowwise_data", None)),
        _cmp("rowwise_scale_inv", getattr(ws, "_rowwise_scale_inv", None), getattr(ref, "_rowwise_scale_inv", None)),
        _cmp("columnwise_data", getattr(ws, "_columnwise_data", None), getattr(ref, "_columnwise_data", None)),
        _cmp("columnwise_scale_inv", getattr(ws, "_columnwise_scale_inv", None), getattr(ref, "_columnwise_scale_inv", None)),
    ]
    a_ws = ws._amax_rowwise.float().item() if ws._amax_rowwise is not None else None
    a_ref = ref._amax_rowwise.float().item() if ref._amax_rowwise is not None else None
    sw = getattr(ws, "_with_gemm_swizzled_scales", None)
    sr = getattr(ref, "_with_gemm_swizzled_scales", None)
    _log(
        f"DEBUG {tag} shape={tuple(e.param.shape)} slot={e.slot} step={_S.opt_steps} "
        f"amax_ws={a_ws!r} amax_te={a_ref!r} amax_eq={a_ws == a_ref} swz_ws={sw} swz_te={sr} "
        f"type_ws={type(ws).__name__} type_te={type(ref).__name__} " + " ".join(parts),
        all_ranks=True,
    )


def _patch_te():
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    import transformer_engine.pytorch.module.base as te_base
    import transformer_engine.pytorch.module.linear as te_linear
    import transformer_engine.pytorch.module.layernorm_linear as te_lnl

    orig_quantize_impl = NVFP4Quantizer.quantize_impl

    def quantize_impl(self, tensor):
        if _S.enabled and isinstance(tensor, torch.Tensor):
            e = _lookup(tensor)
            if e is not None:
                if e.pending:
                    ws = _consume(e)
                    if _S.debug and not torch.cuda.is_current_stream_capturing():
                        _debug_compare(e, orig_quantize_impl(self, tensor), "ops-path")
                    return ws
                _learn(e, self)
                if e.quantizer is not None and _S.amax_ready:
                    e.n_fallback += 1
        return orig_quantize_impl(self, tensor)

    NVFP4Quantizer.quantize_impl = quantize_impl

    orig_quantize_weight = te_base.quantize_weight

    def quantize_weight(
        *,
        tensor=None,
        quantizer=None,
        workspace=None,
        update_workspace=True,
        skip_update_flag=None,
        fsdp_group=None,
        workspace_dtype=None,
        cache=False,
    ):
        if (
            _S.enabled
            and tensor is not None
            and not isinstance(tensor, QuantizedTensor)
            and (quantizer is None or isinstance(quantizer, NVFP4Quantizer))
            and skip_update_flag is None
        ):
            e = _lookup(tensor)
            if e is not None:
                if e.pending:
                    ws = _consume(e)
                    if _S.debug and workspace is not None and not torch.cuda.is_current_stream_capturing():
                        import transformer_engine_torch as tex

                        tex.quantize(tensor, quantizer, workspace, None)
                        _debug_compare(e, workspace, "module-path")
                    return ws, (ws if cache else None)
                if quantizer is not None and e.quantizer is None:
                    saved_internal = quantizer.internal
                    quantizer.internal = False
                    try:
                        _learn(e, quantizer)
                    finally:
                        quantizer.internal = saved_internal
                if e.quantizer is not None and _S.amax_ready:
                    e.n_fallback += 1
        return orig_quantize_weight(
            tensor=tensor,
            quantizer=quantizer,
            workspace=workspace,
            update_workspace=update_workspace,
            skip_update_flag=skip_update_flag,
            fsdp_group=fsdp_group,
            workspace_dtype=workspace_dtype,
            cache=cache,
        )

    te_base.quantize_weight = quantize_weight
    te_linear.quantize_weight = quantize_weight
    te_lnl.quantize_weight = quantize_weight


def _patch_mcore():
    from megatron.core.distributed import param_and_grad_buffer as pgb
    from megatron.core.optimizer import distrib_optimizer as dopt

    BG = pgb._ParamAndGradBucketGroup
    orig_bg_init = BG.__init__

    def bg_init(self, *args, **kwargs):
        orig_bg_init(self, *args, **kwargs)
        if not _S.enabled:
            return
        for bucket in self.buckets:
            for p in bucket.params_list:
                _register_param(p)

    BG.__init__ = bg_init

    orig_start_param_sync = BG.start_param_sync

    def start_param_sync(self, force_sync: bool = False):
        orig_start_param_sync(self, force_sync)
        if not _S.enabled:
            return
        _refresh_ptr_map()
        if force_sync or self.param_gather_handle is None or not _S.amax_ready:
            return
        todo: List[_Entry] = []
        for bucket in self.buckets:
            for p in bucket.params_list:
                e = _S.by_param.get(id(p))
                if e is not None and e.quantizer is not None:
                    todo.append(e)
        if not todo:
            return
        todo.sort(key=lambda e: e.order)
        cur = torch.cuda.current_stream()
        side = _side_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            self.param_gather_handle.wait()
            for e in todo:
                e.ws = _prequantize(e)
                e.event.record(side)
        for e in todo:
            e.pending = True
            e.n_prequant += 1

    BG.start_param_sync = start_param_sync

    DO = dopt.DistributedOptimizer
    orig_copy_main = DO._copy_main_params_to_model_params
    is_nvfp4tensor = getattr(dopt, "is_nvfp4tensor", lambda t: False)

    def copy_main_params_to_model_params(self):
        if (
            not _S.enabled
            or self.ddp_config.fp8_param_gather
            or getattr(self.ddp_config, "fp4_param_gather", False)
            or getattr(self.ddp_config, "use_megatron_fsdp", False)
            or getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        ):
            return orig_copy_main(self)
        _refresh_ptr_map()
        amax = _amax_buffer()
        amax.zero_()

        def shard_view(model_param):
            param_range_map = self._get_model_param_range_map(model_param)
            world_range = param_range_map["gbuf_world_in_bucket"]
            gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
            model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data
            return model_param_buffer.view(-1)[world_range.start : world_range.end]

        n_amax = 0
        for shard_main_group, model_group in zip(self.shard_fp32_from_float16_groups, self.model_float16_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                if self._is_distopt_quantized_param(model_param) or is_nvfp4tensor(model_param):
                    raise RuntimeError("opt wq: quantized primary params are not handled here")
                shard_model_param = shard_view(model_param)
                assert shard_model_param.numel() == shard_main_param.nelement()
                e = _S.by_param.get(id(model_param))
                slot = e.slot if (e is not None and e.quantizer is not None) else -1
                if slot >= 0:
                    n_amax += 1
                copy_f32_to_bf16_amax(shard_main_param, shard_model_param, amax, slot)
        for shard_main_group, model_group in zip(self.shard_fp32_groups, self.model_fp32_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                shard_view(model_param).data.copy_(shard_main_param)
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=self.data_parallel_group)
        _S.amax_ready = True
        _S.opt_steps += 1
        if not _S.reported and _S.opt_steps >= 2:
            _S.reported = True
            learned = [e for e in _S.by_param.values() if e.quantizer is not None]
            _log(
                f"registered={len(_S.by_param)} learned={len(learned)} slots={_S.slots} "
                f"amax_launches_this_rank={n_amax} prequant_total={sum(e.n_prequant for e in learned)} "
                f"consumed_total={sum(e.n_consumed for e in learned)} fallback_total={sum(e.n_fallback for e in learned)} "
                f"internal_true={sum(1 for e in learned if e.quantizer.internal)} "
                f"preswizzle_true={sum(1 for e in learned if e.quantizer.optimize_for_gemm)}"
            )

    DO._copy_main_params_to_model_params = copy_main_params_to_model_params


def install() -> bool:
    if _S.installed:
        return _S.enabled
    _S.installed = True
    if os.environ.get("OPT_WQ_DISABLE", "0") == "1":
        _log("disabled by OPT_WQ_DISABLE=1")
        return False
    _S.debug = os.environ.get("OPT_WQ_DEBUG", "0") == "1"
    _patch_te()
    _patch_mcore()
    _S.enabled = True
    _log(f"installed (debug={_S.debug})")
    return True
