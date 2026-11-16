"""LoRA input dropout with less memory held for the backward, bit-identical to the stock path.

Loaded through the recipe's patch hook (TRAINLOOM_PY_PATCHES=...,lora_dropout_packed). NeMo's
ParallelLinearAdapter applies dropout (p=0.1, position 'pre') to its input before linear_in. Stock, each
adapter holds for the backward the dropout mask (1 byte per element) and linear_in's input, the dropped
tensor (BF16): 64 + 128 MiB per adapter at sequence 8192 x hidden 8192, two adapters (qkv, dense) per layer.

Here:
  - the dropout is the same torch.native_dropout call, so the masks (and the RNG stream) are unchanged; the
    mask is kept as packed bits (8 MiB) and the backward scales the gradient from the bits;
  - for the dense adapter, whose input is the core attention output the attention backward already holds,
    linear_in keeps a reference to that input and the bits instead of the dropped copy, and the dropped
    input is rebuilt when the backward needs it. The qkv adapter's input (the norm output) is held by
    nothing else, so there linear_in keeps the dropped tensor as before.

Both rebuilt tensors use native_dropout's own formula (input * mask * (1/(1-p)) in fp32, rounded once), so the
forward and the gradients are bit-identical to the stock adapter (check_lora_dropout_packed.py checks it).

With fix #2 (hs_lora_epilogue, loaded before this module) the adapters no longer go through
ParallelLinearAdapter.forward: its lora_apply runs the dropout inside its own _LoraDown and folds the LoRA input
gradient into the base GEMM's dgrad with the bool mask. This module then replaces hs_lora_epilogue's _LoraDown and
lora_dgrad_acc_ (both looked up by name at call time) with versions that keep the same packed bits and the same
dense-adapter rebuild, so fix #2's arithmetic is unchanged and only what is held for the backward shrinks.
"""

import os
import sys
import threading

import torch
import triton
import triton.language as tl
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    LoraDenseAttentionAdapter,
    ParallelLinearAdapter,
)
from torch.autograd.graph import saved_tensors_hooks

BLOCK = 4096
# "all": both adapters keep packed masks; "dense": only the dense adapter (where the dropped copy goes), the qkv
# adapter runs NeMo's own forward and pays none of the packing cost. With hs_lora_epilogue both adapters pack.
ADAPTERS = os.getenv("TRAINLOOM_LORA_DROPOUT_PACKED", "all")


@triton.jit
def _pack_kernel(mask_ptr, packed_ptr, n, BLOCK_BYTES: tl.constexpr):
    byte = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    elem = byte[:, None] * 8 + tl.arange(0, 8)[None, :]
    bits = tl.load(mask_ptr + elem, mask=elem < n, other=0).to(tl.int32)
    packed = tl.sum(bits << tl.arange(0, 8)[None, :], axis=1)
    tl.store(packed_ptr + byte, packed.to(tl.uint8), mask=byte * 8 < n)


@triton.jit
def _masked_scale_kernel(x_ptr, packed_ptr, out_ptr, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n
    x = tl.load(x_ptr + offs, mask=valid, other=0.0).to(tl.float32)
    byte = tl.load(packed_ptr + offs // 8, mask=valid, other=0).to(tl.int32)
    keep = ((byte >> (offs % 8)) & 1).to(tl.float32)
    tl.store(out_ptr + offs, (x * keep * scale).to(out_ptr.dtype.element_ty), mask=valid)


def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    n = mask.numel()
    packed = torch.empty((n + 7) // 8, dtype=torch.uint8, device=mask.device)
    _pack_kernel[(triton.cdiv(packed.numel(), BLOCK // 8),)](mask.view(torch.uint8), packed, n, BLOCK_BYTES=BLOCK // 8)
    return packed


def masked_scale(x: torch.Tensor, packed: torch.Tensor, scale: float) -> torch.Tensor:
    """native_dropout's output (or its backward's grad_input) for input/grad `x` and the packed mask."""
    x = x.contiguous()
    out = torch.empty_like(x)
    _masked_scale_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, packed, out, x.numel(), scale, BLOCK=BLOCK)
    return out


class _PackedDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, holder):
        out, mask = torch.native_dropout(x, p, True)
        holder["packed"] = ctx.packed = pack_mask(mask)
        ctx.scale = 1.0 / (1.0 - p)
        return out

    @staticmethod
    def backward(ctx, grad):
        return masked_scale(grad, ctx.packed, ctx.scale), None, None


class _Rebuild:
    """What linear_in keeps instead of the dropped input: the input and the packed mask."""

    __slots__ = ("x", "packed", "scale", "shape")

    def __init__(self, x, packed, scale, shape):
        self.x, self.packed, self.scale, self.shape = x, packed, scale, shape


def _supported(adapter: ParallelLinearAdapter) -> bool:
    return (
        adapter.training
        and torch.is_grad_enabled()
        and adapter.dropout is not None
        and adapter.dropout_position == "pre"
        and 0.0 < adapter.dropout.p < 1.0
        and not adapter.is_expert
        and adapter.norm_position not in ("pre", "post")
        and not adapter._sequence_parallel
        and not (adapter.config.cpu_offloading and adapter.config.cpu_offloading_activations)
    )


_stock_forward = ParallelLinearAdapter.forward


def forward(self, x):
    if not _supported(self) or (ADAPTERS == "dense" and not isinstance(self, LoraDenseAttentionAdapter)):
        return _stock_forward(self, x)
    p = self.dropout.p
    holder = {}
    dropped = _PackedDropout.apply(x, p, holder)
    if isinstance(self, LoraDenseAttentionAdapter):
        # autograd keeps the pack hook alive with each saved tensor, so it must not close over `dropped`
        dropped_ptr, dropped_numel = dropped.data_ptr(), dropped.numel()

        def pack(t):
            if t.data_ptr() == dropped_ptr and t.numel() == dropped_numel:
                return _Rebuild(x, holder["packed"], 1.0 / (1.0 - p), t.shape)
            return t

        def unpack(saved):
            if isinstance(saved, _Rebuild):
                return masked_scale(saved.x, saved.packed, saved.scale).view(saved.shape)
            return saved

        with saved_tensors_hooks(pack, unpack):
            h, _ = self.linear_in(dropped)
    else:
        h, _ = self.linear_in(dropped)
    h = self.activation(h)
    y, _ = self.linear_out(h)
    return y * (self.alpha / self.dim)


ParallelLinearAdapter.forward = forward


# ------------------------------------------------------------------ with fix #2 (hs_lora_epilogue)


@triton.jit
def _lora_dgrad_acc_packed_kernel(dgrad_ptr, ga_ptr, a_w_ptr, packed_ptr, M, N, scale,
                                  K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """hs_lora_epilogue._lora_dgrad_acc_kernel with the mask read from packed bits (same arithmetic)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, K)
    mm = rm < M
    mn = rn < N
    ga = tl.load(ga_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)  # [BM, K]
    aw = tl.load(a_w_ptr + rk[:, None] * N + rn[None, :], mask=mn[None, :], other=0.0)  # [K, BN]
    acc = tl.dot(ga, aw) * scale
    offs = rm[:, None] * N + rn[None, :]
    msk = mm[:, None] & mn[None, :]
    byte = tl.load(packed_ptr + offs // 8, mask=msk, other=0).to(tl.int32)
    acc = tl.where(((byte >> (offs % 8)) & 1) != 0, acc, 0.0)
    d = tl.load(dgrad_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    tl.store(dgrad_ptr + offs, (d + acc).to(tl.bfloat16), mask=msk)


_tls = threading.local()  # set by lora_apply for the adapter being run: does it rebuild its dropped input?


def _install_epilogue(hle):
    stock_dgrad_acc = hle.lora_dgrad_acc_
    stock_apply = hle.lora_apply

    def lora_dgrad_acc_(dgrad2d, ga, a_w, mask, scale):
        if mask is None or mask.dtype == torch.bool:
            return stock_dgrad_acc(dgrad2d, ga, a_w, mask, scale)
        M, N = dgrad2d.shape
        K = ga.shape[1]
        assert dgrad2d.is_contiguous() and ga.is_contiguous() and a_w.is_contiguous() and a_w.shape == (K, N)
        grid = (triton.cdiv(M, hle._BM), triton.cdiv(N, hle._BN))
        _lora_dgrad_acc_packed_kernel[grid](dgrad2d, ga, a_w, mask, M, N, float(scale),
                                            K=K, BM=hle._BM, BN=hle._BN, num_warps=hle._WARPS)
        return dgrad2d

    class _LoraDown(torch.autograd.Function):
        """hs_lora_epilogue._LoraDown keeping packed bits (and, for the dense adapter, its input) instead of the
        bool mask and the dropped input."""

        @staticmethod
        def forward(ctx, x2d, a_w, p, training, node):
            packed = None
            if training and p > 0.0:
                h, mask = torch.native_dropout(x2d, p, True)
                packed = pack_mask(mask)
            else:
                h = x2d
            a = torch.matmul(h, a_w.t())
            ctx.rebuild = packed is not None and getattr(_tls, "dense", False)
            ctx.save_for_backward(x2d if ctx.rebuild else h, a_w)
            ctx.packed, ctx.p, ctx.node = packed, p, node
            return a

        @staticmethod
        def backward(ctx, ga):
            saved, a_w = ctx.saved_tensors
            packed = ctx.packed
            scale = 1.0 / (1.0 - ctx.p) if packed is not None else 1.0
            h = masked_scale(saved, packed, scale).view_as(saved) if ctx.rebuild else saved
            ga = ga.contiguous()
            grad_a_w = torch.matmul(ga.t(), h)
            if not ctx.needs_input_grad[0]:
                return None, grad_a_w, None, None, None
            node, ctx.node = ctx.node, None
            if node is not None and hle.FLAGS.master and hle.FLAGS.bwd and not getattr(node, "_hs_done", False):
                node._hs_lora_pending = (ga, a_w, packed, scale)
                hle.STATS["bwd_folded"] += 1
                return None, grad_a_w, None, None, None
            hle.STATS["bwd_plain"] += 1
            gx = torch.matmul(ga, a_w)
            if packed is not None:
                gx = masked_scale(gx, packed, scale)
            return gx, grad_a_w, None, None, None

    def lora_apply(adapter, x, base, residual=None, base_kind=None):
        _tls.dense = isinstance(adapter, LoraDenseAttentionAdapter)
        try:
            return stock_apply(adapter, x, base, residual, base_kind)
        finally:
            _tls.dense = False

    hle._LoraDown, hle.lora_dgrad_acc_, hle.lora_apply = _LoraDown, lora_dgrad_acc_, lora_apply


if "hs_lora_epilogue" in sys.modules:
    _install_epilogue(sys.modules["hs_lora_epilogue"])
    WITH = "with hs_lora_epilogue"
else:
    WITH = "without hs_lora_epilogue (load it first to combine)"
print(f"[trainloom] lora_dropout_packed: packed dropout masks ({ADAPTERS} adapters), {WITH}", flush=True)
