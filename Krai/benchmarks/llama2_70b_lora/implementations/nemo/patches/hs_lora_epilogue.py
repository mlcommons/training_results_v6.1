# Copyright (c) 2026 Krai Ltd. All Rights Reserved.
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
# Portions of this file (the MCoreSelfAttentionMixin methods get_query_key_value_tensors and forward, re-implemented
# with the LoRA and residual adds folded) are adapted from NVIDIA NeMo,
# nemo/collections/nlp/modules/common/megatron/adapters/mcore_mixins.py,
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved. Licensed under the Apache License, Version 2.0.

"""Fold the elementwise passes around the attention LoRA adapters into GEMM epilogues (fix #2).

Loaded through the recipe's patch hook (HS_MOUNT_CODE=1 HS_PATCH_DIR=<this dir>
HS_PATCH_MODULES="hs_act_store hs_lora_epilogue"), before NeMo/Megatron are imported. Works on top of
hs_act_store (independent seams). Everything is off unless the module is loaded; HS_LORA_EPILOGUE=0 is a kill switch.

What the recipe does today, per layer and step (NeMo mcore_mixins.py MCoreSelfAttentionMixin + parallel_adapters.py
ParallelLinearAdapter, LoRA rank 16, alpha 32, dropout 0.1 "pre", TP=1; shapes in tokens x features, M = 8192):

  forward (80 layers + 11 recomputed = 91 layer-forwards)
    qkv   base = TE LayerNormLinear(h) -> (base [M,10240] bf16, ln_out [M,8192])      FP8 GEMM, frozen weight
          x = dropout(ln_out)              fused_dropout_kernel_vec   (writes x + bool mask)
          a = x @ A^T          [M,16]      hipBLASLt bf16 GEMM
          l = a @ B^T          [M,10240]   hipBLASLt bf16 GEMM, K=16 (memory bound: one write pass)
          l = l * 2.0                      AUnaryFunctor               (one read + one write pass)
          qkv = base + l                   CUDAFunctor_add             (two reads + one write)
    proj  same with [M,8192], then the TransformerLayer residual add  CUDAFunctor_add (hidden_dropout 0 -> plain add)
    mlp   residual add after fc2                                      CUDAFunctor_add
  backward (80 layers)
    g*2.0 on [M,10240] and [M,8192]       AUnaryFunctor (MulBackward0)
    proj  LoRA A dgrad  ga @ A [M,8192]   hipBLASLt K=16 (one write pass), then dropout backward masked_scale
          (read+write), then autograd accumulates it onto the base dgrad (CUDAFunctor_add, add_)
    qkv   same, the accumulate is TE's own `dgrad = dgrad + grad_outputs[1]` in _LayerNormLinear.backward
    residual-stream grad accumulation at the attention and MLP residuals (autograd add_)   2 x CUDAFunctor_add
  => CUDAFunctor_add<BFloat16>: 4 x 91 + 4 x 80 = 684 (measured 682, 74 ms/step); AUnaryFunctor: 2 x 91 + 2 x 80
     = 342 (measured 342, 26 ms/step); fused_dropout 2 x 91 = 182 (20 ms); masked_scale 2 x 80 = 160 (13 ms).
     The "cat" kernels (308/step, 15 ms) are not LoRA: 80 are SplitAlongDim's backward (dq,dk,dv -> d(mixed_qkv))
     and the rest are tiny FP8 amax bookkeeping cats. The fused LoraKQVAdapter has no cat.

Items (each an env switch, default on when the module is loaded):
  HS_LORA_EPILOGUE_ADD=1       LoRA up-projection with the epilogue  out = base + (alpha/r) * a @ B^T  in one
                               kernel (Triton, K=16 tl.dot, one read + one write pass): removes the K=16 GEMM
                               write, the scalar multiply and the add in the forward, and the g*2 multiply in the
                               backward (the scale becomes the alpha of the two rank-16 backward GEMMs; exact for
                               power-of-two alpha/r). HS_LORA_EPILOGUE_IMPL=blas would use hipBLASLt instead
                               (base.addmm_(a, B^T, alpha) in place, beta=1 epilogue; no residual fold), but TE
                               returns its outputs as views of a multi-output autograd function, which cannot be
                               modified in place, so with TE linears it falls back to the Triton kernel.
  HS_LORA_EPILOGUE_RESIDUAL=1  (needs ADD, Triton impl) the attention residual add is folded into the same kernel
                               at proj: out = residual + base + scale * a @ B^T; the TransformerLayer's
                               bias_dropout_add then passes the tensor through (hidden_dropout must be 0,
                               bias_dropout_fusion False). The post-MLP residual is NOT folded: fc2 is a TE FP8
                               GEMM whose only epilogue option here is beta=1 with C == D, which needs a copy of
                               the residual first (copy ~= the add it replaces); hipBLASLt C != D is not exposed by
                               TE or torch (torch.addmm(out=) copies).
  HS_LORA_EPILOGUE_BWD=1       LoRA A dgrad folded into the base dgrad: one Triton kernel does
                               dgrad += mask * (1/(1-p)) * ga @ A in place on the base GEMM's dgrad, replacing the
                               K=16 GEMM write, masked_scale and the accumulate add. The LoRA branch returns no
                               input gradient to autograd; instead it stashes (ga, A, mask) on the base linear's
                               autograd node, and wrappers around TE _Linear.backward / _LayerNormLinear.backward
                               apply it (for LayerNormLinear just before the RMSNorm backward, where TE adds
                               grad_outputs[1] today).
  HS_LORA_EPILOGUE_CAT         accepted for symmetry, no-op: there is no cat in the fused-QKV LoRA path (see above).

Numerics (see test_hs_lora_epilogue.py): ADD/RESIDUAL/BWD outputs differ from the current path by rounding only
(one BF16 rounding of the fp32 sum instead of 2-3 successive BF16 roundings; <= 1 BF16 ulp on production shapes),
so they need an L2 check; the gradients of A and B are bit-identical to the current path for power-of-two alpha/r.
Dropout masks are unchanged (same torch.native_dropout call, same RNG consumption).
"""

import hashlib
import inspect
import os
import threading

import torch
import triton
import triton.language as tl


def _env(name, default="1"):
    return os.getenv(name, default) == "1"


class Flags:
    """Read once from the env; the unit test toggles them per configuration."""

    master = _env("HS_LORA_EPILOGUE")
    add = _env("HS_LORA_EPILOGUE_ADD")
    residual = _env("HS_LORA_EPILOGUE_RESIDUAL")
    bwd = _env("HS_LORA_EPILOGUE_BWD")
    cat = _env("HS_LORA_EPILOGUE_CAT")  # no-op, documented above
    impl = os.getenv("HS_LORA_EPILOGUE_IMPL", "triton")  # triton | blas
    verbose = _env("HS_LORA_EPILOGUE_VERBOSE", "0")
    rope = _env("HS_ROPE_FUSED", "0")  # fix 4: fused qkv split + RoPE (hs_rope_qkv.py), default off
    fast = _env("HS_LORA_EPILOGUE_FAST", "0")  # fix 14: restructured lora_up / lora_dgrad_acc_ kernels, default off


FLAGS = Flags()
_tls = threading.local()  # pending LoRA dgrad fold between _LayerNormLinear.backward and the norm backward
_tls.pending = None
STATS = {"up_fused": 0, "up_blas": 0, "up_plain": 0, "bwd_folded": 0, "bwd_plain": 0, "residual_folded": 0,
         "residual_passthrough": 0, "fallback_adapter": 0, "rope_fused": 0, "rope_unfused": 0, "up_fast": 0,
         "dgrad_fast": 0}


# ------------------------------------------------------------------------------------------------- Triton kernels


@triton.jit
def _lora_up_kernel(base_ptr, a_ptr, b_ptr, res_ptr, out_ptr, M, N, scale,
                    HAS_RES: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """out[m, n] = base[m, n] (+ res[m, n]) + scale * sum_k a[m, k] * b[n, k]; all row-major contiguous, bf16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, K)
    mm = rm < M
    mn = rn < N
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)  # [BM, K]
    b = tl.load(b_ptr + rn[None, :] * K + rk[:, None], mask=mn[None, :], other=0.0)  # [K, BN]
    acc = tl.dot(a, b) * scale  # fp32
    offs = rm[:, None] * N + rn[None, :]
    msk = mm[:, None] & mn[None, :]
    acc += tl.load(base_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    if HAS_RES:
        acc += tl.load(res_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=msk)


@triton.jit
def _lora_dgrad_acc_kernel(dgrad_ptr, ga_ptr, a_w_ptr, mask_ptr, M, N, scale,
                           HAS_MASK: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """dgrad[m, n] += mask[m, n] * scale * sum_k ga[m, k] * A[k, n], in place. A is [K, N] row-major (Megatron
    weight layout [out=rank, in]); mask is the bool dropout mask viewed as uint8 (scale = 1/(1-p)), or absent."""
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
    if HAS_MASK:
        keep = tl.load(mask_ptr + offs, mask=msk, other=0)
        acc = tl.where(keep != 0, acc, 0.0)
    d = tl.load(dgrad_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    tl.store(dgrad_ptr + offs, (d + acc).to(tl.bfloat16), mask=msk)


# Fix 14 (microbenchmark lane l7_fix2_tune, campaign kfix_261010): the same arithmetic restructured for bandwidth,
# bit-identical outputs. (1) the rank-16 product as a transposed dot, tl.trans(tl.dot(B^T, a^T)): the MFMA result
# layout then gives 16-byte stores; (2) a 1-D grid with the N tile fastest, so neighbouring workgroups stream
# neighbouring memory; (3) .cg loads (global_load ... nt, bypass L1) on the [M, N] streams; (4) unmasked tiles when
# the shape divides the tile. Measured 0.49x (lora_up) and 0.63x (lora_dgrad_acc_) of the kernels above.
@triton.jit
def _lora_up_fast_kernel(base_ptr, a_ptr, b_ptr, res_ptr, out_ptr, M, N, scale,
                         HAS_RES: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         EVEN: tl.constexpr):
    nn = tl.cdiv(N, BN)
    pid_m = tl.program_id(0) // nn
    pid_n = tl.program_id(0) % nn
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, K)
    mm = rm < M
    mn = rn < N
    at = tl.load(a_ptr + rm[None, :] * K + rk[:, None], mask=mm[None, :], other=0.0)  # [K, BM]
    bt = tl.load(b_ptr + rn[:, None] * K + rk[None, :], mask=mn[:, None], other=0.0)  # [BN, K]
    acc = tl.trans(tl.dot(bt, at)) * scale
    offs = rm[:, None] * N + rn[None, :]
    if EVEN:
        acc += tl.load(base_ptr + offs, cache_modifier=".cg").to(tl.float32)
        if HAS_RES:
            acc += tl.load(res_ptr + offs, cache_modifier=".cg").to(tl.float32)
        tl.store(out_ptr + offs, acc.to(tl.bfloat16))
    else:
        msk = mm[:, None] & mn[None, :]
        acc += tl.load(base_ptr + offs, mask=msk, other=0.0, cache_modifier=".cg").to(tl.float32)
        if HAS_RES:
            acc += tl.load(res_ptr + offs, mask=msk, other=0.0, cache_modifier=".cg").to(tl.float32)
        tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=msk)


@triton.jit
def _lora_dgrad_acc_fast_kernel(dgrad_ptr, ga_ptr, a_w_ptr, mask_ptr, M, N, scale,
                                HAS_MASK: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                                EVEN: tl.constexpr):
    nn = tl.cdiv(N, BN)
    pid_m = tl.program_id(0) // nn
    pid_n = tl.program_id(0) % nn
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, K)
    mm = rm < M
    mn = rn < N
    gat = tl.load(ga_ptr + rm[None, :] * K + rk[:, None], mask=mm[None, :], other=0.0)  # [K, BM]
    awt = tl.load(a_w_ptr + rk[None, :] * N + rn[:, None], mask=mn[:, None], other=0.0)  # [BN, K]
    acc = tl.trans(tl.dot(awt, gat)) * scale
    offs = rm[:, None] * N + rn[None, :]
    if EVEN:
        if HAS_MASK:
            keep = tl.load(mask_ptr + offs, cache_modifier=".cg")
            acc = tl.where(keep != 0, acc, 0.0)
        d = tl.load(dgrad_ptr + offs, cache_modifier=".cg").to(tl.float32)
        tl.store(dgrad_ptr + offs, (d + acc).to(tl.bfloat16))
    else:
        msk = mm[:, None] & mn[None, :]
        if HAS_MASK:
            keep = tl.load(mask_ptr + offs, mask=msk, other=0, cache_modifier=".cg")
            acc = tl.where(keep != 0, acc, 0.0)
        d = tl.load(dgrad_ptr + offs, mask=msk, other=0.0, cache_modifier=".cg").to(tl.float32)
        tl.store(dgrad_ptr + offs, (d + acc).to(tl.bfloat16), mask=msk)


_FBM, _FBN, _FWARPS = 32, 128, 4   # lane l7 sweet spot; matrix_instr_nonkdim 16, never kpack 2 (not bit-exact)


# tile sweep on one MI300X (test_hs_lora_epilogue + scratch sweep, 2026-10-09): 128x128 / 8 warps is best or
# within noise of best for both production shapes (the [8192x8192] kernels are then at parity with a plain add)
_BM, _BN, _WARPS = (int(os.getenv("HS_LORA_EPILOGUE_BM", "128")), int(os.getenv("HS_LORA_EPILOGUE_BN", "128")),
                    int(os.getenv("HS_LORA_EPILOGUE_WARPS", "8")))


def _rank_ok(k):
    return k >= 16 and (k & (k - 1)) == 0 and k <= 128


def lora_up(base2d, a, b, scale, res2d=None, out=None):
    """base + scale * a @ b^T (+ res), one pass. base2d [M,N], a [M,K], b [N,K] (torch Linear weight layout)."""
    M, N = base2d.shape
    K = a.shape[1]
    assert base2d.is_contiguous() and a.is_contiguous() and b.is_contiguous() and _rank_ok(K)
    assert base2d.dtype == a.dtype == b.dtype == torch.bfloat16
    if out is None:
        out = torch.empty_like(base2d)
    has_res = res2d is not None
    if FLAGS.fast:
        STATS["up_fast"] += 1
        _lora_up_fast_kernel[(triton.cdiv(M, _FBM) * triton.cdiv(N, _FBN),)](
            base2d, a, b, res2d if has_res else base2d, out, M, N, float(scale), HAS_RES=has_res, K=K, BM=_FBM,
            BN=_FBN, EVEN=(M % _FBM == 0 and N % _FBN == 0), num_warps=_FWARPS, matrix_instr_nonkdim=16)
        return out
    grid = (triton.cdiv(M, _BM), triton.cdiv(N, _BN))
    _lora_up_kernel[grid](base2d, a, b, res2d if has_res else base2d, out, M, N, float(scale),
                          HAS_RES=has_res, K=K, BM=_BM, BN=_BN, num_warps=_WARPS)
    return out


def lora_dgrad_acc_(dgrad2d, ga, a_w, mask2d, scale):
    """dgrad += mask * scale * ga @ A in place. dgrad2d [M,N], ga [M,K], a_w [K,N], mask2d bool [M,N] or None."""
    M, N = dgrad2d.shape
    K = ga.shape[1]
    assert dgrad2d.is_contiguous() and ga.is_contiguous() and a_w.is_contiguous() and _rank_ok(K)
    assert a_w.shape == (K, N) and dgrad2d.dtype == ga.dtype == a_w.dtype == torch.bfloat16
    has_mask = mask2d is not None
    if has_mask:
        assert mask2d.dtype == torch.bool and mask2d.is_contiguous() and mask2d.shape == (M, N)
        mask2d = mask2d.view(torch.uint8)
    if FLAGS.fast:
        STATS["dgrad_fast"] += 1
        _lora_dgrad_acc_fast_kernel[(triton.cdiv(M, _FBM) * triton.cdiv(N, _FBN),)](
            dgrad2d, ga, a_w, mask2d if has_mask else dgrad2d, M, N, float(scale), HAS_MASK=has_mask, K=K, BM=_FBM,
            BN=_FBN, EVEN=(M % _FBM == 0 and N % _FBN == 0), num_warps=_FWARPS, matrix_instr_nonkdim=16)
        return dgrad2d
    grid = (triton.cdiv(M, _BM), triton.cdiv(N, _BN))
    _lora_dgrad_acc_kernel[grid](dgrad2d, ga, a_w, mask2d if has_mask else dgrad2d, M, N, float(scale),
                                 HAS_MASK=has_mask, K=K, BM=_BM, BN=_BN, num_warps=_WARPS)
    return dgrad2d


# ------------------------------------------------------------------------------------------- autograd functions

_VIEW_NODES = ("ViewBackward0", "UnsafeViewBackward0", "ReshapeAliasBackward0", "AliasBackward0", "TBackward0")


def find_base_node(t, kind, max_depth=4):
    """The TE autograd node (`_LinearBackward` / `_LayerNormLinearBackward`) that produced tensor t, through views."""
    node = t.grad_fn if isinstance(t, torch.Tensor) else None
    for _ in range(max_depth):
        if node is None:
            return None
        name = type(node).__name__
        if name == kind:
            return node
        if name not in _VIEW_NODES or not node.next_functions:
            return None
        node = node.next_functions[0][0]
    return None


class _LoraDown(torch.autograd.Function):
    """a = dropout(x) @ A^T. Backward: grad_A as Megatron's linear does it; grad_x either materialized
    (ga @ A, dropout backward) or - when a fold target node is given - stashed on that node and applied in place
    on the base GEMM's dgrad by the TE backward wrappers below (no input gradient is returned to autograd)."""

    @staticmethod
    def forward(ctx, x2d, a_w, p, training, node):
        if training and p > 0.0:
            h, mask = torch.native_dropout(x2d, p, True)  # same op/RNG consumption as nn.Dropout in the adapter
        else:
            h, mask = x2d, None
        a = torch.matmul(h, a_w.t())  # [M, K]; the op Megatron's ColumnParallelLinear runs
        ctx.save_for_backward(h, a_w, mask)
        ctx.p = p
        ctx.node = node
        return a

    @staticmethod
    def backward(ctx, ga):
        h, a_w, mask = ctx.saved_tensors
        ga = ga.contiguous()
        grad_a_w = torch.matmul(ga.t(), h)  # [K, in], as Megatron: grad_output.t().matmul(total_input)
        if not ctx.needs_input_grad[0]:
            return None, grad_a_w, None, None, None
        node = ctx.node
        ctx.node = None
        scale = 1.0 / (1.0 - ctx.p) if mask is not None else 1.0
        if node is not None and FLAGS.master and FLAGS.bwd and not getattr(node, "_hs_done", False):
            node._hs_lora_pending = (ga, a_w, mask, scale)
            STATS["bwd_folded"] += 1
            return None, grad_a_w, None, None, None
        STATS["bwd_plain"] += 1
        gx = torch.matmul(ga, a_w)  # [M, in]
        if mask is not None:
            gx = torch.ops.aten.native_dropout_backward(gx, mask, scale)
        return gx, grad_a_w, None, None, None


def _scaled_mm(x, y, scale):
    """scale * (x @ y) with the scale in the hipBLASLt epilogue (alpha), one kernel: beta=0 ignores `out`."""
    if scale == 1.0:
        return torch.matmul(x, y)
    out = torch.empty((x.shape[0], y.shape[1]), dtype=x.dtype, device=x.device)
    return torch.addmm(out, x, y, beta=0.0, alpha=scale, out=out)


class _LoraUp(torch.autograd.Function):
    """out = base (+ residual) + scale * a @ B^T in one Triton pass. Backward: identity to base/residual, the two
    rank-16 GEMMs for a and B (the scale applied to their small results; exact for power-of-two scale)."""

    @staticmethod
    def forward(ctx, base, a, b_w, scale, residual):
        base2d = base.view(-1, base.shape[-1])
        res2d = residual.view(-1, residual.shape[-1]) if residual is not None else None
        out = torch.empty_like(base)
        lora_up(base2d, a, b_w, scale, res2d, out=out.view(-1, base.shape[-1]))
        ctx.save_for_backward(a, b_w)
        ctx.scale = scale
        ctx.has_res = residual is not None
        return out

    @staticmethod
    def backward(ctx, g):
        a, b_w = ctx.saved_tensors
        g2d = g.reshape(-1, g.shape[-1])
        ga = _scaled_mm(g2d, b_w, ctx.scale)  # [M, K]   (Megatron: (scale * grad_output).matmul(weight))
        gb = _scaled_mm(g2d.t(), a, ctx.scale)  # [N, K]  (Megatron: (scale * grad_output).t().matmul(total_input))
        return g, ga, gb, None, (g if ctx.has_res else None)


class _LoraUpBlas(torch.autograd.Function):
    """hipBLASLt variant: base.addmm_(a, B^T, alpha=scale) in place (beta = 1 epilogue). No residual fold."""

    @staticmethod
    def forward(ctx, base, a, b_w, scale):
        base.view(-1, base.shape[-1]).addmm_(a, b_w.t(), alpha=scale)
        ctx.mark_dirty(base)
        ctx.save_for_backward(a, b_w)
        ctx.scale = scale
        return base

    @staticmethod
    def backward(ctx, g):
        a, b_w = ctx.saved_tensors
        g2d = g.reshape(-1, g.shape[-1])
        ga = _scaled_mm(g2d, b_w, ctx.scale)
        gb = _scaled_mm(g2d.t(), a, ctx.scale)
        return g, ga, gb, None


def adapter_foldable(adapter):
    """The NeMo ParallelLinearAdapter configurations this module replaces (the recipe's: LoRA on attention, identity
    bottleneck, no norm, dropout 'pre', TP=1, no SP, no expert, no CPU offload, no grad-accumulation fusion)."""
    try:
        lin_in, lin_out = adapter.linear_in, adapter.linear_out
        return (
            type(adapter.activation).__name__ == "Identity"
            and adapter.norm_position is None
            and adapter.layer_norm is None
            and (adapter.dropout is None or adapter.dropout_position == "pre")
            and not adapter._sequence_parallel
            and not getattr(adapter, "is_expert", False)
            and not adapter.config.cpu_offloading
            and lin_in.weight.dim() == 2 and lin_out.weight.dim() == 2
            and lin_in.weight.shape[0] == lin_out.weight.shape[1]
            and _rank_ok(lin_in.weight.shape[0])
            and not lin_in.gradient_accumulation_fusion and not lin_out.gradient_accumulation_fusion
            and not getattr(lin_in, "allreduce_dgrad", False) and not getattr(lin_out, "allreduce_dgrad", False)
            and not getattr(lin_in, "explicit_expert_comm", False)
            and not getattr(lin_out, "explicit_expert_comm", False)
            and lin_in.weight.dtype == torch.bfloat16 and lin_out.weight.dtype == torch.bfloat16
            and _tp_world_size() == 1
        )
    except AttributeError:
        return False


def _tp_world_size():
    try:
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_world_size()
    except Exception:  # noqa: BLE001  (parallel state not initialised -> single GPU test)
        return 1


def lora_apply(adapter, x, base, residual=None, base_kind=None):
    """base + adapter(x) [+ residual] with the folds enabled by FLAGS. x [.., in], base [.., out] (TE output),
    base_kind the TE autograd node class name that produced `base` (for the dgrad fold)."""
    in_f = x.shape[-1]
    x2d = x.reshape(-1, in_f)
    a_w = adapter.linear_in.weight
    b_w = adapter.linear_out.weight
    scale = float(adapter.alpha / adapter.dim)
    drop = adapter.dropout
    p = float(drop.p) if drop is not None else 0.0
    training = bool(drop.training) if drop is not None else False
    node = None
    if FLAGS.master and FLAGS.bwd and torch.is_grad_enabled() and x.requires_grad and base_kind is not None:
        src = x if base_kind == "_LayerNormLinearBackward" else base
        node = find_base_node(src, base_kind)
        if node is not None:
            if base_kind == "_LayerNormLinearBackward":
                ok = (node.return_layernorm_output and not node.return_layernorm_output_gathered
                      and not node.requires_wgrad and node.normalization in ("RMSNorm", "LayerNorm")
                      and not node.ub_bulk_wgrad and not node.ub_overlap_rs_dgrad and _HOOKED["lnl"])
                if ok:
                    node.set_materialize_grads(False)  # our None for ln_out must not become a zeros tensor
            else:
                ok = bool(node.requires_dgrad) and not node.ub_bulk_wgrad and not node.ub_overlap_rs_dgrad \
                    and _HOOKED["linear"]
            if not ok:
                node = None
    a = _LoraDown.apply(x2d, a_w, p, training, node)
    # hipBLASLt in-place variant: not applicable to TE outputs (they are views of a multi-output autograd function,
    # which autograd refuses to modify in place) -> falls back to the Triton kernel there
    if FLAGS.master and FLAGS.add and FLAGS.impl == "blas" and not (base.is_leaf and base.requires_grad) \
            and not base._is_view():
        STATS["up_blas"] += 1
        return _LoraUpBlas.apply(base, a, b_w, scale), False
    if FLAGS.master and FLAGS.add:
        res = residual if (FLAGS.residual and residual is not None) else None
        if res is not None and (res.shape != base.shape or res.dtype != base.dtype or not res.is_contiguous()):
            res = None
        STATS["up_fused"] += 1
        if res is not None:
            STATS["residual_folded"] += 1
        return _LoraUp.apply(base, a, b_w, scale, res), res is not None
    STATS["up_plain"] += 1
    lora = torch.matmul(a, b_w.t()).view_as(base) * scale  # the current order of ops
    return base + lora, False


# ------------------------------------------------------------------------------------------ TE backward wrappers

_HOOKED = {"lnl": False, "linear": False, "mixin": False, "bda": False}


def _install_te_hooks():
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.module import layernorm_linear as lnl_mod
    from transformer_engine.pytorch.module import linear as lin_mod

    # --- _Linear: apply the stash on the returned dgrad (output index 1: (wgrad, dgrad, grad_bias, ...))
    _Linear = lin_mod._Linear
    orig_lin_bwd = _Linear.backward

    def linear_backward(ctx, *grad_outputs):
        ctx._hs_done = True
        outs = orig_lin_bwd(ctx, *grad_outputs)
        pending = ctx.__dict__.pop("_hs_lora_pending", None)
        if pending is not None:
            dgrad = outs[1]
            assert dgrad is not None, "LoRA dgrad fold target has no dgrad"
            ga, a_w, mask, scale = pending
            lora_dgrad_acc_(dgrad.view(-1, dgrad.shape[-1]), ga, a_w, mask, scale)
        return outs

    _Linear.backward = staticmethod(linear_backward)
    _HOOKED["linear"] = True

    # --- _LayerNormLinear: skip TE's `dgrad = dgrad + grad_outputs[1]`, apply the stash right before the norm
    # backward (the dgrad passed to it is the GEMM output, the point where TE adds grad_outputs[1] today).
    _LNL = lnl_mod._LayerNormLinear
    orig_lnl_bwd = _LNL.backward

    def lnl_backward(ctx, *grad_outputs):
        ctx._hs_done = True
        pending = ctx.__dict__.pop("_hs_lora_pending", None)
        if pending is None:
            return orig_lnl_bwd(ctx, *grad_outputs)
        assert ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered
        ctx.return_layernorm_output = False
        _tls.pending = pending
        try:
            return orig_lnl_bwd(ctx, grad_outputs[0], None)
        finally:
            ctx.return_layernorm_output = True
            left, _tls.pending = _tls.pending, None
            assert left is None, "LoRA dgrad fold was not consumed by the norm backward"

    _LNL.backward = staticmethod(lnl_backward)

    def hook_norm_bwd(fn):
        def wrapped(dgrad, *args, **kwargs):
            pending = getattr(_tls, "pending", None)
            if pending is not None:
                _tls.pending = None
                ga, a_w, mask, scale = pending
                lora_dgrad_acc_(dgrad.view(-1, dgrad.shape[-1]), ga, a_w, mask, scale)
            return fn(dgrad, *args, **kwargs)

        wrapped._hs_orig = fn
        return wrapped

    for name in ("te_rmsnorm_bwd_triton", "te_layernorm_bwd_triton"):
        if hasattr(lnl_mod, name):
            setattr(lnl_mod, name, hook_norm_bwd(getattr(lnl_mod, name)))
    tex.rmsnorm_bwd = hook_norm_bwd(tex.rmsnorm_bwd)
    tex.layernorm_bwd = hook_norm_bwd(tex.layernorm_bwd)
    _HOOKED["lnl"] = True


# ------------------------------------------------------------------------- NeMo mixin + Megatron bias_dropout_add

# md5 of inspect.getsource of the functions this module re-implements (NeMo 2.3 / Megatron-core 0.10 in the
# v51-hs image); on a mismatch the mixin patch is not installed (the TE hooks are harmless without it).
_EXPECTED_SRC = {
    "get_query_key_value_tensors": "15d6b57a6c0156cdcc6ba60c28ededbf",
    "forward": "066ea2d79e72283c1e2d5bce27e5fc3a",
    "_bias_dropout_add_func": "649bbe4f84a7cee5ab71be3bd18ccddf",
}


def _src_md5(fn):
    return hashlib.md5(inspect.getsource(fn).encode()).hexdigest()


def _install_bda_patch():
    import megatron.core.fusions.fused_bias_dropout as fbd

    orig = fbd._bias_dropout_add_func

    def _bias_dropout_add_func(x_with_bias, residual, prob, training):
        x, bias = x_with_bias
        if getattr(x, "_hs_residual_folded", False):
            assert bias is None and prob == 0.0
            STATS["residual_passthrough"] += 1
            return x
        return orig(x_with_bias, residual, prob, training)

    fbd._bias_dropout_add_func = _bias_dropout_add_func  # looked up by name inside bias_dropout_add_unfused
    _HOOKED["bda"] = True


def _install_mixin_patch():
    from megatron.core.extensions.transformer_engine import SplitAlongDim
    from nemo.collections.nlp.modules.common.megatron.adapters import mcore_mixins as mm
    from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import AdapterName
    from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
    import transformer_engine.pytorch as te

    Mixin = mm.MCoreSelfAttentionMixin
    orig_qkv = Mixin.get_query_key_value_tensors
    orig_fwd = Mixin.forward

    def _mixed_qkv(self, hidden_states):
        """linear_qkv + the (folded) LoRA add: mixed_qkv [s, b, 10240] and the layernorm output."""
        linear_qkv_output, _ = self.linear_qkv(hidden_states)
        layernorm_output = None
        if isinstance(linear_qkv_output, tuple):
            if len(linear_qkv_output) == 2:
                mixed_qkv, layernorm_output = linear_qkv_output
            else:
                raise ValueError(f"Unexpected number of outputs from linear_qkv output: {len(linear_qkv_output)}")
        else:
            mixed_qkv = linear_qkv_output

        if self.is_adapter_available():
            lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
            if lora_kqv_adapter and self.adapter_cfg[AdapterName.LORA_KQV_ADAPTER]['enabled']:
                x = layernorm_output if layernorm_output is not None else hidden_states
                if FLAGS.master and adapter_foldable(lora_kqv_adapter):
                    mixed_qkv, _ = lora_apply(lora_kqv_adapter, x, mixed_qkv, None, "_LayerNormLinearBackward")
                else:
                    STATS["fallback_adapter"] += 1
                    mixed_qkv = mixed_qkv + lora_kqv_adapter(x)
        return mixed_qkv, layernorm_output

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """NeMo's method with the LoRA add folded (lora_apply)."""
        mixed_qkv, layernorm_output = _mixed_qkv(self, hidden_states)

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        if SplitAlongDim is not None:
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.is_adapter_available():
            key_infused_adapter = self.get_adapter_module(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_adapter_module(AdapterName.VALUE_INFUSED)
            if key_infused_adapter and self.adapter_cfg[AdapterName.KEY_INFUSED]['enabled']:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key.shape
                key = key_infused_adapter(key.reshape(kls[0], kls[1], -1)).reshape(kls).to(query.dtype)
            if value_infused_adapter and self.adapter_cfg[AdapterName.VALUE_INFUSED]['enabled']:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value.shape
                value = value_infused_adapter(value.reshape(vls[0], vls[1], -1)).reshape(vls).to(query.dtype)
            lora_unfused_kqv_adapter = self.get_adapter_module(AdapterName.LORA_UNFUSED_KQV_ADAPTER)
            if lora_unfused_kqv_adapter and self.adapter_cfg[AdapterName.LORA_UNFUSED_KQV_ADAPTER]['enabled']:
                assert lora_kqv_adapter is None
                if layernorm_output is not None:
                    lq, lk, lv = lora_unfused_kqv_adapter(layernorm_output)
                else:
                    lq, lk, lv = lora_unfused_kqv_adapter(hidden_states)
                query, key, value = query + lq, key + lk, value + lv
        return query, key, value

    def forward(self, hidden_states, attention_mask, key_value_states=None, inference_params=None,
                rotary_pos_emb=None, packed_seq_params=None, rotary_pos_cos=None, rotary_pos_sin=None,
                attention_bias=None, sequence_len_offset=None):
        """NeMo's method with the proj LoRA add (and the attention residual add) folded."""
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2
        if _rope_fusable(self, hidden_states, key_value_states, inference_params, packed_seq_params, rotary_pos_emb):
            # fix 4: one kernel from mixed_qkv to rotated q, rotated k and v (contiguous sbhd); replaces the split,
            # the q reshape copy, apex RoPE on q and k and TE's v .contiguous(); backward writes d_mixed directly
            from hs_rope_qkv import fused_qkv_rope
            mixed_qkv, _ = _mixed_qkv(self, hidden_states)
            query, key, value = fused_qkv_rope(mixed_qkv, rotary_pos_emb[0])
            STATS["rope_fused"] += 1
            rotary_pos_emb = None
        else:
            if FLAGS.rope and rotary_pos_emb is not None:
                STATS["rope_unfused"] += 1
            query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)
        if self.checkpoint_core_attention:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, attn_mask_type=attn_mask_type, packed_seq_params=packed_seq_params
            )
        else:
            core_attn_out = self.core_attention(
                query, key, value, attention_mask, attn_mask_type=attn_mask_type, packed_seq_params=packed_seq_params
            )
        if packed_seq_params is not None:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_attn_out)
        if self.is_adapter_available():
            lora_linear_proj_adapter = self.get_adapter_module(AdapterName.LORA_DENSE_ATTENTION_ADAPTER)
            if lora_linear_proj_adapter and self.adapter_cfg[AdapterName.LORA_DENSE_ATTENTION_ADAPTER]['enabled']:
                if FLAGS.master and adapter_foldable(lora_linear_proj_adapter):
                    # the residual of the TransformerLayer is its input, which is what this module receives when the
                    # norm is fused into linear_qkv (input_layernorm is an IdentityOp)
                    residual = None
                    if (FLAGS.add and FLAGS.residual and FLAGS.impl != "blas" and bias is None and _HOOKED["bda"]
                            and isinstance(self.linear_qkv, te.LayerNormLinear)
                            and float(self.config.hidden_dropout) == 0.0 and not self.config.bias_dropout_fusion):
                        residual = hidden_states
                    output, folded = lora_apply(lora_linear_proj_adapter, core_attn_out, output, residual,
                                                "_LinearBackward")
                    if folded:
                        output._hs_residual_folded = True
                else:
                    STATS["fallback_adapter"] += 1
                    output = output + lora_linear_proj_adapter(core_attn_out)
        return output, bias

    def _adapter_on(self, name):
        if not self.is_adapter_available():
            return False
        ad = self.get_adapter_module(name)
        return bool(ad) and bool(self.adapter_cfg[name]['enabled'])

    def _rope_fusable(self, hidden_states, key_value_states, inference_params, packed_seq_params, rotary_pos_emb):
        """Fix 4 applies only where the fused kernels reproduce Megatron's path exactly: apex fused RoPE on bshd
        (apply_rope_fusion, not interleaved, no cu_seqlens), batch 1, no context parallelism, no inference cache,
        Llama 2 70B head layout (64 q heads in 8 groups of dim 128), one shared fp32 freqs [s,1,1,128] for q and k,
        and no adapter that edits q/k/v after the split."""
        if not FLAGS.rope or rotary_pos_emb is None or key_value_states is not None:
            return False
        if inference_params is not None or packed_seq_params is not None:
            return False
        cfg = self.config
        if not cfg.apply_rope_fusion or getattr(cfg, "rotary_interleaved", False) \
                or getattr(cfg, "context_parallel_size", 1) != 1 or getattr(cfg, "multi_latent_attention", False):
            return False
        if (self.num_attention_heads_per_partition, self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head) != (64, 8, 128):
            return False
        q_pos, k_pos = rotary_pos_emb
        if q_pos is not k_pos or q_pos.dtype != torch.float32 or not q_pos.is_contiguous():
            return False
        S = hidden_states.shape[0]
        if hidden_states.dim() != 3 or hidden_states.shape[1] != 1 or tuple(q_pos.shape) != (S, 1, 1, 128):
            return False
        return not any(_adapter_on(self, n) for n in (AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED,
                                                     AdapterName.LORA_UNFUSED_KQV_ADAPTER))

    Mixin.get_query_key_value_tensors = get_query_key_value_tensors
    Mixin.forward = forward
    Mixin._hs_orig_get_query_key_value_tensors = orig_qkv
    Mixin._hs_orig_forward = orig_fwd
    _HOOKED["mixin"] = True


def _sources_match():
    """Hash the NeMo/Megatron functions this module re-implements before anything is patched."""
    import megatron.core.fusions.fused_bias_dropout as fbd
    from nemo.collections.nlp.modules.common.megatron.adapters import mcore_mixins as mm

    found = {"get_query_key_value_tensors": _src_md5(mm.MCoreSelfAttentionMixin.get_query_key_value_tensors),
             "forward": _src_md5(mm.MCoreSelfAttentionMixin.forward),
             "_bias_dropout_add_func": _src_md5(fbd._bias_dropout_add_func)}
    bad = {k: v for k, v in found.items() if _EXPECTED_SRC.get(k) != v}
    if bad:
        print(f"[hyperspindle] hs_lora_epilogue: source mismatch {bad}; mixin patch NOT installed", flush=True)
    return not bad


def install():
    if not FLAGS.master:
        print("[hyperspindle] hs_lora_epilogue: HS_LORA_EPILOGUE=0, nothing installed", flush=True)
        return
    ok = _sources_match()
    _install_te_hooks()
    if ok:
        _install_bda_patch()
        _install_mixin_patch()
    print(f"[hyperspindle] hs_lora_epilogue: add={'on' if FLAGS.add else 'off'} ({FLAGS.impl}) "
          f"residual={'on' if FLAGS.residual and FLAGS.add and FLAGS.impl != 'blas' else 'off'} "
          f"bwd={'on' if FLAGS.bwd else 'off'} cat=n/a rope(fix 4)={'on' if FLAGS.rope else 'off'} "
          f"fast kernels(fix 14)={'on' if FLAGS.fast else 'off'}; hooks {_HOOKED}",
          flush=True)


install()
