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
# Portions of this file (the Triton RMSNorm backward kernel, reproduced with an added epilogue) are adapted from ROCm
# Transformer Engine, transformer_engine/pytorch/triton_kernels/rmsnorm.py, Copyright (c) 2025, Advanced Micro
# Devices, Inc. All rights reserved. AMD's contributions to Transformer Engine are licensed under the MIT License
# (see NOTICE).

"""Fix 8b: the residual-stream gradient accumulation and the e5m2 grad cast folded into the RMSNorm backward
(env HS_RESGRAD_FOLD=1; load through HS_PATCH_MODULES after hs_lora_epilogue, or anywhere: it imports it).

What the recipe does today, per layer and step (ACL 0, fix #2 residual fold on, HS_GRAD_CAST_ONLY=1):
  the residual stream h feeds two consumers: the next TE LayerNormLinear (qkv or fc1: RMSNorm fused in) and the
  residual add (fix #2's proj epilogue for the attention residual; Megatron's bias_dropout_add for the MLP one).
  backward: the residual add passes its gradient g to both branches; the linear whose output was added (proj or
  fc2) casts g to e5m2 for its dgrad GEMM (TE unary_kernel, 65-85 us); the LayerNormLinear's norm backward
  (TE Triton _rmsnorm_bwd_triton) writes dx; autograd then accumulates dx + g for h (CUDAFunctor_add, 107 us).
  => per layer 2 adds + 2 casts (~340 us), 160 kernels / step.

With this module the residual add's backward returns no gradient for the residual branch and stashes g; the
LayerNormLinear backward's norm kernel (a copy of TE's `_rmsnorm_bwd_triton`, same tree, same launch config)
computes dx in registers, adds g, writes BF16 (dx + g) once AND its e5m2 bytes with the consuming linear's
grad_output quantizer (+ amax, scale_inv); the Float8Tensor travels through a side channel keyed by the BF16
tensor's storage to the residual add's backward, which returns it as the grad of the linear's output, so TE's
grad_output_preprocess skips the cast (as fixes 3a / 3b do). Bit-exact with today's chain by construction:
  dx       : TE's kernel arithmetic verbatim (same source, same num_warps / BLOCK / grid), RNE to BF16 as stored
  dx + g   : fp32 add of the two BF16 values, RNE (torch's BF16 add; commutative, so the engine's order is moot)
  e5m2     : TE unary_kernel semantics via hs_dswiglu_e5m2._e5m2_bits (x scale, clamp +-57344 if finite, hardware
             cvt); amax = max |BF16 sum| (fmaxf: NaN ignored), atomic max on the fp32 bits; scale_inv = 1/scale
Fallbacks keep the gradient exact when a fold cannot complete (no quantizer, blocked norm shape, C++ norm path):
the LayerNormLinear backward wrapper adds the stashed g to dx itself and TE casts as before.

Roles, per residual-stream tensor h with its two consumers (LayerNormLinear L and residual add S):
  producer P of h   (the residual add that made it): supplies the e5m2 quantizer of its linear (fc2 or proj) and
                    takes the companion in its backward (the grad of h is what its linear's output gets)
  LayerNormLinear L: claims the fold in its forward when P is found directly (no checkpointing in between) and
                    tags h; its norm backward computes (dx + g) and the companion
  residual add S    (the one that reads h as its residual, built after L in the forward): sees the tag, stashes g
                    in its backward and returns no gradient for h
  qkv norm of layer l+1: P = _ResidualAddMLP(l), S = _LoraUp(proj, l+1);  fc1 norm of l: P = _LoraUp(proj, l),
  S = _ResidualAddMLP(l). Layer 0's qkv norm (embedding input) and the last MLP add (final norm) stay as they are.
"""

import os
import threading

import torch
import triton
import triton.language as tl

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module import layernorm_linear as lnl_mod
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
from transformer_engine.pytorch.triton_kernels import rmsnorm as _rms
from transformer_engine.pytorch.triton_kernels.norm_common import block_size, num_programs, use_blocked

import hs_lora_epilogue as _le  # installs fix #2's hooks first (our wrappers go outside them)
from hs_dswiglu_e5m2 import _e5m2_bits

_ENABLED = os.getenv("HS_RESGRAD_FOLD", "0") == "1"
_QKV = os.getenv("HS_QKVGRAD_E5M2", "0") == "1"   # fix 8c: fix 4's RoPE backward also emits the e5m2 d_mixed
_VIEW_NODES = _le._VIEW_NODES + ("MakeViewlessTensorBackward",)
_PRODUCERS = ("_ResidualAddMLPBackward", "_LoraUpBackward")
STATS = {"fold": 0, "fallback_add": 0, "e5m2_taken": 0, "e5m2_missed": 0, "marked": 0, "mlp_add": 0, "qkv_e5m2": 0}
_tls = threading.local()   # read with getattr: autograd's backward runs on its own worker thread
# _tls.pending: residual-branch gradient stashed by a residual add's backward, for the next norm backward
# _tls.fold: (g_res, quantizer, inp_shape) while the folded LayerNormLinear backward runs
_E5M2 = {}            # (data_ptr, numel) of a BF16 grad -> its e5m2 Float8Tensor companion


def set_enabled(on: bool):
    global _ENABLED
    _ENABLED = bool(on)


def enabled():
    return _ENABLED


def set_qkv_e5m2(on: bool):
    global _QKV
    _QKV = bool(on)


# ------------------------------------------------------------------------------------------------------- kernel

@triton.jit
def _rmsnorm_bwd_resfold_kernel(grad_output_ptr, input_ptr, g_ptr, rsigma_ptr, dx_ptr, dg_ptr, res_ptr, f8_ptr,
                                scale_ptr, amax_ptr, sinv_ptr, input_row_stride, output_row_stride, res_row_stride,
                                n_rows, n_cols, ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                                NUM_PRGMS: tl.constexpr):
    """TE's _rmsnorm_bwd_triton (non-blocked path, verbatim) + epilogue: dx (BF16-rounded) + res -> dx_ptr (BF16),
    e5m2 bytes of that sum -> f8_ptr, amax / scale_inv of the quantizer."""
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dg_col_redux = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    m = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    scale = tl.load(scale_ptr)

    for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        grad_output_ptrs = grad_output_ptr + row_idx * output_row_stride + col_offsets
        dx_ptrs = dx_ptr + row_idx * input_row_stride + col_offsets

        input_ptrs = tl.multiple_of(input_ptrs, (16, ))
        grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))
        dx_ptrs = tl.multiple_of(dx_ptrs, (16, ))

        x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
        grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        if (ZERO_CENTERED_GAMMA):
            g += 1.

        norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
        grad_sum = tl.sum(grad_output * x * g, axis=0)

        grad_input = grad_output * norm_factor * g - (norm_factor * norm_factor * norm_factor) * x * (grad_sum /
                                                                                                      n_cols)
        # --- epilogue (fix 8b): the value TE would have stored, + the residual-branch gradient, RNE once
        gi = grad_input.to(dx_ptr.type.element_ty).to(tl.float32)
        r = tl.load(res_ptr + row_idx * res_row_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
        s = (gi + r).to(dx_ptr.type.element_ty)
        tl.store(dx_ptrs, s, mask=mask)
        f = s.to(tl.float32)
        m = tl.maximum(m, tl.abs(f))
        # e5m2 bytes stored as 16-bit pairs behind a reshape: a plain 1-byte store makes Triton lay the whole
        # row out 16 elements per thread (TE's kernel: 8), which changes the grad_sum reduction tree
        b2 = tl.reshape(_e5m2_bits(f, scale), [BLOCK_SIZE // 2, 2])
        lo, hi = tl.split(b2)
        packed = lo.to(tl.uint16) | (hi.to(tl.uint16) << 8)
        cols2 = tl.arange(0, BLOCK_SIZE // 2)
        tl.store(f8_ptr.to(tl.pointer_type(tl.uint16)) + (row_idx * n_cols) // 2 + cols2, packed,
                 mask=cols2 < n_cols // 2)

        dg = grad_output * x * norm_factor
        dg_col_redux += dg.to(tl.float32)

    tl.store(dg_ptr + tl.program_id(0) * input_row_stride + col_offsets, dg_col_redux, mask=mask)
    # amax of |BF16 sum| (masked lanes are 0; maximum ignores NaN like fmaxf; all-NaN -> 0 as TE's max = 0 init)
    mx = tl.maximum(tl.max(m, axis=0), 0.0)
    mi = mx.to(tl.int32, bitcast=True)
    cur = tl.load(amax_ptr, volatile=True)
    tl.atomic_max(amax_ptr, mi, mask=mi > cur, sem="relaxed")
    tl.store(sinv_ptr, 1.0 / scale, mask=row_start == 0)


def rmsnorm_bwd_resfold(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma, res, quantizer):
    """TE te_rmsnorm_bwd_triton(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma) with the fold:
    returns (dx + res) BF16 [M, N], dgamma, e5m2 bytes uint8 [M, N] of the sum in `quantizer`, scale_inv [1].
    The quantizer's amax is max-accumulated in place."""
    dz_ = dz.contiguous()
    x_ = x.contiguous()
    rsigma_ = rsigma.contiguous()
    gamma_ = gamma.contiguous()
    res_ = res.contiguous()
    M, N = x_.shape
    assert not use_blocked(x_) and N % 2 == 0, "fix 8b: blocked RMSNorm backward shape"
    assert res_.shape == (M, N) and res_.dtype == x_.dtype == dz_.dtype, (res_.shape, res_.dtype, x_.dtype)
    assert x_.data_ptr() % 2 == 0
    out = torch.empty_like(x_)
    dgamma = torch.empty_like(gamma_)
    f8 = torch.empty((M, N), dtype=torch.uint8, device=x_.device)
    sinv = torch.empty(1, dtype=torch.float32, device=x_.device)
    blk_size = block_size(x_)
    NUM_PRGMS = num_programs(x_, sm_margin)
    dg_tmp = torch.empty(NUM_PRGMS, N, device=x_.device, dtype=torch.float32, requires_grad=False)
    _rmsnorm_bwd_resfold_kernel[(NUM_PRGMS, )](
        dz_, x_, gamma_, rsigma_, out, dg_tmp, res_, f8, quantizer.scale, quantizer.amax.view(torch.int32), sinv,
        x_.stride(0), dz_.stride(0), res_.stride(0), M, N, zero_centered_gamma, blk_size, NUM_PRGMS, num_warps=8,
        waves_per_eu=1)   # occupancy hint only (the layout, hence the row-sum tree, is unchanged): 202 -> 194 us
    grid_reduce = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    _rms._rmsnorm_bwd_dg_reduce_triton[grid_reduce](dg_tmp, dgamma, dg_tmp.stride(0), dg_tmp.shape[0],
                                                    dg_tmp.shape[1], BLOCK_SIZE_M=128, BLOCK_SIZE_N=64)
    return out, dgamma, f8, sinv


# --------------------------------------------------------------------------------------------- autograd plumbing

def _grad_quantizer(node):
    """The e5m2 grad_output quantizer of a TE _Linear node if the pre-quantized path is valid for it, else None
    (the conditions of fix 3a: FP8, no wgrad, cast-only grads, no comm overlap)."""
    if node is None or not getattr(node, "fp8", False) or getattr(node, "requires_wgrad", True):
        return None
    if os.environ.get("HS_GRAD_CAST_ONLY", "0") != "1" or getattr(node, "ub_overlap_ag", False):
        return None
    if getattr(node, "sequence_parallel", False) and getattr(node, "parallel_mode", None) == "row":
        return None
    q = getattr(node, "grad_output_quantizer", None)
    if type(q) is not Float8Quantizer or q.dtype != tex.DType.kFloat8E5M2:
        return None
    return q


def _key(t):
    return (t.data_ptr(), t.numel())


def _register_e5m2(out, f8, sinv, q, shape):
    if len(_E5M2) > 4:   # never accumulate: a companion that is not consumed is dropped, not kept
        _E5M2.clear()
    f8t = Float8Tensor(shape=tuple(shape), dtype=out.dtype, data=f8.view(tuple(shape)), fp8_scale_inv=sinv,
                       fp8_dtype=q.dtype, requires_grad=False, data_transpose=None, quantizer=q)
    _E5M2[_key(out)] = f8t


def _take_e5m2(g, q):
    f8t = _E5M2.pop(_key(g), None)
    if f8t is None or f8t._quantizer is not q or tuple(f8t.shape) != tuple(g.shape):
        STATS["e5m2_missed"] += 1
        return None
    STATS["e5m2_taken"] += 1
    return f8t


def _claim_stasher(ctx, residual):
    """Called in a residual add's forward: if a LayerNormLinear tagged this residual, this add's backward will
    stash the gradient for its norm backward instead of returning it."""
    lnl = residual.__dict__.pop("_hs_fold_lnl", None) if isinstance(residual, torch.Tensor) else None
    if lnl is not None:
        lnl._hs_resfold_stasher = True
        ctx._hs_fold_consumer = True


def _stash_res(g):
    assert getattr(_tls, "pending", None) is None, "fix 8b: a residual gradient was stashed and never consumed"
    _tls.pending = g


def _producer(t, max_depth=4):
    """The _ResidualAddMLP / residual-folded _LoraUp node that produced t, through view nodes; else None."""
    node = t.grad_fn if isinstance(t, torch.Tensor) else None
    for _ in range(max_depth):
        if node is None:
            return None
        name = type(node).__name__
        if name in _PRODUCERS:
            if name == "_LoraUpBackward" and not getattr(node, "has_res", False):
                return None
            return node
        if name not in _VIEW_NODES or not node.next_functions:
            return None
        node = node.next_functions[0][0]
    return None


class _ResidualAddMLP(torch.autograd.Function):
    """out = residual + x (Megatron's bias_dropout_add with bias None, dropout 0). Backward: x gets the e5m2
    companion of g when one was produced for its linear; the residual branch gets g, or nothing when a
    LayerNormLinear downstream has claimed the fold (then g is stashed for its norm backward)."""

    @staticmethod
    def forward(ctx, x, residual):
        ctx.hs_x_q = _grad_quantizer(_le.find_base_node(x, "_LinearBackward"))
        if ctx.needs_input_grad[1]:
            _claim_stasher(ctx, residual)
        STATS["mlp_add"] += 1
        return residual + x

    @staticmethod
    def backward(ctx, g):
        gx = g
        q = ctx.hs_x_q
        ctx.hs_x_q = None
        if q is not None:
            e5 = _take_e5m2(g, q)
            if e5 is not None:
                gx = e5
        if getattr(ctx, "_hs_fold_consumer", False):
            _stash_res(g)
            return gx, None
        return gx, g


def _install():
    import megatron.core.fusions.fused_bias_dropout as fbd

    # --- MLP residual add -> _ResidualAddMLP (on top of fix #2's passthrough for the folded attention residual)
    prev_bda = fbd._bias_dropout_add_func

    def _bias_dropout_add_func(x_with_bias, residual, prob, training):
        x, bias = x_with_bias
        if (_ENABLED and bias is None and prob == 0.0 and not getattr(x, "_hs_residual_folded", False)
                and torch.is_grad_enabled() and x.requires_grad and residual.requires_grad
                and residual.dtype == x.dtype and residual.shape == x.shape
                and x.is_contiguous() and residual.is_contiguous()):
            return _ResidualAddMLP.apply(x, residual)
        return prev_bda(x_with_bias, residual, prob, training)

    fbd._bias_dropout_add_func = _bias_dropout_add_func

    # --- fix #2's _LoraUp: remember proj's quantizer; backward takes the e5m2 companion, yields the residual fold
    _LoraUp = _le._LoraUp
    orig_up_fwd, orig_up_bwd = _LoraUp.forward, _LoraUp.backward

    def up_forward(ctx, base, a, b_w, scale, residual):
        out = orig_up_fwd(ctx, base, a, b_w, scale, residual)
        ctx.hs_x_q = None
        if _ENABLED and residual is not None:   # (grad mode is off inside Function.forward: do not test it)
            ctx.hs_x_q = _grad_quantizer(_le.find_base_node(base, "_LinearBackward"))
            if ctx.needs_input_grad[4]:
                _claim_stasher(ctx, residual)
        elif _QKV and residual is None:         # qkv: the companion comes from the fused RoPE backward (fix 8c)
            ctx.hs_x_q = _grad_quantizer(_le.find_base_node(base, "_LayerNormLinearBackward"))
        return out

    def up_backward(ctx, g):
        outs = list(orig_up_bwd(ctx, g))
        q = getattr(ctx, "hs_x_q", None)
        ctx.hs_x_q = None
        if q is not None:
            e5 = _take_e5m2(g, q)
            if e5 is not None:
                outs[0] = e5
        if getattr(ctx, "_hs_fold_consumer", False) and ctx.has_res:
            _stash_res(g)
            outs[4] = None
        return tuple(outs)

    _LoraUp.forward = staticmethod(up_forward)
    _LoraUp.backward = staticmethod(up_backward)

    # --- te.LayerNormLinear.forward: claim the fold when the input's producer is a residual add with a quantizer
    orig_mod_fwd = te.LayerNormLinear.forward

    def lnl_module_forward(self, inp, *args, **kwargs):
        out = orig_mod_fwd(self, inp, *args, **kwargs)
        if _ENABLED and torch.is_grad_enabled() and self.training and self.normalization == "RMSNorm":
            prod = _producer(inp)
            if prod is not None and getattr(prod, "hs_x_q", None) is not None:
                o = out[0] if isinstance(out, tuple) else out
                node = _le.find_base_node(o, "_LayerNormLinearBackward")
                if (node is not None and getattr(node, "fp8", False) and getattr(node, "requires_dgrad", False)
                        and not getattr(node, "ub_bulk_wgrad", False) and not getattr(node, "ub_bulk_dgrad", False)
                        and not getattr(node, "ub_overlap_rs_dgrad", False)):
                    node._hs_resfold_q = prod.hs_x_q
                    inp._hs_fold_lnl = node          # read by the residual add that consumes `inp` next
                    STATS["marked"] += 1
        return out

    te.LayerNormLinear.forward = lnl_module_forward

    # --- _LayerNormLinear.backward: expose the stash to the norm backward; add it in plain BF16 if not consumed
    _LNL = lnl_mod._LayerNormLinear
    prev_lnl_bwd = _LNL.backward

    def lnl_backward(ctx, *grad_outputs):
        q = ctx.__dict__.pop("_hs_resfold_q", None)
        stasher = ctx.__dict__.pop("_hs_resfold_stasher", False)
        if q is None or not stasher:
            return prev_lnl_bwd(ctx, *grad_outputs)
        g_res = getattr(_tls, "pending", None)
        _tls.pending = None
        assert g_res is not None, "fix 8b: fold claimed but no residual gradient was stashed"
        assert tuple(g_res.shape) == tuple(ctx.inp_shape), (g_res.shape, ctx.inp_shape)
        _tls.fold = (g_res, q, tuple(ctx.inp_shape))
        try:
            outs = prev_lnl_bwd(ctx, *grad_outputs)
        finally:
            fold, _tls.fold = getattr(_tls, "fold", None), None
        if fold is not None:   # the norm backward did not fold (C++ norm path / blocked shape): plain accumulate
            STATS["fallback_add"] += 1
            outs = list(outs)
            if outs[0] is not None:
                outs[0] = outs[0] + g_res.view_as(outs[0])
            outs = tuple(outs)
        return outs

    _LNL.backward = staticmethod(lnl_backward)

    # --- the RMSNorm backward (TE Triton path), outside fix #2's LoRA-dgrad hook
    prev_norm_bwd = lnl_mod.te_rmsnorm_bwd_triton

    def norm_bwd(dgrad, x, rsigma, gamma, sm_margin, zero_centered_gamma):
        fold = getattr(_tls, "fold", None)
        if fold is None:
            return prev_norm_bwd(dgrad, x, rsigma, gamma, sm_margin, zero_centered_gamma)
        g_res, q, inp_shape = fold
        x2 = x.contiguous()
        if use_blocked(x2) or dgrad.shape != x2.shape or q is None:
            return prev_norm_bwd(dgrad, x, rsigma, gamma, sm_margin, zero_centered_gamma)
        pending = getattr(_le._tls, "pending", None)   # what fix #2's hook would apply before the norm backward
        if pending is not None:
            _le._tls.pending = None
            ga, a_w, mask, scale = pending
            _le.lora_dgrad_acc_(dgrad.view(-1, dgrad.shape[-1]), ga, a_w, mask, scale)
        out, dgamma, f8, sinv = rmsnorm_bwd_resfold(dgrad, x2, rsigma, gamma, sm_margin, zero_centered_gamma,
                                                    g_res.view(-1, x2.shape[-1]), q)
        _tls.fold = None
        _register_e5m2(out, f8, sinv, q, inp_shape)
        STATS["fold"] += 1
        return out, dgamma

    norm_bwd._hs_orig = prev_norm_bwd
    lnl_mod.te_rmsnorm_bwd_triton = norm_bwd

    # --- fix 8c: fix 4's fused RoPE backward writes the e5m2 copy of d_mixed for the qkv LayerNormLinear
    import hs_rope_qkv as rq

    orig_rope_fwd, orig_rope_bwd = rq.FusedQKVRoPE.forward, rq.FusedQKVRoPE.backward

    def rope_forward(ctx, mixed, freqs, cfg_fwd=None, cfg_bwd=None):
        outs = orig_rope_fwd(ctx, mixed, freqs, cfg_fwd, cfg_bwd)
        ctx.hs_q = None
        if _QKV:
            node = mixed.grad_fn
            if node is not None and type(node).__name__ == "_LoraUpBackward":
                ctx.hs_q = getattr(node, "hs_x_q", None)
        return outs

    def rope_backward(ctx, dq, dk, dv):
        q = getattr(ctx, "hs_q", None)
        ctx.hs_q = None
        if q is None:
            return orig_rope_bwd(ctx, dq, dk, dv)
        (freqs,) = ctx.saved_tensors
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        f8 = torch.empty((dq.shape[0], 1, rq.ROW), dtype=torch.uint8, device=dq.device)
        sinv = torch.empty(1, dtype=torch.float32, device=dq.device)
        cfg = dict(ctx.cfg_bwd or rq.DEFAULT, num_warps=4)   # the emission path's sweet spot (148 vs 168 us at 8)
        out = rq.rope_qkv_bwd(dq, dk, dv, freqs, cfg, e5m2=(f8, q.scale, q.amax.view(torch.int32), sinv))
        _register_e5m2(out, f8, sinv, q, out.shape)
        STATS["qkv_e5m2"] += 1
        return out, None, None, None

    rq.FusedQKVRoPE.forward = staticmethod(rope_forward)
    rq.FusedQKVRoPE.backward = staticmethod(rope_backward)


_install()
print(f"[hyperspindle] hs_resgrad_fold: {'on' if _ENABLED else 'off'} (residual-grad add + e5m2 cast folded into "
      f"the RMSNorm backward; HS_RESGRAD_FOLD=1 enables); qkv-grad e5m2 from the RoPE backward "
      f"{'on' if _QKV else 'off'} (HS_QKVGRAD_E5M2=1)", flush=True)


def report():
    return f"[hyperspindle] hs_resgrad_fold: {STATS}"
