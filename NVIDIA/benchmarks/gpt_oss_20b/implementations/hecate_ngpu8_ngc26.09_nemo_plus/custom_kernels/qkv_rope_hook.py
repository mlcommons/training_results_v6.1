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

"""Patches the gpt-oss-20b FP8 attention input/output chain in Megatron and TransformerEngine (six rebindings, resolved at call time):
  1. SelfAttention.get_query_key_value_tensors: the unsplit (fused_single_qkv_rope) path returns the qkv GEMM output as the
     3-D [s, b, ng*(dq+dk+dv)] tensor, so the RoPE backward can hand the MXFP8 d(qkv) straight to the Linear.
  2. apply_fused_qkv_rotary_pos_emb -> autograd Function: fwd gpt_oss::qkv_rope_fwd_amax (RoPE(q,k) + v into one flat q|k|v bf16
     buffer plus the Float8CurrentScaling amax/scale; q/k/v are views of it); bwd gpt_oss::qkv_rope_bwd_mxfp8 + gpt_oss::dbias_reduce
     -> MXFP8Tensor d(qkv) (rowwise + columnwise, GEMM-swizzled scales) carrying the bias gradient as `_fused_grad_bias`.
  3. backends.combine_and_quantize: q/k/v from (2) are quantized as the flat buffer with the precomputed scale (one TE cast_fp8_1D;
     no torch.cat, zero_amax, amax_kernel, compute_scale_from_amax or at::reciprocal launches).
  4. TransformerEngineBaseModule.grad_output_preprocess: an MXFP8 grad from (2) is returned as-is with its bias gradient.
  5./6. Attention.forward_pre_attn_and_core_attn records the rotary freqs / head geometry; layernorm_linear.general_gemm then runs
     the qkv projection (MXFP8 x MXFP8 + bias -> bf16) as gpt_oss::qkv_gemm_rope_fwd_amax (qkv_gemm_rope_rubin.py, Rubin-only
     block-scaled GEMM with bias + RoPE + relocation + amax in the epilogue), so (2) only builds the views. Backward unchanged.
Anything outside the fast path (other layouts, CP, interleaved RoPE, other recipes) runs the original code. Gates, default on: QKV_ROPE=0
leaves Megatron/TE untouched; QKV_GEMM=0 keeps cuBLAS + the standalone RoPE/amax pass; QKV_DUMP_DIR=<dir> dumps rank 0's first qkv forward.
"""
import os
import sys
from typing import Optional

import torch

_INSTALLED = False
_STATS = {"fwd": 0, "bwd": 0, "quant_fast": 0, "quant_fallback": 0, "rope_fallback": 0, "gop_fast": 0}
_REG = {}          # q.data_ptr() -> entry (scale, amax, numel + storage bytes of the flat buffer, (q,k,v) ptrs + shapes);
                   # holds NO reference to the flat buffer itself (its lifetime is q/k/v's), so a stale entry costs ~100 B
_TABLES = {}       # (freqs ptr, shape, version) -> (cos, sin) fp32 [s, d]
_ORIG = {}         # the originals the rebindings wrap
_REG_MAX = 64
_GRAD_QUANTIZERS = {}   # TE fp8 dtype -> MXFP8Quantizer carried by the d(qkv) MXFP8Tensor (one per dtype, not per call)
_STATS["gop_slow"] = 0
SWIZZLED_SCALES = os.environ.get("QKV_ROPE_SWIZZLED", "1") == "1"
_DUMP_DIR = os.environ.get("QKV_DUMP_DIR", "")   # numerics dump of the first qkv forward on rank 0 (diagnostic; see docstring)
_DUMPED = set()
_FUSED = {}        # flat.data_ptr() -> {scale, amax, numel} of a fused GEMM+RoPE output awaiting its RoPE Function (2)
_PENDING = [None]  # the attention forward's RoPE context (cos32, sin32, s, b, ng, nqpg, N) while linear_qkv runs
_TABLES32 = {}     # (freqs ptr, shape, version, s) -> (cos32, sin32) fp32 [s, 32] (the 64-wide tables are 2 copies)
_GEMM_OK = False   # the fused GEMM kernel module imported
_ARCH = [None]     # the device compute capability, read once at the first qkv GEMM (the kernel is Rubin-only)
_STATS.update({"gemm_fused": 0, "gemm_fallback": 0, "rope_fused": 0, "gemm_calls": 0, "ctx_miss": 0, "pre_attn_calls": 0})


def _say(msg: str) -> None:
    # utils.py replaces builtins.print with a rank-0-only print; write through sys.__stdout__ instead
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"QKV_ROPE rank={rank} {msg}\n")
    sys.__stdout__.flush()


def _dump(name: str, **items) -> None:
    """torch.save `items` (tensors copied to the CPU, scalars as-is) as <QKV_DUMP_DIR>/<name>.pt -- rank 0, first
    call per name only (the first qkv attention forward of the process: layer 0, step 0). Never raises."""
    if not _DUMP_DIR or name in _DUMPED or os.getenv("RANK", os.getenv("SLURM_PROCID", "0")) != "0":
        return
    _DUMPED.add(name)
    try:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        payload = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in items.items()}
        torch.save(payload, os.path.join(_DUMP_DIR, f"{name}.pt"))
        _say(f"numerics dump {name} -> {_DUMP_DIR}/{name}.pt: "
             + ", ".join(f"{k}{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}" if torch.is_tensor(v) else f"{k}={v}"
                         for k, v in items.items()))
    except Exception as e:  # noqa
        _say(f"numerics dump {name} failed: {type(e).__name__}: {e}")


def _tables(freqs: torch.Tensor, s: int):
    """cos/sin [s, d] fp32 of the rotary freqs [s2, 1, 1, d] (s2 >= s), cached per (data_ptr, shape, _version, s).
    Megatron's RotaryEmbedding.forward is lru_cached, so every layer of every step gets the same never-written freqs tensor
    and the key is exact; a caller that builds fresh freqs per step could hit a stale table and must clear _TABLES first."""
    key = (freqs.data_ptr(), tuple(freqs.shape), freqs._version, s)
    ent = _TABLES.get(key)
    if ent is None:
        f = freqs.detach()[:s].float().reshape(s, freqs.shape[-1]).contiguous()
        ent = (torch.cos(f).contiguous(), torch.sin(f).contiguous())
        if len(_TABLES) >= 8:
            _TABLES.clear()
        _TABLES[key] = ent
    return ent


def _tables32(freqs: torch.Tensor, s: int):
    """cos/sin [s, 32] fp32 of the rotary freqs (same cache key discipline as _tables). Megatron's RotaryEmbedding
    builds freqs = cat(f, f), so cos[d] == cos[d + 32]; the fused GEMM epilogue uses the 32-wide half. Returns None
    when the two halves differ (the epilogue would be wrong for such freqs; the standalone pass handles them)."""
    key = (freqs.data_ptr(), tuple(freqs.shape), freqs._version, s)
    if key in _TABLES32:
        return _TABLES32[key]
    f = freqs.detach()[:s].float().reshape(s, freqs.shape[-1])
    ent = None
    if f.shape[-1] == 64 and bool(torch.equal(f[:, :32], f[:, 32:])):
        ent = (torch.cos(f[:, :32]).contiguous(), torch.sin(f[:, :32]).contiguous())
    if len(_TABLES32) >= 8:
        _TABLES32.clear()
    _TABLES32[key] = ent
    return ent


def _same_freqs(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a is b or (a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride())


class _FusedQKVRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, ng, dq, dk, dv, grad_fp8_code, swizzled, grad_quantizer):
        s, b, hidden = qkv.shape
        fused = _FUSED.pop(qkv.data_ptr(), None)
        if fused is not None and fused["numel"] == qkv.numel() and qkv.is_contiguous():
            # the qkv GEMM epilogue already applied bias + RoPE and wrote the flat q|k|v layout (+ amax/scale)
            flat = qkv.view(-1)
            scale, amax = fused["scale"], fused["amax"]
            _STATS["rope_fused"] += 1
            if _DUMP_DIR:
                _dump("rope_fwd", path="fused", flat=flat, scale=scale, amax=amax, cos=cos, sin=sin, s=s, b=b,
                      ng=ng, dq=dq, dk=dk, dv=dv)
        else:
            if fused is not None:
                _say(f"fused GEMM result not usable (numel {fused['numel']} vs {qkv.numel()}, contiguous={qkv.is_contiguous()}); standalone pass")
            flat, scale, amax = torch.ops.gpt_oss.qkv_rope_fwd_amax(qkv, cos, sin, ng, dq, dk, dv)
            if _DUMP_DIR:
                # qkv here is the cuBLAS GEMM(+bias) bf16 output before RoPE: the chain's own input to the standalone pass
                _dump("rope_fwd", path="standalone", qkv=qkv, flat=flat, scale=scale, amax=amax, cos=cos, sin=sin, s=s, b=b,
                      ng=ng, dq=dq, dk=dk, dv=dv)
        d = dk
        nqh = ng * (dq // d)
        nq = s * b * nqh * d
        nk = s * b * ng * d
        q = flat[:nq].view(s, b, nqh, d)
        k = flat[nq:nq + nk].view(s, b, ng, d)
        v = flat[nq + nk:].view(s, b, ng, d)
        if len(_REG) >= _REG_MAX:
            _REG.pop(next(iter(_REG)))
        _REG[q.data_ptr()] = {
            "scale": scale, "amax": amax, "numel": flat.numel(), "nbytes": flat.untyped_storage().nbytes(),
            "ptrs": (q.data_ptr(), k.data_ptr(), v.data_ptr()), "shapes": (q.shape, k.shape, v.shape),
        }
        ctx.save_for_backward(cos, sin)
        ctx.meta = (s, b, hidden, ng, dq, dk, dv, int(grad_fp8_code), bool(swizzled))
        ctx.grad_quantizer = grad_quantizer
        _STATS["fwd"] += 1
        if _STATS["fwd"] <= 2:
            _say(f"fwd: qkv={tuple(qkv.shape)} -> flat q|k|v + amax/scale (ng={ng} split=({dq},{dk},{dv}))")
        return q, k, v

    @staticmethod
    def backward(ctx, gq, gk, gv):
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        cos, sin = ctx.saved_tensors
        s, b, hidden, ng, dq, dk, dv, code, swizzled = ctx.meta
        grads = []
        for g in (gq, gk, gv):
            if isinstance(g, QuantizedTensorStorage):
                g = g.dequantize(dtype=torch.bfloat16)
            grads.append(g.contiguous())
        gq, gk, gv = grads
        rd, rs, cd, cs, part = torch.ops.gpt_oss.qkv_rope_bwd_mxfp8(
            gq, gk, gv, cos, sin, ng, dq, dk, dv, code, swizzled)
        dbias = torch.ops.gpt_oss.dbias_reduce(part, gq.dtype)
        fp8_dtype = tex.DType.kFloat8E4M3 if code == 7 else tex.DType.kFloat8E5M2
        grad = MXFP8Tensor(
            shape=(s, b, hidden), dtype=gq.dtype,
            rowwise_data=rd.view(s, b, hidden), rowwise_scale_inv=rs,
            columnwise_data=cd.view(s, b, hidden), columnwise_scale_inv=cs,
            fp8_dtype=fp8_dtype, quantizer=ctx.grad_quantizer, with_gemm_swizzled_scales=swizzled,
            requires_grad=False, device=gq.device,
        )
        grad._fused_grad_bias = dbias
        grad._fused_mxfp8 = True
        _STATS["bwd"] += 1
        if _STATS["bwd"] <= 2:
            _say(f"bwd: d(qkv) MXFP8 row+col ({'e4m3' if code == 7 else 'e5m2'}, swizzled={swizzled}) + dbias[{hidden}]")
        return grad, None, None, None, None, None, None, None, None, None


def install() -> bool:
    """Install the rebindings. Returns True when the fast path is active."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.getenv("QKV_ROPE", "1") != "1":
        _say("disabled by QKV_ROPE=0 (control path: TE fused_qkv_rope + current-scaling quantize chain)")
        return False
    try:
        import importlib
        try:
            importlib.import_module("custom_kernels.qkv_rope_fp8")
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            importlib.import_module("qkv_rope_fp8")
    except Exception as e:  # noqa
        _say(f"kernel import failed, hooks NOT installed: {type(e).__name__}: {e}")
        return False

    import transformer_engine.pytorch  # noqa: F401  (loads the transformer_engine_torch extension)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.attention.dot_product_attention import backends
    from transformer_engine.pytorch.attention import rope as te_rope
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, get_fp8_te_dtype
    from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer, Float8Tensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.common.recipe import Format
    from megatron.core.transformer import attention as mcore_attention
    from megatron.core.typed_torch import apply_module

    # ------------------------------------------------------------------ 1. Megatron: unsplit qkv as 3-D
    # Mirrors megatron-core 0.20.0 SelfAttention.get_query_key_value_tensors up to `if not split_qkv: return mixed_qkv,
    # split_arg_list`, minus the [s, b, ng, gsz] view. linear_qkv is TELayerNormColumnParallelLinear (skip_bias_add=False),
    # whose forward returns (out, None), hence the 2-tuple unpack. Anything else (split_qkv, output gate,
    # num_query_groups < tp size) goes to the original.
    _orig_gqkv = mcore_attention.SelfAttention.get_query_key_value_tensors
    _ORIG["gqkv"] = _orig_gqkv

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
        if split_qkv or output_gate or getattr(self.config, "qk_layernorm", False) or (
            self.config.num_query_groups is not None and self.config.num_query_groups < self.world_size
        ):
            return _orig_gqkv(self, hidden_states, key_value_states, output_gate=output_gate, split_qkv=split_qkv)
        mixed_qkv, _ = apply_module(self.linear_qkv)(hidden_states)
        nqpg = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        hn = self.hidden_size_per_attention_head
        return mixed_qkv, [nqpg * hn, hn, hn]

    mcore_attention.SelfAttention.get_query_key_value_tensors = get_query_key_value_tensors

    # ------------------------------------------------------------------ 2. RoPE forward/backward
    _te_apply = te_rope.apply_fused_qkv_rotary_pos_emb

    def apply_fused_qkv_rotary_pos_emb(qkv, q_freqs, k_freqs, qkv_split_arg_list, tensor_format="sbhd",
                                       start_positions=None, interleaved=False, cu_seqlens=None, cp_size=1, cp_rank=0,
                                       **norm_kwargs):
        why = None
        if any(v is not None for v in norm_kwargs.values()):
            why = "qk-norm fused into rope (" + ", ".join(k for k, v in norm_kwargs.items() if v is not None) + ")"
        elif qkv.dim() != 3:
            why = f"qkv.dim={qkv.dim()}"
        elif tensor_format != "sbhd" or start_positions is not None or interleaved or cp_size != 1:
            why = f"format={tensor_format} start_positions={start_positions is not None} interleaved={interleaved} cp={cp_size}"
        elif qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
            why = f"dtype={qkv.dtype} contiguous={qkv.is_contiguous()}"
        else:
            s, b, hidden = qkv.shape
            dq_, dk_, dv_ = (int(x) for x in qkv_split_arg_list)
            gsz = dq_ + dk_ + dv_
            ng = hidden // gsz
            if hidden % gsz or dk_ != 64 or dv_ != 64 or dq_ % 64 or (dq_ // 64) % 4:
                why = f"split={qkv_split_arg_list} hidden={hidden}"
            elif q_freqs.shape[-1] != 64 or q_freqs.shape[0] < s or not _same_freqs(q_freqs, k_freqs):
                why = f"freqs q={tuple(q_freqs.shape)} k={tuple(k_freqs.shape)} same={_same_freqs(q_freqs, k_freqs)}"
            elif (s * b) % 128 or hidden % 128:
                why = f"M={s * b} N={hidden} not multiples of 128"
            elif not FP8GlobalStateManager.is_fp8_enabled():
                why = "fp8 disabled"
            else:
                recipe = FP8GlobalStateManager.get_fp8_recipe()
                if not recipe.mxfp8():
                    why = f"linear recipe {type(recipe).__name__} is not MXFP8"
        if why is not None:
            _STATS["rope_fallback"] += 1
            if _STATS["rope_fallback"] <= 3:
                _say(f"rope fallback to TE fused_qkv_rope: {why}")
            if qkv.dim() == 3:
                s, b, hidden = qkv.shape
                gsz = sum(int(x) for x in qkv_split_arg_list)
                qkv = qkv.view(s, b, hidden // gsz, gsz)
            return _te_apply(qkv, q_freqs, k_freqs, qkv_split_arg_list, tensor_format, start_positions,
                             interleaved, cu_seqlens, cp_size, cp_rank, **norm_kwargs)
        cos, sin = _tables(q_freqs, s)
        grad_dtype = get_fp8_te_dtype(recipe, fprop_tensor=False)
        code = 7 if grad_dtype == tex.DType.kFloat8E4M3 else 8
        grad_quantizer = _GRAD_QUANTIZERS.get(grad_dtype)
        if grad_quantizer is None:
            grad_quantizer = MXFP8Quantizer(grad_dtype, rowwise=True, columnwise=True)
            _GRAD_QUANTIZERS[grad_dtype] = grad_quantizer
        return _FusedQKVRoPE.apply(qkv, cos, sin, ng, dq_, dk_, dv_, code, SWIZZLED_SCALES, grad_quantizer)

    mcore_attention.apply_fused_qkv_rotary_pos_emb = apply_fused_qkv_rotary_pos_emb

    # ------------------------------------------------------------------ 3. TE: quantize the flat buffer, no cat/amax
    _orig_caq = backends.combine_and_quantize

    def combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer, used_in_forward=True, used_in_backward=False,
                             keep_same_data_and_scale_inv_format=False, combined_qkv=None, combined_kv=None):
        # pop on every call: an entry left behind (any configuration outside the fast path) must not outlive this call
        ent = _REG.pop(q.data_ptr(), None)
        fast = (ent is not None and qkv_layout == "sbhd_sbhd_sbhd"
                and isinstance(qkv_quantizer, Float8CurrentScalingQuantizer)
                and not getattr(qkv_quantizer, "with_amax_reduction", False)
                and ent["ptrs"] == (q.data_ptr(), k.data_ptr(), v.data_ptr())
                and ent["shapes"] == (q.shape, k.shape, v.shape)
                and q.storage_offset() == 0 and q.untyped_storage().nbytes() == ent["nbytes"])
        if not fast:
            _STATS["quant_fallback"] += 1
            if _STATS["quant_fallback"] <= 3:
                _say(f"combine_and_quantize fallback: layout={qkv_layout} quantizer={type(qkv_quantizer).__name__} registry_hit={ent is not None}")
            return _orig_caq(qkv_layout, q, k, v, qkv_quantizer, used_in_forward, used_in_backward,
                             keep_same_data_and_scale_inv_format, combined_qkv, combined_kv)
        # the flat q|k|v buffer is q's storage (q = flat[:nq].view(...)); rebuild the 1-D view instead of pinning it
        flat = torch.empty(0, dtype=q.dtype, device=q.device).set_(q.untyped_storage(), 0, (ent["numel"],))
        scale, amax = ent["scale"], ent["amax"]
        fq = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=qkv_quantizer.dtype, rowwise=True, columnwise=False)
        out = Float8Tensor(
            shape=tuple(flat.shape), dtype=q.dtype,
            data=torch.empty(flat.shape, dtype=torch.uint8, device=flat.device),
            fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=flat.device),
            fp8_dtype=qkv_quantizer.dtype, quantizer=qkv_quantizer, requires_grad=False, device=flat.device,
        )
        tex.quantize(flat, fq, out, None)          # TE cast_fp8_1D: data = sat(x * scale), scale_inv = 1/scale
        numels = [q.numel(), k.numel(), v.numel()]
        if _DUMP_DIR:
            _dump("fp8", data=out._data, scale_inv=getattr(out, "_scale_inv", None), scale=scale, amax=amax, numels=numels,
                  fp8_dtype=str(qkv_quantizer.dtype))
        offs = [0, numels[0], numels[0] + numels[1]]
        datas = [out._data[o:o + n].view(x.shape) for o, n, x in zip(offs, numels, (q, k, v))]
        q_fp8, k_fp8, v_fp8 = [Float8Tensor.make_like(out, data=d_, dtype=q.dtype) for d_ in datas]
        _STATS["quant_fast"] += 1
        if _STATS["quant_fast"] <= 2:
            _say(f"combine_and_quantize fast path: flat[{flat.numel()}] -> fp8 {qkv_quantizer.dtype} with precomputed scale")
        return q_fp8, k_fp8, v_fp8, qkv_layout, None

    backends.combine_and_quantize = combine_and_quantize

    # ------------------------------------------------------------------ 4. TE: accept the MXFP8 grad + bias grad
    _orig_gop = TransformerEngineBaseModule.grad_output_preprocess

    def grad_output_preprocess(ctx, grad_output, row_parallel_mode, quantizer):
        if getattr(grad_output, "_fused_mxfp8", False):
            gather = row_parallel_mode and ctx.sequence_parallel
            use_fp8_bwd = ctx.fp8 and ctx.backward_override is None
            if use_fp8_bwd and not gather and not getattr(ctx, "debug", False) and not getattr(ctx, "ub_overlap_ag", False):
                _STATS["gop_fast"] += 1
                if _STATS["gop_fast"] <= 2:
                    _say(f"grad_output_preprocess fast path: MXFP8 grad {tuple(grad_output.shape)} use_bias={ctx.use_bias}")
                return grad_output, (grad_output._fused_grad_bias if ctx.use_bias else None)
            _STATS["gop_slow"] += 1
            if _STATS["gop_slow"] <= 3:
                _say(f"grad_output_preprocess: configuration outside the fast path (fp8={ctx.fp8} override={ctx.backward_override} "
                     f"gather={gather}), dequantizing the MXFP8 grad")
            grad_output = grad_output.dequantize()
        return _orig_gop(ctx, grad_output, row_parallel_mode, quantizer)

    TransformerEngineBaseModule.grad_output_preprocess = staticmethod(grad_output_preprocess)

    # ------------------------------------------------------------------ 5./6. the fused qkv GEMM epilogue
    gemm_desc = _install_fused_gemm(mcore_attention)

    _INSTALLED = True
    _say("installed: SelfAttention.get_query_key_value_tensors (3-D qkv), apply_fused_qkv_rotary_pos_emb -> fused "
         "RoPE fwd(amax)/bwd(MXFP8+dbias), backends.combine_and_quantize (flat buffer, precomputed scale), "
         f"TransformerEngineBaseModule.grad_output_preprocess (MXFP8 grad pass-through); swizzled_scales={SWIZZLED_SCALES}; "
         f"fused qkv GEMM epilogue: {gemm_desc}")
    return True


def _install_fused_gemm(mcore_attention) -> str:
    """Rebinding points 5 and 6 (see the module docstring). Returns a one-line description for the install log."""
    global _GEMM_OK
    import inspect

    if os.getenv("QKV_GEMM", "1") != "1":
        return "disabled by QKV_GEMM=0 (control path: cuBLAS GEMM + standalone RoPE/amax pass)"
    try:
        import importlib
        try:
            importlib.import_module("custom_kernels.qkv_gemm_rope_rubin")
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            importlib.import_module("qkv_gemm_rope_rubin")
    except Exception as e:  # noqa
        if isinstance(e, RuntimeError) and "development-only knob" in str(e):
            # a development-only kernel knob in a training run: stop here, never fall back silently
            _say(f"fused qkv GEMM kernel refused to import: {e}")
            raise
        _say(f"fused qkv GEMM kernel import failed, cuBLAS GEMM + standalone pass stay: {type(e).__name__}: {e}")
        return f"kernel import failed ({type(e).__name__})"
    import torch as _torch
    import transformer_engine_torch as tex
    import transformer_engine.pytorch.module.layernorm_linear as te_lnl
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8TensorStorage

    # -------- 5. Megatron: record the RoPE context of this attention forward while linear_qkv runs
    _orig_pre = mcore_attention.Attention.forward_pre_attn_and_core_attn
    _ORIG["pre_attn"] = _orig_pre
    sig = inspect.signature(_orig_pre)

    def forward_pre_attn_and_core_attn(self, *args, **kwargs):
        ctx = None
        _STATS["pre_attn_calls"] += 1
        try:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            hidden_states = a["hidden_states"]
            rope = a.get("rotary_pos_emb")
            if _torch.is_tensor(rope):
                rope = (rope, rope)   # Megatron: a single freqs tensor applies to both q and k
            cfg = self.config
            if _STATS["pre_attn_calls"] <= 2:
                _say(f"pre_attn call: cls={type(self).__name__} attn_type={getattr(self, 'attention_type', None)} "
                     f"fused_single_qkv_rope={getattr(cfg, 'fused_single_qkv_rope', None)} test_mode={getattr(cfg, 'test_mode', None)} "
                     f"flash_decode={getattr(cfg, 'flash_decode', None)} output_gate={getattr(cfg, 'attention_output_gate', None)} "
                     f"HAVE_FUSED_QKV_ROPE={getattr(mcore_attention, 'HAVE_FUSED_QKV_ROPE', None)} "
                     f"inference_context={a.get('inference_context') is not None} inference_params={a.get('inference_params') is not None} "
                     f"packed_seq={a.get('packed_seq_params') is not None} kv_states={a.get('key_value_states') is not None} "
                     f"q_ln={type(self.q_layernorm).__name__} k_ln={type(self.k_layernorm).__name__} "
                     f"rope={type(rope).__name__}{(len(rope) if isinstance(rope, (tuple, list)) else '')} "
                     f"rope_shapes={[tuple(t.shape) if _torch.is_tensor(t) else None for t in rope] if isinstance(rope, (tuple, list)) else None} "
                     f"hs={tuple(hidden_states.shape) if _torch.is_tensor(hidden_states) else type(hidden_states).__name__} "
                     f"head_dim={self.hidden_size_per_attention_head} nqg={cfg.num_query_groups} world={self.world_size} "
                     f"ng_part={self.num_query_groups_per_partition} nh_part={self.num_attention_heads_per_partition}")
            # mirror Megatron's own fused_single_qkv_rope conditions: anything else takes the split q/k/v path, whose
            # consumers would misread a flat q|k|v buffer, so the fused GEMM must not run there
            if (
                self.attention_type != "cross"
                and not getattr(cfg, "qk_layernorm", False)
                and getattr(cfg, "fused_single_qkv_rope", False)
                and not getattr(cfg, "test_mode", False)
                and not getattr(cfg, "flash_decode", False)
                and not getattr(cfg, "attention_output_gate", False)
                and getattr(mcore_attention, "HAVE_FUSED_QKV_ROPE", False)
                and a.get("inference_context") is None and a.get("inference_params") is None
                and a.get("packed_seq_params") is None and a.get("key_value_states") is None
                and (self.q_layernorm is None or isinstance(self.q_layernorm, mcore_attention.IdentityOp))
                and (self.k_layernorm is None or isinstance(self.k_layernorm, mcore_attention.IdentityOp))
                and isinstance(rope, (tuple, list)) and len(rope) == 2 and rope[0] is not None and rope[1] is not None
                and _same_freqs(rope[0], rope[1])
                and _torch.is_tensor(hidden_states) and hidden_states.dim() == 3
                and self.hidden_size_per_attention_head == 64
                and (cfg.num_query_groups is None or cfg.num_query_groups >= self.world_size)
            ):
                s, b = int(hidden_states.shape[0]), int(hidden_states.shape[1])
                ng = self.num_query_groups_per_partition
                nqpg = self.num_attention_heads_per_partition // ng
                qf = rope[0]
                if qf.shape[-1] == 64 and qf.shape[0] >= s and (s * b) % 256 == 0 and (ng * (nqpg + 2) * 64) % 256 == 0:
                    t32 = _tables32(qf, s)
                    if t32 is not None:
                        ctx = {"cos32": t32[0], "sin32": t32[1], "s": s, "b": b, "ng": ng, "nqpg": nqpg,
                               "N": ng * (nqpg + 2) * 64}
                    elif _STATS["ctx_miss"] < 3:
                        _say(f"fused GEMM context miss: the two 32-wide halves of the rotary freqs differ (shape {tuple(qf.shape)})")
                else:
                    if _STATS["ctx_miss"] < 3:
                        _say(f"fused GEMM context miss: geometry freqs={tuple(qf.shape)} s={s} b={b} ng={ng} nqpg={nqpg}")
            elif _STATS["ctx_miss"] < 3:
                _say("fused GEMM context miss: Megatron fused-single-qkv-rope conditions not met (see the pre_attn call line)")
        except Exception as e:  # noqa
            _say(f"fused GEMM context skipped: {type(e).__name__}: {e}")
            ctx = None
        if ctx is None:
            _STATS["ctx_miss"] += 1
        _PENDING[0] = ctx
        try:
            out = _orig_pre(self, *args, **kwargs)
        finally:
            _PENDING[0] = None
        if _FUSED:
            # a fused GEMM output was produced but no RoPE Function consumed it: its consumer would misread the
            # flat q|k|v layout -- fail loudly rather than train on wrong math
            _FUSED.clear()
            raise RuntimeError("QKV_GEMM: fused qkv GEMM output was not consumed by the fused RoPE forward")
        return out

    mcore_attention.Attention.forward_pre_attn_and_core_attn = forward_pre_attn_and_core_attn

    # -------- 6. TE: the LayerNormLinear forward GEMM of the qkv projection
    _orig_gg = te_lnl.general_gemm
    _ORIG["general_gemm"] = _orig_gg

    def _why_not(A, B, ctx, out_dtype, quantization_params, gelu, alpha, beta, accumulate, layout, out, bias, ub, extra_output):
        if _ARCH[0] is None:
            _ARCH[0] = tuple(_torch.cuda.get_device_capability(getattr(B, "device", None) or _torch.cuda.current_device()))
        if _ARCH[0] != (10, 7):
            # the block-scaled GEMM is the CUTLASS Rubin example (cutlass.utils.rubin_helpers, sm_107 only); every
            # other part keeps the cuBLAS GEMM and the standalone RoPE/amax pass
            return f"block-scaled GEMM kernel is Rubin-only (sm_107); this device is sm_{_ARCH[0][0]}{_ARCH[0][1]}"
        if not (isinstance(A, MXFP8TensorStorage) and isinstance(B, MXFP8TensorStorage)):
            return f"operands {type(A).__name__}/{type(B).__name__} not MXFP8"
        if A._rowwise_data is None or B._rowwise_data is None:
            return "rowwise data missing"
        if not (getattr(A, "_with_gemm_swizzled_scales", False) and getattr(B, "_with_gemm_swizzled_scales", False)):
            return "scales not GEMM-swizzled"
        if A._fp8_dtype != tex.DType.kFloat8E4M3 or B._fp8_dtype != tex.DType.kFloat8E4M3:
            return f"fp8 dtypes {A._fp8_dtype}/{B._fp8_dtype}"
        N, K = A._rowwise_data.shape[-2], A._rowwise_data.shape[-1]
        if A._rowwise_data.dim() != 2 or N != ctx["N"]:
            return f"weight shape {tuple(A._rowwise_data.shape)} vs N={ctx['N']}"
        M = B._rowwise_data.numel() // B._rowwise_data.shape[-1]
        if B._rowwise_data.shape[-1] != K or M != ctx["s"] * ctx["b"]:
            return f"input shape {tuple(B._rowwise_data.shape)} vs M={ctx['s'] * ctx['b']} K={K}"
        if M % 256 or N % 256 or K % 32:
            return f"M={M} N={N} K={K} alignment"
        if out_dtype != _torch.bfloat16 or quantization_params is not None or gelu or out is not None:
            return f"out_dtype={out_dtype} quantizer={quantization_params is not None} gelu={gelu} out={out is not None}"
        if alpha != 1.0 or beta is not None or accumulate or layout != "TN":
            return f"alpha={alpha} beta={beta} accumulate={accumulate} layout={layout}"
        if bias is None or bias.dtype != _torch.bfloat16 or tuple(bias.shape) != (N,) or not bias.is_contiguous():
            return f"bias {None if bias is None else (bias.dtype, tuple(bias.shape))}"
        if ub is not None or extra_output is not None:
            return "userbuffers / extra output"
        if not (A._rowwise_data.is_contiguous() and B._rowwise_data.is_contiguous()):
            return "non-contiguous fp8 data"
        return None

    def general_gemm(A, B, out_dtype=None, quantization_params=None, gelu=False, gelu_in=None, alpha=1.0, beta=None,
                     accumulate=False, layout="TN", out=None, bias=None, use_split_accumulator=False, grad=False,
                     ub=None, ub_type=None, extra_output=None, bulk_overlap=False):
        ctx = _PENDING[0]
        _STATS["gemm_calls"] += 1
        if _STATS["gemm_calls"] <= 2:
            _say(f"layernorm_linear.general_gemm call: ctx={'set' if ctx is not None else 'none'} A={type(A).__name__}{tuple(getattr(A, 'shape', ()))} "
                 f"B={type(B).__name__}{tuple(getattr(B, 'shape', ()))} out_dtype={out_dtype} bias={None if bias is None else tuple(bias.shape)} "
                 f"layout={layout} quant={quantization_params is not None} ub={ub is not None}")
        if ctx is not None:
            _PENDING[0] = None  # one qkv GEMM per attention forward
            why = _why_not(A, B, ctx, out_dtype, quantization_params, gelu, alpha, beta, accumulate, layout, out, bias, ub, extra_output)
            if why is None:
                K = A._rowwise_data.shape[-1]
                M = ctx["s"] * ctx["b"]
                if _DUMP_DIR:
                    _dump("gemm_inputs", a_data=B._rowwise_data.view(M, K), a_scale_inv=B._rowwise_scale_inv,
                          w_data=A._rowwise_data, w_scale_inv=A._rowwise_scale_inv, bias=bias, cos32=ctx["cos32"],
                          sin32=ctx["sin32"], s=ctx["s"], b=ctx["b"], ng=ctx["ng"], nqpg=ctx["nqpg"],
                          a_fp8_dtype=str(B._fp8_dtype), w_fp8_dtype=str(A._fp8_dtype))
                flat, scale, amax = _torch.ops.gpt_oss.qkv_gemm_rope_fwd_amax(
                    B._rowwise_data.view(M, K), B._rowwise_scale_inv, A._rowwise_data, A._rowwise_scale_inv,
                    bias, ctx["cos32"], ctx["sin32"], ctx["b"], ctx["ng"], ctx["nqpg"])
                _FUSED[flat.data_ptr()] = {"scale": scale, "amax": amax, "numel": flat.numel()}
                _STATS["gemm_fused"] += 1
                if _STATS["gemm_fused"] <= 2:
                    _say(f"fused qkv GEMM+RoPE: A(ln_out) {tuple(B._rowwise_data.shape)} x W {tuple(A._rowwise_data.shape)} -> flat q|k|v [{flat.numel()}] + amax/scale")
                return flat.view(*B._rowwise_data.shape[:-1], ctx["N"]), None, None, extra_output
            _STATS["gemm_fallback"] += 1
            if _STATS["gemm_fallback"] <= 3:
                _say(f"fused qkv GEMM not applicable, cuBLAS: {why}")
        return _orig_gg(A, B, out_dtype=out_dtype, quantization_params=quantization_params, gelu=gelu, gelu_in=gelu_in,
                        alpha=alpha, beta=beta, accumulate=accumulate, layout=layout, out=out, bias=bias,
                        use_split_accumulator=use_split_accumulator, grad=grad, ub=ub, ub_type=ub_type,
                        extra_output=extra_output, bulk_overlap=bulk_overlap)

    te_lnl.general_gemm = general_gemm
    _GEMM_OK = True
    return ("Attention.forward_pre_attn_and_core_attn (RoPE context) + layernorm_linear.general_gemm -> "
            "gpt_oss::qkv_gemm_rope_fwd_amax (CuTe DSL Rubin block-scaled GEMM, bias+RoPE+relocation+amax epilogue)")
