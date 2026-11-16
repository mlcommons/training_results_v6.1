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
from __future__ import annotations

import inspect
import math
import os
import sys
from typing import Dict, Tuple

import torch


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


PLAN_INDEX = _env_int("OPT_ATTN_FWD_PLAN", 0)
_DEBUG = _env_int("OPT_ATTN_FWD_DEBUG", 0)


def _dbg_force(msg: str) -> None:
    sys.stderr.write(f"[opt attn_fwd rank {os.environ.get('RANK', '?')}] {msg}\n")
    sys.stderr.flush()


def _dbg(msg: str) -> None:
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[opt attn-fwd r{rank}] {msg}\n")
    sys.stderr.flush()


class CudnnFeFwd:

    def __init__(self, b: int, s: int, h: int, hk: int, d: int, attn_scale: float, device: torch.device,
                 plan_index: int = 0, heur_modes: str = "AB"):
        import cudnn

        self.cudnn = cudnn
        self.shape = (b, s, h, hk, d)
        self.attn_scale = float(attn_scale)
        self.handle = cudnn.create_handle()
        cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream(device).cuda_stream)
        f32, e4, bf = cudnn.data_type.FLOAT, cudnn.data_type.FP8_E4M3, cudnn.data_type.BFLOAT16
        g = cudnn.pygraph(io_data_type=e4, intermediate_data_type=f32, compute_data_type=f32, handle=self.handle)
        self.g = g
        self.tq = g.tensor(name="Q", dim=[b, h, s, d], stride=[h * d, d, b * h * d, 1], data_type=e4)
        self.tk = g.tensor(name="K", dim=[b, hk, s, d], stride=[hk * d, d, b * hk * d, 1], data_type=e4)
        self.tv = g.tensor(name="V", dim=[b, hk, s, d], stride=[hk * d, d, b * hk * d, 1], data_type=e4)
        sc = lambda n: g.tensor(name=n, dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=f32)
        self.t_dq, self.t_dk, self.t_dv = sc("descale_q"), sc("descale_k"), sc("descale_v")
        self.t_ds, self.t_ss, self.t_so = sc("descale_s"), sc("scale_s"), sc("scale_o")
        o, st, amax_s, amax_o = g.sdpa_fp8(
            q=self.tq, k=self.tk, v=self.tv, descale_q=self.t_dq, descale_k=self.t_dk, descale_v=self.t_dv,
            descale_s=self.t_ds, scale_s=self.t_ss, scale_o=self.t_so, is_inference=False,
            attn_scale=self.attn_scale, use_causal_mask=True, name="sdpa_fp8",
        )
        o.set_output(True).set_dim([b, h, s, d]).set_stride([h * d, d, b * h * d, 1]).set_data_type(bf)
        st.set_output(True).set_dim([b, h, s, 1]).set_stride([h * s, s, 1, 1]).set_data_type(f32)
        for t in (amax_s, amax_o):
            t.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(f32)
        self.t_o, self.t_st, self.t_amax_s, self.t_amax_o = o, st, amax_s, amax_o
        g.validate()
        g.build_operation_graph()
        modes = [getattr(cudnn.heur_mode, m) for m in heur_modes]
        g.create_execution_plans(modes)
        g.check_support()
        g.build_plans(policy=cudnn.build_plan_policy.ALL)
        self.num_plans = int(g.get_execution_plan_count())
        if not (0 <= plan_index < self.num_plans):
            raise ValueError(f"cudnn plan index {plan_index} out of range ({self.num_plans} plans)")
        self.plan_index = plan_index
        try:
            self.plan_name = str(g.get_plan_name_at_index(plan_index))
        except Exception:  # pragma: no cover - diagnostics only
            self.plan_name = "?"
        ws = int(g.get_workspace_size_plan_at_index(plan_index))
        self.workspace = torch.empty(max(1, ws), dtype=torch.uint8, device=device)
        self.scale_o = torch.ones(1, dtype=torch.float32, device=device)

    def __call__(self, q8, k8, v8, dq, dk, dv, ds, ss, out, stats, amax_s, amax_o) -> None:
        self.cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        vp = {
            self.tq: q8, self.tk: k8, self.tv: v8, self.t_o: out, self.t_st: stats,
            self.t_dq: dq, self.t_dk: dk, self.t_dv: dv, self.t_ds: ds, self.t_ss: ss, self.t_so: self.scale_o,
            self.t_amax_s: amax_s, self.t_amax_o: amax_o,
        }
        self.g.execute_plan_at_index(vp, self.workspace, self.plan_index, handle=self.handle)


class _RngConsumer:

    S, D = 64, 64

    def __init__(self, device: torch.device):
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (
            AttnBiasType, AttnMaskType, FusedAttnBackend, QKVFormat, QKVLayout, SoftmaxType)
        import transformer_engine_torch as tex

        self.tex = tex
        s, d = self.S, self.D
        self.q = torch.zeros((s, 1, 1, d), dtype=torch.bfloat16, device=device)
        self.k = torch.zeros((s, 1, 1, d), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((s, 1, 1, d), dtype=torch.bfloat16, device=device)
        self.cu = torch.tensor([0, s], dtype=torch.int32, device=device)
        self.enums = (QKVLayout["sbhd_sbhd_sbhd"], QKVFormat["sbhd"], QKVFormat[None], AttnBiasType["no_bias"],
                      AttnMaskType["no_mask"], SoftmaxType["vanilla"])
        self.backend = FusedAttnBackend["F16_arbitrary_seqlen"]

    def __call__(self, rng_elts_per_thread: int) -> torch.Tensor:
        s = self.S
        ql, of, sf, bt, mt, st = self.enums
        outs = self.tex.fused_attn_fwd(
            s, s, True, 1.0 / math.sqrt(self.D), 0.0, True, ql, of, sf, bt, mt, st, [-1, -1], False,
            self.cu, self.cu, self.q, self.k, self.v, torch.bfloat16, None, None, None, None, None, None,
            None, None, None, int(rng_elts_per_thread), False, bool(torch.cuda.is_current_stream_capturing()),
        )
        return outs[-1]


class _State:
    installed = False
    orig_fused_attn_fwd = None
    orig_quantize_impl = None
    sig = None
    backends: Dict[tuple, object] = {}
    pending: Dict[int, Tuple[int, torch.Tensor]] = {}
    rng_state = None
    rng_consumer = None
    scratch = None
    ring_i = 0
    RING = 128
    STRIDE = 8
    n_fwd = 0
    n_rng_advance = 0
    n_fallback = 0
    n_amax_reused = 0
    n_quant_fallback = 0
    debug_left = 0


def _ring_slot(zero: bool) -> torch.Tensor:
    st = _State
    if st.scratch is None:
        st.scratch = torch.zeros(st.RING * st.STRIDE, dtype=torch.float32, device="cuda")
    base = st.ring_i * st.STRIDE
    st.ring_i = (st.ring_i + 1) % st.RING
    slot = st.scratch[base: base + st.STRIDE]
    if zero:
        slot.zero_()
    return slot


def _supported(a: dict) -> bool:
    from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
    import transformer_engine_torch as tex

    try:
        if int(a["fused_attention_backend"]) != int(FusedAttnBackend["FP8"]):
            return False
    except Exception:
        return False
    q, k, v = a["q"], a["k"], a["v"]
    if not (isinstance(q, Float8Tensor) and isinstance(k, Float8Tensor) and isinstance(v, Float8Tensor)):
        return False
    e4 = tex.DType.kFloat8E4M3
    if any(int(t._fp8_dtype) != int(e4) for t in (q, k, v)):
        return False
    if not (a["is_training"] and a["qkv_layout"] == "sbhd_sbhd_sbhd" and a["o_format"] == "sbhd"):
        return False
    if a["attn_bias_type"] != "no_bias" or a["attn_mask_type"] != "causal" or a["softmax_type"] != "vanilla":
        return False
    if tuple(a["window_size"]) not in ((-1, 0), (-1, -1)) or a.get("bottom_right_diagonal"):
        return False
    if float(a["dropout"]) != 0.0 or a.get("attn_bias") is not None or a.get("softmax_offset") is not None:
        return False
    if a.get("cu_seqlens_q_padded") is not None or a.get("cu_seqlens_kv_padded") is not None:
        return False
    if a.get("page_table_k") is not None or a.get("page_table_v") is not None or a.get("return_max_logit"):
        return False
    if q._data.dim() != 4 or k._data.dim() != 4 or v._data.dim() != 4:
        return False
    s, b, h, d = q._data.shape
    sk, bk, hk, dk = k._data.shape
    if (sk, bk, dk) != (s, b, d) or tuple(v._data.shape) != tuple(k._data.shape):
        return False
    if d != 128 or h % hk != 0 or s % 256 != 0 or s != int(a["max_seqlen_q"]) or s != int(a["max_seqlen_kv"]):
        return False
    if not (q._data.is_contiguous() and k._data.is_contiguous() and v._data.is_contiguous()):
        return False
    sq = a.get("s_quantizer")
    if sq is None or not torch.is_tensor(getattr(sq, "amax", None)) or not torch.is_tensor(getattr(sq, "scale", None)):
        return False
    return True


def _get_backend(b, s, h, hk, d, attn_scale, device):
    key = (b, s, h, hk, d, round(float(attn_scale), 12))
    be_ = _State.backends.get(key)
    if be_ is None:
        be_ = CudnnFeFwd(b, s, h, hk, d, attn_scale, device, plan_index=PLAN_INDEX)
        _dbg(f"cudnn_fe graph built: plans={be_.num_plans} pinned={be_.plan_index} ({be_.plan_name}) "
             f"workspace={be_.workspace.numel()} B shape=b{b} s{s} h{h} kv{hk} d{d}")
        _State.backends[key] = be_
    return be_


def _fused_attn_fwd_patched(*args, **kwargs):
    st = _State
    ba = st.sig.bind(*args, **kwargs)
    ba.apply_defaults()
    a = ba.arguments
    if not _supported(a):
        st.n_fallback += 1
        return st.orig_fused_attn_fwd(*args, **kwargs)
    q, k, v = a["q"], a["k"], a["v"]
    sq = a["s_quantizer"]
    s, b, h, d = q._data.shape
    hk = k._data.shape[2]
    attn_scale = a["attn_scale"]
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(d)
    device = q._data.device
    be_ = _get_backend(b, s, h, hk, d, attn_scale, device)
    out = torch.empty((s, b, h, d), dtype=a["fake_dtype"], device=device)
    stats = torch.empty((b, h, s, 1), dtype=torch.float32, device=device)
    slot = _ring_slot(zero=False)
    amax_o, amax_r, descale_s = slot[0:1], slot[1:2], slot[2:3]
    if _env_int("OPT_ATTN_FWD_DIAG", 0) and st.n_fwd < 4 * 32 and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
        _dbg_force(f"fwd #{st.n_fwd}: scale_s={float(sq.scale.float().reshape(-1)[0])!r} amax_s={float(sq.amax.float().reshape(-1)[0])!r} "
                   f"scale_ptr={hex(sq.scale.data_ptr())} amax_ptr={hex(sq.amax.data_ptr())}")
    torch.reciprocal(sq.scale.reshape(1), out=descale_s)
    be_(q._data, k._data, v._data, q._scale_inv.reshape(1), k._scale_inv.reshape(1), v._scale_inv.reshape(1),
        descale_s, sq.scale.reshape(1), out, stats, sq.amax.reshape(1), amax_o)
    if _env_int("OPT_ATTN_FWD_DIAG", 0) and st.n_fwd < 4 * 32 and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
        _dbg_force(f"fwd #{st.n_fwd} after cudnn: amax_s={float(sq.amax.float().reshape(-1)[0])!r}")
    rng_elts = (int(a["max_seqlen_q"]) * int(a["max_seqlen_q"]) + 127) // 128
    rng_mode = _env_int("OPT_ATTN_FWD_RNG", 2)
    if rng_mode == 2:
        from opt_kernels.rht_cast_nvfp4 import _philox_ext

        _philox_ext().philox_cuda_state(rng_elts)
        if st.rng_state is None or st.rng_state.device != device:
            st.rng_state = torch.zeros(2, dtype=torch.int64, device=device)
        rng_state = st.rng_state
        st.n_rng_advance += 1
    elif rng_mode == 1:
        if st.rng_consumer is None:
            st.rng_consumer = _RngConsumer(device)
        rng_state = st.rng_consumer(rng_elts)
    else:
        if st.rng_state is None or st.rng_state.device != device:
            st.rng_state = torch.zeros(2, dtype=torch.int64, device=device)
        rng_state = st.rng_state
    st.n_fwd += 1
    if len(st.pending) > 64:
        st.pending.clear()
    st.pending[out.data_ptr()] = (out.numel(), amax_r, amax_o)
    if _DEBUG and st.debug_left > 0 and not torch.cuda.is_current_stream_capturing():
        st.debug_left -= 1
        try:
            torch.cuda.synchronize()
            s_amax_mine = float(sq.amax.float().reshape(-1)[0])
            ref_out, ref_aux = st.orig_fused_attn_fwd(*args, **kwargs)
            torch.cuda.synchronize()
            d_o = (out.float() - ref_out.float()).abs()
            d_st = (stats - ref_aux[0]).abs()
            amax_r.copy_(amax_o.to(torch.bfloat16))
            _dbg(f"fwd {st.n_fwd}: rng_state mine={rng_state.tolist()} te={ref_aux[1].tolist() if len(ref_aux) > 1 else None}")
            _dbg(f"fwd {st.n_fwd}: backend=cudnn_fe shape=s{s} b{b} h{h} kv{hk} d{d} scale_inv="
                 f"{[float(t._scale_inv.float().reshape(-1)[0]) for t in (q, k, v)]} s_scale={float(sq.scale.float().reshape(-1)[0])} "
                 f"| O vs TE: max_abs={float(d_o.max()):.3e} mean_abs={float(d_o.mean()):.3e} bitwise={bool(torch.equal(out, ref_out))} "
                 f"nan={int(torch.isnan(out).sum())} | stats vs TE: max_abs={float(d_st.max()):.3e} bitwise={bool(torch.equal(stats, ref_aux[0]))} "
                 f"| amax_o={float(amax_o.item()):.6f} amax_bf16={float(amax_r.item()):.6f} O_absmax={float(out.abs().amax().float()):.6f} "
                 f"eq={bool(float(amax_r.item()) == float(out.abs().amax().float()))} | s_amax mine={s_amax_mine:.7f} te={float(sq.amax.float().reshape(-1)[0]):.7f}")
            sq.amax.fill_(s_amax_mine)
        except Exception as exc:
            _dbg(f"  debug compare failed: {exc!r}")
    return out, [stats, rng_state]


def _round_amax(quantizer, amax_o: torch.Tensor, amax_r: torch.Tensor) -> None:
    amax_r.copy_(amax_o.to(torch.bfloat16))


def _quantize_impl_patched(self, tensor: torch.Tensor):
    st = _State
    ent = st.pending.pop(tensor.data_ptr(), None) if st.pending else None
    if ent is not None:
        import transformer_engine_torch as tex

        numel, amax_r, amax_o = ent
        ok = (
            numel == tensor.numel() and tensor.dtype == torch.bfloat16 and tensor.dim() >= 2
            and self.rowwise_usage and not self.columnwise_usage
            and not getattr(self, "stochastic_rounding", False)
            and not getattr(self, "with_amax_reduction", False)
            and not getattr(self, "row_scaled_nvfp4", False)
            and not getattr(self, "nvfp4_use_4over6", False)
            and (not self.with_rht or self.with_post_rht_amax)
        )
        if ok:
            st.n_amax_reused += 1
            _round_amax(self, amax_o, amax_r)
            if _DEBUG and st.debug_left > 0 and not torch.cuda.is_current_stream_capturing():
                try:
                    torch.cuda.synchronize()
                    ref = st.orig_quantize_impl(self, tensor)
                    mine = tex.nvfp4_quantize_with_amax(tensor, self, amax_r, amax_r)
                    torch.cuda.synchronize()
                    _dbg(f"quant {st.n_amax_reused}: amax ref={float(ref._amax_rowwise.float().item()):.6f} mine={float(amax_r.item()):.6f} "
                         f"data_eq={bool(torch.equal(ref._rowwise_data, mine._rowwise_data))} "
                         f"scale_eq={bool(torch.equal(ref._rowwise_scale_inv, mine._rowwise_scale_inv))} "
                         f"swz={getattr(ref, '_with_gemm_swizzled_scales', None)}/{getattr(mine, '_with_gemm_swizzled_scales', None)} "
                         f"rht={self.with_rht} post={self.with_post_rht_amax} col={self.columnwise_usage}")
                    return mine
                except Exception as exc:
                    _dbg(f"  quant debug compare failed: {exc!r}")
            return tex.nvfp4_quantize_with_amax(tensor, self, amax_r, amax_r)
        st.n_quant_fallback += 1
        if _DEBUG and st.debug_left > 0:
            _dbg(f"quant fallback: numel {numel}/{tensor.numel()} dtype={tensor.dtype} row={self.rowwise_usage} "
                 f"col={self.columnwise_usage} rht={self.with_rht} post={self.with_post_rht_amax} sr={getattr(self, 'stochastic_rounding', None)}")
    return st.orig_quantize_impl(self, tensor)


def install() -> None:
    st = _State
    if st.installed or _env_int("OPT_ATTN_FWD_DISABLE", 0):
        return
    import transformer_engine_torch as tex
    import transformer_engine.pytorch.attention.dot_product_attention.backends as dpa_backends
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if not hasattr(tex, "nvfp4_quantize_with_amax"):
        raise RuntimeError("transformer_engine_torch.nvfp4_quantize_with_amax is missing")
    st.orig_fused_attn_fwd = dpa_backends.fused_attn_fwd
    st.sig = inspect.signature(st.orig_fused_attn_fwd)
    dpa_backends.fused_attn_fwd = _fused_attn_fwd_patched
    st.orig_quantize_impl = NVFP4Quantizer.quantize_impl
    NVFP4Quantizer.quantize_impl = _quantize_impl_patched
    st.debug_left = _DEBUG * 8
    st.installed = True
    if _DEBUG:
        _dbg(f"installed backend=cudnn_fe plan={PLAN_INDEX}")
