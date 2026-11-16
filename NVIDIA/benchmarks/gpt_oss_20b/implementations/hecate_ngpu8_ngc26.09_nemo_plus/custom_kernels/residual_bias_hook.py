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

"""Fuse the residual and bias adds of the gpt-oss-20b transformer layer into their neighbours.  Installed after
norm_bwd_hook, whose state it shares; every rebinding wraps what is bound at install time and falls back to it.
  1. Attention proj bias -> the cuBLASLt BIAS epilogue of the MXFP8 GEMM instead of Megatron's unfused bias-dropout-add
     (24 x 216 us/step); the bias gradient comes from the partial sums the fused norm2 backward already computes.
  2. Residual add -> the MoE combine (gpt_oss::unpermute_add: HybridEP's sum with the residual row in the accumulator);
     the MLP bias-dropout-add recognises the fused output by data_ptr and returns it unchanged.
  3. Router-gradient add at the MoE fan-out -> the dispatch backward's unpermute, through an identity tap on the router
     input whose backward runs last and sums the parked row gradient with the router gradient in one pass.
  4. MOE_DGLU_STRIP=1 (default): the expert MLP emits the unscaled z = fc2(h) + b2 and the combine applies the routing
     probability (gpt_oss::unpermute_scale_add); its backward produces the MXFP8 fc2 output gradient, dprob and the fc2
     bias-gradient source selected by MOE_DGLU_STRIP_DBIAS2 in one pass and parks them for TE via moe_qperm_hook.
Gates: RESIDUAL_BIAS_FUSION=0 leaves the reference path; RESIDUAL_BIAS_FUSION_BIAS, RESIDUAL_BIAS_FUSION_COMBINE and
RESIDUAL_BIAS_FUSION_TAP switch the three parts (default on); RESIDUAL_BIAS_FUSION_SELFCHECK=0 skips the first-call
checks, which otherwise run before the CUDA graph is captured and keep the original path for the run when they fail.
"""
import os
import sys

import torch

import custom_kernels.norm_bwd_hook as NB   # shared state and the BDA/norm2 fast paths

_INSTALLED = False
_ORIG = {}
_STATS = {"proj_bias_fused_modules": 0, "attn_bda": 0, "attn_bda_side": 0, "gop_bias_fast": 0, "gop_bias_bf16": 0,
          "combine_fused": 0, "combine_plain": 0, "mlp_bda_identity": 0, "tap_fwd": 0, "dispatch_stashed": 0,
          "dispatch_plain": 0, "tap_fused": 0, "tap_plain": 0, "fallback": 0}
_MLP_RES_SLOT = {}       # id(_HybridEPManager) -> residual [s, b, h] recorded by the pre-MLP norm of that layer
_NEXT_COMBINE = [None]   # residual for the hybrid_ep_combine call made by the manager's combine() now running
_FUSED_COMBINE = {}      # combine output data_ptr -> (residual data_ptr, numel, probe): the MLP BDA returns x unchanged
                         # (probe=True on the first call's HybridEP output: only proves the data path, residual still added)
_BDA_REACHED = [False]   # the first call's combine output was seen at the MLP BDA with its residual (data-path check)
_TAP_SLOT = {}           # id(_HybridEPManager) -> _TapState created by route() of that layer
_NEXT_DISPATCH = [None]  # tap state for the hybrid_ep_dispatch call made by the manager's dispatch() now running
_COMBINE_OK = [True]
_COMBINE_CHECKED = [False]
_TAP_OK = [True]
_TAP_CHECKED = [False]
_NEXT_PROBS = [None]     # (dense routing probs [T, E] fp32, dispatched/permuted probs [P] fp32) of the combine now running
_DGLU_BWD_CHECKED = [False]
_DGLU_BWD_OK = [True]
_DGLU_ENGAGED = [False]  # set by moe_dglu_strip_hook.install(): the MLP output is the UNscaled z from then on

ENABLED = os.getenv("RESIDUAL_BIAS_FUSION", "1") == "1"
BIAS = os.getenv("RESIDUAL_BIAS_FUSION_BIAS", "1") == "1"
# The fused combine kernel runs 94-95 us at the default tile (16/256/4, KMAX=8) and replaces HybridEP's unpermute_kernel
# (76.9 us in situ) plus the standalone [24576, 2880] bf16 add it absorbs (41-48 us in situ): the residual add in the
# forward (2a) and the router-gradient add in the backward (2b).  RESIDUAL_BIAS_FUSION_COMBINE=0 / RESIDUAL_BIAS_FUSION_TAP=0
# switch either one off.
COMBINE = os.getenv("RESIDUAL_BIAS_FUSION_COMBINE", "1") == "1"
TAP = os.getenv("RESIDUAL_BIAS_FUSION_TAP", "1") == "1"
SELFCHECK = os.getenv("RESIDUAL_BIAS_FUSION_SELFCHECK", "1") == "1"
DGLU = os.getenv("MOE_DGLU_STRIP", "1") == "1"   # routing probability applied in the combine (module doc, 4.)
REL_TOL = 2.0 ** -6      # fused (one rounding) vs original (two roundings) of bf16 values, relative to max|ref|
MAX_LOG = 3


def _say(msg: str) -> None:
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"RESIDUAL_BIAS_FUSION rank={rank} {msg}\n")
    sys.__stdout__.flush()


def _fallback(what: str) -> None:
    _STATS["fallback"] += 1
    if _STATS["fallback"] <= 12:
        _say(f"fallback: {what}")


def _close(a: torch.Tensor, b: torch.Tensor):
    d = (a.float() - b.float()).abs().max().item()
    s = b.float().abs().max().item()
    return d <= REL_TOL * max(s, 1e-30), d, s


# ------------------------------------------------------------------------------------------------------------
# 1. proj bias -> GEMM epilogue.  Megatron builds the attention output projection as a TERowParallelLinear with
#    skip_bias_add=True, so TE returns the bias and Megatron adds it in the unfused bias-dropout-add.  For the 'proj'
#    module with tp_size == 1 (no TP reduction the bias would have to wait for) te_return_bias / return_bias /
#    gemm_bias_unfused_add are cleared and apply_bias is set, so TE passes the bias to cuBLASLt's BIAS epilogue (the qkv
#    LayerNormLinear of this model already runs it on MXFP8) and owns the bias gradient in the backward (use_bias).
# ------------------------------------------------------------------------------------------------------------
def _flip_proj_bias(mod, buffer_name, module_name):
    is_proj = buffer_name == "proj" or (isinstance(module_name, str) and module_name.endswith("linear_proj"))
    if not is_proj:
        return
    if not (getattr(mod, "te_return_bias", False) and getattr(mod, "use_bias", False)):
        _fallback(f"proj module without a returned bias (te_return_bias={getattr(mod, 'te_return_bias', None)})")
        return
    if getattr(mod, "tp_size", 0) != 1 or getattr(mod, "parallel_mode", None) != "row":
        _fallback(f"proj tp_size={getattr(mod, 'tp_size', None)} parallel_mode={getattr(mod, 'parallel_mode', None)}; bias stays unfused")
        return
    if not hasattr(mod, "gemm_bias_unfused_add") or not hasattr(mod, "apply_bias"):
        _fallback("TE Linear without gemm_bias_unfused_add/apply_bias; bias stays unfused")
        return
    mod.te_return_bias = False
    mod.return_bias = False
    mod.apply_bias = True
    mod.gemm_bias_unfused_add = False
    _STATS["proj_bias_fused_modules"] += 1
    if _STATS["proj_bias_fused_modules"] <= MAX_LOG:
        _say(f"proj bias fused into the GEMM epilogue ({module_name or 'linear_proj'}: in={mod.in_features} out={mod.out_features})")


def _install_proj_bias(TERowParallelLinear):
    orig_init = TERowParallelLinear.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            _flip_proj_bias(self, kwargs.get("tp_comm_buffer_name"), kwargs.get("name"))
        except Exception as ex:  # noqa
            _fallback(f"proj bias flip raised {type(ex).__name__}: {ex}")

    TERowParallelLinear.__init__ = __init__
    _ORIG["TERowParallelLinear.__init__"] = orig_init


# ------------------------------------------------------------------------------------------------------------
# 1b. attention bias-dropout-add with bias=None (+ 2a: MLP bias-dropout-add identity after a fused combine).
#     grad_output_preprocess is wrapped so the MXFP8 gradient the fused norm2 backward hands the proj Linear is accepted
#     together with the dbias partial sums that pass already computes (TE's own path would bgrad_quantize a bf16 grad).
# ------------------------------------------------------------------------------------------------------------
class _AttnBDANoBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, res_key):
        ctx.res_key = int(res_key)
        return residual + x                       # Megatron's `residual + out` (hidden_dropout = 0 is the identity)

    @staticmethod
    def backward(ctx, d_out):
        side = NB._GRAD_SIDE.pop(d_out.data_ptr(), None)
        if side is not None:
            d_x, dbias = side                     # MXFP8 grad + dbias partial sums from the fused norm2 backward
            d_x._fused_dbias = dbias            # proj's grad_output_preprocess returns it as TE's bias gradient
            _STATS["attn_bda_side"] += 1
        else:
            d_x = d_out
        if ctx.res_key:
            if NB._PENDING_RES1:
                raise RuntimeError("RESIDUAL_BIAS_FUSION: a stashed residual gradient was never consumed by norm1's fused "
                                   f"backward (keys {list(NB._PENDING_RES1)}); refusing to continue")
            NB._PENDING_RES1[ctx.res_key] = d_out
            d_res = None
        else:
            d_res = d_out
        _STATS["attn_bda"] += 1
        if _STATS["attn_bda"] <= MAX_LOG:
            _say(f"attention BDA (bias in epilogue) bwd: proj grad {'MXFP8 + dbias' if side is not None else 'bf16 (TE bgrad_quantize)'}"
                 f", residual grad {'stashed for the fused norm1 pass' if ctx.res_key else 'returned'}")
        return d_x, d_res, None


def _bias_dropout_add_func(x_with_bias, residual, prob, training, *rest):
    x, bias = x_with_bias
    if (bias is None and not rest and prob == 0.0 and torch.is_tensor(x) and torch.is_tensor(residual)
            and x.dtype == residual.dtype == torch.bfloat16 and x.shape == residual.shape):
        f = _FUSED_COMBINE.pop(x.data_ptr(), None)
        if f is not None:
            res_ptr, res_numel, probe = f
            if (res_ptr, res_numel) != (residual.data_ptr(), residual.numel()):
                raise RuntimeError("RESIDUAL_BIAS_FUSION: a fused combine output reached a bias-dropout-add with a different "
                                   "residual; refusing to continue")
            if probe:
                # first-call data-path check: HybridEP's own output arrived here with the recorded residual, so a fused
                # output (residual already inside) will take the identity return below; the residual is still added here
                _BDA_REACHED[0] = True
                _say(f"combine self-check: HybridEP's combine output reaches the MLP bias-dropout-add with its residual "
                     f"{tuple(x.shape)}; fused combine active from the next call")
                return _ORIG["_bias_dropout_add_func"](x_with_bias, residual, prob, training, *rest)
            _STATS["mlp_bda_identity"] += 1
            if _STATS["mlp_bda_identity"] <= MAX_LOG:
                _say(f"MLP BDA: residual already inside the combine output {tuple(x.shape)}; identity")
            return x
        if (BIAS and training and torch.is_grad_enabled() and (x.requires_grad or residual.requires_grad)
                and NB.ENABLED and NB.FUSE1 and residual.requires_grad
                and NB._NORM1_INPUTS.pop(residual.data_ptr(), None) == residual.numel()):
            # the attention BDA: its residual is the layer input the fused norm1 backward will consume
            if NB._PENDING_RES1:
                raise RuntimeError("RESIDUAL_BIAS_FUSION: a stashed residual gradient from the previous backward was never "
                                   f"consumed (keys {list(NB._PENDING_RES1)}); refusing to continue")
            return _AttnBDANoBias.apply(x, residual, residual.data_ptr())
    return _ORIG["_bias_dropout_add_func"](x_with_bias, residual, prob, training, *rest)


def _make_grad_output_preprocess(prev):
    def grad_output_preprocess(ctx, grad_output, row_parallel_mode, quantizer):
        dbias = getattr(grad_output, "_fused_dbias", None)
        if dbias is not None and getattr(grad_output, "_fused_mxfp8", False) and getattr(ctx, "use_bias", False):
            gather = row_parallel_mode and ctx.sequence_parallel
            use_fp8_bwd = ctx.fp8 and ctx.backward_override is None
            recipe_ok = False
            try:
                from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
                recipe_ok = isinstance(quantizer, MXFP8Quantizer) and quantizer.dtype == grad_output._fp8_dtype
            except Exception:  # noqa
                recipe_ok = False
            if (use_fp8_bwd and not gather and not getattr(ctx, "debug", False)
                    and not getattr(ctx, "ub_overlap_ag", False) and recipe_ok
                    and dbias.numel() == grad_output.shape[-1]):
                _STATS["gop_bias_fast"] += 1
                if _STATS["gop_bias_fast"] <= MAX_LOG:
                    _say(f"grad_output_preprocess: MXFP8 grad {tuple(grad_output.shape)} + precomputed dbias passed to the proj GEMMs")
                return grad_output, dbias.reshape(-1)
            _STATS["gop_bias_bf16"] += 1
            if _STATS["gop_bias_bf16"] <= MAX_LOG:
                _say(f"grad_output_preprocess: outside the fast path (fp8={ctx.fp8} gather={gather} recipe_ok={recipe_ok}); "
                     "bf16 twin, TE computes the bias gradient")
            grad_output = grad_output._fused_bf16
        return prev(ctx, grad_output, row_parallel_mode, quantizer)
    return grad_output_preprocess


# ------------------------------------------------------------------------------------------------------------
# shared: HybridEP handle inspection
# ------------------------------------------------------------------------------------------------------------
def _handle_parts(handle, x, num_tokens_expected=None):
    """(dense_to_expert_map, dense_chunk_layout, T) of a dense-layout HybridEP handle, or (None, reason)."""
    if not isinstance(handle, (tuple, list)) or len(handle) != 12:
        return None, f"handle with {len(handle) if isinstance(handle, (tuple, list)) else 'no'} entries"
    layout, m, T = handle[5], handle[6], handle[10]
    if not (torch.is_tensor(layout) and torch.is_tensor(m)) or isinstance(T, bool) or not isinstance(T, int):
        return None, "handle entries 5/6/10 are not (tensor, tensor, int)"
    if m.dim() != 2 or m.dtype != torch.int32 or not m.is_contiguous() or m.shape[0] < T:
        return None, f"dense_to_expert_map {tuple(m.shape)} {m.dtype}"
    if layout.dtype != torch.int32 or layout.numel() < 1 or not layout.is_contiguous():
        return None, f"dense_chunk_layout {tuple(layout.shape)} {layout.dtype}"
    if not (torch.is_tensor(x) and x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] % 8 == 0):
        return None, f"permuted rows {tuple(x.shape) if torch.is_tensor(x) else type(x)} {getattr(x, 'dtype', None)}"
    if num_tokens_expected is not None and num_tokens_expected != T * x.shape[1]:
        return None, f"token count mismatch (handle {T} rows x {x.shape[1]} vs {num_tokens_expected} elements)"
    return (m, layout, T), ""


def _hybridep_manager(moe):
    td = getattr(moe, "token_dispatcher", None)
    mgr = getattr(td, "_comm_manager", None)
    if mgr is None or type(mgr).__name__ != "_HybridEPManager":
        return None
    if getattr(td, "shared_experts", None) is not None or getattr(moe, "shared_experts", None) is not None:
        return None
    if getattr(moe, "moe_layer_recompute", False):
        return None
    return mgr


# ------------------------------------------------------------------------------------------------------------
# 2a. combine + residual.  TransformerLayer._pre_mlp_layernorm_and_residual records the residual per MoE layer and
#     _HybridEPManager.combine hands it to the rebound token_dispatcher.hybrid_ep_combine (HybridEP at EP=1 is the dense
#     unpermute kernel).  Backward = HybridEP's dispatch_with_permute, residual gradient passed through.  First-call
#     self-check: the kernel without the add must be bitwise HybridEP's output and with the add within bf16 rounding of
#     HybridEP + residual; HybridEP's own output of that call must then reach the MLP bias-dropout-add with the recorded
#     residual (proves the identity return cannot leave a residual added twice).  Other dtypes, padded dispatch inputs,
#     permute fusion, shared experts, MoE-layer recompute or a non-HybridEP dispatcher keep the original path.
# ------------------------------------------------------------------------------------------------------------
class _CombineAddResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, handle, num_permuted_tokens, pad_multiple, fused):
        m, layout, T = handle[6], handle[5], handle[10]
        out = torch.ops.gpt_oss.unpermute_add(x, m, layout, residual.reshape(T, x.shape[1]), T)
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.fused = fused
        ctx.res_shape = residual.shape
        return out

    @staticmethod
    def backward(ctx, grad_x):
        from megatron.core.transformer.moe import fused_a2a
        grad_x = grad_x.contiguous()
        dispatched_hidden, _, _, _, _ = fused_a2a._hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=ctx.handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
            **({"fuse_permute_dispatch": ctx.fused} if ctx.fused else {}),
        )
        return dispatched_hidden, grad_x.view(ctx.res_shape), None, None, None, None


class _CombineScaleAddResidual(torch.autograd.Function):
    """Combine with the routing probability applied: out = residual + sum_e probs[t, e] * z[map[t, e]]; backward = one
    gather pass producing the MXFP8 fc2 output gradient (parked for TE) and the permuted-probability gradient dprob = dres . z."""

    @staticmethod
    def forward(ctx, z, residual, pprob, probs_dense, handle, num_permuted_tokens, pad_multiple, fused):
        m, layout, T = handle[6], handle[5], handle[10]
        out = torch.ops.gpt_oss.unpermute_scale_add(z, m, layout, probs_dense, residual.reshape(T, z.shape[1]), T)
        ctx.save_for_backward(z, pprob)
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.fused = fused
        ctx.res_shape = residual.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        import custom_kernels.moe_qperm_hook as QH
        z, pprob = ctx.saved_tensors
        handle = ctx.handle
        m, layout, tpe = handle[6], handle[5], handle[7]
        P = ctx.num_permuted_tokens
        dres = grad_out.contiguous()
        T, H = dres.shape
        E = tpe.numel()
        tpe32 = tpe.to(torch.int32).contiguous()
        r2t, _ = QH._row_to_token(handle, m.contiguous(), dres, layout.contiguous(), P, False)
        mode = QH.DBIAS2_MODE                                    # where the fc2 bias gradient comes from (gemm / bf16 / fused)
        rw, rws, cw, cws, dprob, dz_rows, dbias2 = torch.ops.gpt_oss.moe_qperm_gather_quant_scaled(
            dres, r2t, pprob, z, tpe32, P, ctx.pad_multiple, mode == "bf16", mode == "fused",
            QH.RW_LAYOUT, QH.CW_LAYOUT, QH.DIV_MODE, QH.CW_PARAMS)
        placeholder = QH.park_backward(P, H, E, rw, rws, cw, cws, dres.device,
                                       rows=dz_rows if mode == "bf16" else None,
                                       dbias=dbias2 if mode == "fused" else None)
        if SELFCHECK and not _DGLU_BWD_CHECKED[0]:
            _DGLU_BWD_CHECKED[0] = True
            ok = False
            dz_ref = dprob_ref = None
            try:
                with torch.no_grad():
                    padded = QH._padded_counts(None, tpe, ctx.pad_multiple)
                    n_used = int(padded.to(torch.int64).sum().item())
                    tok = r2t.to(torch.int64)
                    valid = (tok >= 0) & (tok < T)
                    rows = torch.where(valid[:, None], dres[tok.clamp(min=0)].float(), torch.zeros((), device=dres.device))
                    dz_ref = (rows * pprob[:, None]).to(torch.bfloat16)
                    dprob_ref = torch.where(valid, (rows * z.float()).sum(dim=1), torch.zeros((), device=dres.device))
                    same_q = QH._compare_quantized("combine-bwd", rw, rws, cw, cws, dz_ref.contiguous(), padded, P, H)
                    d = (dprob[:n_used] - dprob_ref[:n_used]).abs().max().item()
                    sc = dprob_ref[:n_used].abs().max().item()
                    close = d <= 2.0 ** -12 * max(sc, 1e-30)
                    extra = ""
                    if mode == "bf16":
                        rows_bit = torch.equal(dz_rows[:n_used], dz_ref[:n_used])
                        extra = f"; bf16 dz rows bitwise bf16(p * dres) = {rows_bit}"
                        close = close and rows_bit
                    elif mode == "fused":
                        import custom_kernels.moe_qperm as K
                        db_ref = K.grouped_colsum_reference(dz_ref, tpe32, ctx.pad_multiple)
                        dbd = (dbias2 - db_ref).abs().max().item()
                        dbs = db_ref.abs().max().item()
                        db_ok = dbd <= 2.0 ** -12 * max(dbs, 1e-30)
                        extra = (f"; fused fc2 dbias (fp32 column sum of the bf16 dz rows) vs torch: max|diff|={dbd:.3e} "
                                 f"max|ref|={dbs:.3e} ({'OK' if db_ok else 'FAIL'})")
                        close = close and db_ok
                    _say(f"combine-bwd self-check [dbias2={mode}]: dz MXFP8 bitwise tex.group_quantize(bf16(p * dres)) = {same_q}; "
                         f"dprob max|diff|={d:.3e} max|ref|={sc:.3e} ({'OK' if close else 'FAIL'}){extra}; T={T} H={H} E={E} P={P} n_used={n_used}")
                    ok = bool(same_q and close)
            except Exception as ex:  # noqa
                _say(f"combine-bwd self-check raised {type(ex).__name__}: {ex}")
            if not ok:
                _DGLU_BWD_OK[0] = False
                _COMBINE_OK[0] = False
                _say("combine-bwd self-check FAILED: this backward takes the torch path; the fused combine is off for the rest of the run")
                if dz_ref is None or dprob_ref is None:
                    raise RuntimeError("RESIDUAL_BIAS_FUSION: combine-bwd self-check could not build its reference; refusing to continue")
                QH.unpark(placeholder)
                placeholder = dz_ref.contiguous()          # real rows: TE quantizes them itself
                dprob = dprob_ref
        _STATS["combine_bwd_fused"] = _STATS.get("combine_bwd_fused", 0) + 1
        if _STATS["combine_bwd_fused"] <= MAX_LOG:
            what = {"gemm": "", "bf16": " + bf16 dz rows (TE's dbias pass reads them)", "fused": " + fp32 fc2-dbias column sum"}[mode]
            _say(f"combine bwd: gather + probability scale + MXFP8 quantize + dprob{what} in one pass [{T}, {H}] -> [{P}, {H}]")
        return placeholder, dres.view(ctx.res_shape), dprob, None, None, None, None, None


def _combine_dglu(x, handle, num_permuted_tokens, pad_multiple, fused, res, probs, orig):
    """The combine with the routing probability applied here (the MLP output x is the UNscaled z)."""
    if probs is None or not torch.is_tensor(probs[1]):
        raise RuntimeError("RESIDUAL_BIAS_FUSION: MOE_DGLU_STRIP=1 but the combine has no dispatched probabilities to apply "
                           "(hybrid_ep_combine reached outside _HybridEPManager.combine); refusing to continue")
    pprob = probs[1].reshape(-1)
    if pprob.dtype != torch.float32:
        pprob = pprob.to(torch.float32)
    pdense = probs[0]

    def fallback():
        _STATS["combine_plain"] += 1
        xs = (x.float() * pprob.unsqueeze(1)).to(torch.bfloat16)
        return orig(xs, handle, num_permuted_tokens, pad_multiple, fused)

    if res is None or fused or not _COMBINE_OK[0]:
        return fallback()
    parts, why = _handle_parts(handle, x, res.numel())
    if parts is None or res.dtype != torch.bfloat16 or not res.is_contiguous():
        _fallback(f"combine+residual: {why or 'residual dtype/layout'}")
        return fallback()
    m, layout, T = parts
    if pprob.numel() != x.shape[0]:
        _fallback(f"combine+residual: dispatched probs {tuple(pprob.shape)} vs {x.shape[0]} permuted rows")
        return fallback()
    if not (torch.is_tensor(pdense) and pdense.dim() == 2 and pdense.shape[0] >= T and pdense.shape[1] == m.shape[1]
            and pdense.dtype == torch.float32):
        _fallback(f"combine+residual: dense probs {getattr(pdense, 'shape', None)} {getattr(pdense, 'dtype', None)}")
        return fallback()
    pdense_d = pdense.detach().contiguous()
    if SELFCHECK and not _COMBINE_CHECKED[0]:
        out_ref = fallback()
        ok = False
        try:
            with torch.no_grad():
                xd = x.detach()
                fused_out = torch.ops.gpt_oss.unpermute_scale_add(xd, m, layout, pdense_d, res.detach().reshape(T, xd.shape[1]), T)
                ref_add = out_ref.detach() + res.detach().reshape(T, xd.shape[1])
                close, d, s = _close(fused_out, ref_add)
                ok = bool(close)
                _say(f"combine self-check (probability in the combine): kernel vs HybridEP(bf16(z p)) + residual "
                     f"max|diff|={d:.3e} max|ref|={s:.3e} ({'OK' if close else 'FAIL'}); T={T} E={m.shape[1]} P={xd.shape[0]}")
        except Exception as ex:  # noqa
            _say(f"combine self-check raised {type(ex).__name__}: {ex}")
        _COMBINE_CHECKED[0] = True
        if not ok:
            _COMBINE_OK[0] = False
            _say("combine self-check FAILED: HybridEP combine of bf16(z p) + separate residual add for this run")
        elif out_ref.data_ptr() in _FUSED_COMBINE:
            _COMBINE_OK[0] = False
            _say("combine self-check FAILED: HybridEP's output pointer is already marked; fused combine off for this run")
        else:
            _FUSED_COMBINE[out_ref.data_ptr()] = (res.data_ptr(), res.numel(), True)
        return out_ref
    if SELFCHECK and not _BDA_REACHED[0]:
        _fallback("combine+residual: the data-path check has not passed yet; HybridEP combine of bf16(z p) + separate add")
        return fallback()
    try:
        out = _CombineScaleAddResidual.apply(x, res, pprob, pdense_d, handle, num_permuted_tokens, pad_multiple, fused)
    except Exception as ex:  # noqa
        _fallback(f"combine+residual (probability in the combine) raised {type(ex).__name__}: {ex}")
        return fallback()
    if len(_FUSED_COMBINE) >= 64:
        _FUSED_COMBINE.clear()
    _FUSED_COMBINE[out.data_ptr()] = (res.data_ptr(), res.numel(), False)
    _STATS["combine_fused"] += 1
    if _STATS["combine_fused"] <= MAX_LOG:
        _say(f"combine fused with the probability multiply and the residual add: [{T}, {x.shape[1]}] from {x.shape[0]} permuted rows")
    return out


def _wrap_pre_mlp(cur):
    def _pre_mlp_layernorm_and_residual(self, hidden_states, *args, **kwargs):
        out = cur(self, hidden_states, *args, **kwargs)
        if COMBINE and _COMBINE_OK[0]:
            if _FUSED_COMBINE:
                if all(v[2] for v in _FUSED_COMBINE.values()):
                    # the first call's probe marker never reached the MLP bias-dropout-add: a different BDA path (or a
                    # copy between the combine and the add) -- the fused output could not be recognised there, so
                    # the fusion stays off for the run (HybridEP's output had its residual added by the original path)
                    _FUSED_COMBINE.clear()
                    _COMBINE_OK[0] = False
                    _say("combine self-check FAILED: HybridEP's combine output never reached the MLP bias-dropout-add "
                         "with its residual; residual stays in the MLP bias-dropout-add for this run")
                else:
                    # a fused output (residual already inside) never reached its bias-dropout-add: the layer output
                    # would carry the residual twice -- never continue silently
                    raise RuntimeError("RESIDUAL_BIAS_FUSION: a fused combine output never reached its bias-dropout-add "
                                       f"({len(_FUSED_COMBINE)} marker(s)); refusing to continue")
            try:
                mgr = _hybridep_manager(getattr(self, "mlp", None))
                if mgr is None:
                    _STATS["no_manager"] = _STATS.get("no_manager", 0) + 1
                    if _STATS["no_manager"] <= 1:
                        _fallback(f"layer {getattr(self, 'layer_number', '?')}: mlp {type(getattr(self, 'mlp', None)).__name__} has no HybridEP flex dispatcher (or shared experts / recompute); combine+residual off there")
                if mgr is not None and isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
                    if len(_MLP_RES_SLOT) >= 64:
                        _MLP_RES_SLOT.clear()
                    _MLP_RES_SLOT[id(mgr)] = out[1]
            except Exception as ex:  # noqa
                _fallback(f"residual record raised {type(ex).__name__}: {ex}")
        return out
    return _pre_mlp_layernorm_and_residual


def _wrap_manager_combine(orig):
    def combine(self, hidden_states, *args, **kwargs):
        res = _MLP_RES_SLOT.pop(id(self), None)
        if res is not None and getattr(self, "_padded_num_tokens", None) not in (None, getattr(self, "_original_num_tokens", None)):
            _fallback("HybridEP padded the dispatch inputs; combine+residual off for this call")
            res = None
        _NEXT_COMBINE[0] = res
        _NEXT_PROBS[0] = (getattr(self, "token_probs", None), getattr(self, "dispatched_probs", None)) if DGLU else None
        try:
            return orig(self, hidden_states, *args, **kwargs)
        finally:
            _NEXT_COMBINE[0] = None
            _NEXT_PROBS[0] = None
    return combine


def _hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple, fused=False):
    orig = _ORIG["hybrid_ep_combine"]
    res = _NEXT_COMBINE[0]
    _NEXT_COMBINE[0] = None
    probs = _NEXT_PROBS[0]
    _NEXT_PROBS[0] = None
    if DGLU and _DGLU_ENGAGED[0]:
        return _combine_dglu(x, handle, num_permuted_tokens, pad_multiple, fused, res, probs, orig)
    if res is None or fused or not _COMBINE_OK[0]:
        _STATS["combine_plain"] += 1
        return orig(x, handle, num_permuted_tokens, pad_multiple, fused)
    parts, why = _handle_parts(handle, x, res.numel())
    if parts is None or res.dtype != torch.bfloat16 or not res.is_contiguous():
        _fallback(f"combine+residual: {why or 'residual dtype/layout'}")
        _STATS["combine_plain"] += 1
        return orig(x, handle, num_permuted_tokens, pad_multiple, fused)
    m, layout, T = parts
    if SELFCHECK and not _COMBINE_CHECKED[0]:
        # first call: HybridEP's own combine builds this layer's graph; the kernel is compared against it
        out_ref = orig(x, handle, num_permuted_tokens, pad_multiple, fused)
        ok = False
        try:
            with torch.no_grad():
                xd = x.detach()
                plain = torch.ops.gpt_oss.unpermute_add(xd, m, layout, None, T)
                bitwise = torch.equal(plain, out_ref.detach())
                fused_out = torch.ops.gpt_oss.unpermute_add(xd, m, layout, res.detach().reshape(T, xd.shape[1]), T)
                ref_add = out_ref.detach() + res.detach().reshape(T, xd.shape[1])
                close, d, s = _close(fused_out, ref_add)
                ok = bool(bitwise and close)
                _say(f"combine self-check: plain kernel bitwise HybridEP = {bitwise}; +residual max|diff|={d:.3e} "
                     f"max|ref|={s:.3e} ({'OK' if close else 'FAIL'}); T={T} E={m.shape[1]} P={xd.shape[0]}")
        except Exception as ex:  # noqa
            _say(f"combine self-check raised {type(ex).__name__}: {ex}")
        _COMBINE_CHECKED[0] = True
        if not ok:
            _COMBINE_OK[0] = False
            _say("combine self-check FAILED: residual stays in the MLP bias-dropout-add for this run")
        elif out_ref.data_ptr() in _FUSED_COMBINE:
            _COMBINE_OK[0] = False
            _say("combine self-check FAILED: HybridEP's output pointer is already marked; residual stays in the MLP bias-dropout-add for this run")
        else:
            _FUSED_COMBINE[out_ref.data_ptr()] = (res.data_ptr(), res.numel(), True)   # data-path probe (see the BDA)
        _STATS["combine_plain"] += 1
        return out_ref
    if SELFCHECK and not _BDA_REACHED[0]:
        _fallback("combine+residual: the data-path check has not passed yet; HybridEP combine + separate add")
        _STATS["combine_plain"] += 1
        return orig(x, handle, num_permuted_tokens, pad_multiple, fused)
    try:
        out = _CombineAddResidual.apply(x, res, handle, num_permuted_tokens, pad_multiple, fused)
    except Exception as ex:  # noqa
        _fallback(f"combine+residual raised {type(ex).__name__}: {ex}")
        _STATS["combine_plain"] += 1
        return orig(x, handle, num_permuted_tokens, pad_multiple, fused)
    if len(_FUSED_COMBINE) >= 64:
        _FUSED_COMBINE.clear()
    _FUSED_COMBINE[out.data_ptr()] = (res.data_ptr(), res.numel(), False)
    _STATS["combine_fused"] += 1
    if _STATS["combine_fused"] <= MAX_LOG:
        _say(f"combine fused with the residual add: [{T}, {x.shape[1]}] from {x.shape[0]} permuted rows")
    return out


# ------------------------------------------------------------------------------------------------------------
# 2b. router-gradient tap + dispatch backward.  pre_mlp_layernorm_output feeds both the router and the dispatcher, so
#     autograd sums their two [s, b, h] gradients.  MoELayer.route passes the router input through _Tap (created before
#     the router, so its backward node has the lowest sequence number of the fan-out and runs last); _DispatchTap's
#     backward computes only the probability gradient (gpt_oss::probs_grad_gather) and parks the row gradient, which
#     _Tap.backward unpermutes and adds to the router gradient in one gpt_oss::unpermute_add pass.  Either order is
#     correct: a tap that fires first marks itself done and the dispatch backward takes HybridEP's path.  First-call
#     self-check: the probability gradient bitwise HybridEP's, the fused gradient within bf16 rounding of row + router grad.
# ------------------------------------------------------------------------------------------------------------
class _TapState:
    __slots__ = ("numel", "stash", "tap_done")

    def __init__(self, numel):
        self.numel = numel
        self.stash = None          # (grad of the dispatched rows [P, H], handle, pad_multiple, HybridEP's row grad or None)
        self.tap_done = False


class _Tap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, st):
        ctx.st = st
        _STATS["tap_fwd"] += 1
        return h.view_as(h)

    @staticmethod
    def backward(ctx, d_router):
        st = ctx.st
        st.tap_done = True
        s = st.stash
        st.stash = None
        if s is None:
            _STATS["tap_plain"] += 1
            if _STATS["tap_plain"] <= MAX_LOG:
                _say("tap bwd before the dispatch bwd: router grad returned plain (autograd adds)")
            return d_router, None
        grad_rows, handle, pad_multiple, ref_rows = s
        m, layout, T = handle[6], handle[5], handle[10]
        H = grad_rows.shape[1]
        d2 = d_router.contiguous().view(T, H)
        out = None
        try:
            out = torch.ops.gpt_oss.unpermute_add(grad_rows, m, layout, d2, T)
            if ref_rows is not None:              # first-call self-check against HybridEP's own unpermute
                close, d, sc = _close(out, ref_rows + d2)
                _say(f"tap self-check: fused dispatch-bwd + router grad max|diff|={d:.3e} max|ref|={sc:.3e} ({'OK' if close else 'FAIL'})")
                if not close:
                    _TAP_OK[0] = False
                    _say("tap self-check FAILED: router grad accumulate stays with autograd for this run")
                    out = ref_rows + d2
        except Exception as ex:  # noqa
            _fallback(f"tap bwd kernel raised {type(ex).__name__}: {ex}; HybridEP combine + add instead")
            out = None
        if out is None:
            from megatron.core.transformer.moe import fused_a2a
            if ref_rows is None:
                ref_rows, _ = fused_a2a._hybrid_ep_buffer.combine_with_unpermute(hidden=grad_rows, handle=handle, pad_multiple=pad_multiple)
            out = ref_rows + d2
        _STATS["tap_fused"] += 1
        if _STATS["tap_fused"] <= MAX_LOG:
            _say(f"tap bwd: dispatch-bwd unpermute + router grad in one pass [{T}, {H}]")
        return out.view(d_router.shape), None


class _DispatchTap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, routing_map, probs, group, num_local_experts, num_sms_dispatch_api, num_sms_combine_api,
                num_blocks_permute, num_blocks_unpermute, fused, num_permuted_tokens, pad_multiple,
                num_sms_preprocessing_api):
        from megatron.core.transformer.moe import fused_a2a
        outs = fused_a2a.HybridEPDispatch.forward(
            ctx, x, routing_map, probs, group, num_local_experts, num_sms_dispatch_api, num_sms_combine_api,
            num_blocks_permute, num_blocks_unpermute, fused, num_permuted_tokens, pad_multiple, num_sms_preprocessing_api)
        st = _NEXT_DISPATCH[0]
        _NEXT_DISPATCH[0] = None
        ctx.st = None
        if (st is not None and TAP and _TAP_OK[0] and not fused and torch.is_tensor(probs) and probs.dim() == 2
                and probs.dtype == torch.float32 and probs.shape[1] == num_local_experts and x.numel() == st.numel
                and x.dtype == torch.bfloat16):
            ctx.st = st
            ctx.probs_shape = probs.shape
        return outs

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        from megatron.core.transformer.moe import fused_a2a
        st = ctx.st
        if (st is None or st.tap_done or not _TAP_OK[0] or not torch.is_tensor(grad_x)
                or not (torch.is_tensor(grad_probs) or grad_probs is None)):
            _STATS["dispatch_plain"] += 1
            return fused_a2a.HybridEPDispatch.backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle)
        handle = ctx.handle
        grad_x = grad_x.contiguous()
        parts, why = _handle_parts(handle, grad_x, st.numel)
        if parts is None or (grad_probs is not None and (grad_probs.numel() != grad_x.shape[0]
                                                         or (grad_probs.dim() != 1 and grad_probs.shape[0] != grad_x.shape[0]))):
            _fallback(f"dispatch bwd: {why or 'grad_probs shape'}; HybridEP path")
            _STATS["dispatch_plain"] += 1
            st.tap_done = True       # the tap must not expect a stash
            return fused_a2a.HybridEPDispatch.backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle)
        m, layout, T = parts
        gp = grad_probs.contiguous().reshape(-1) if grad_probs is not None else None
        try:
            d_probs = torch.ops.gpt_oss.probs_grad_gather(gp, m, layout, T).view(ctx.probs_shape) if gp is not None else None
        except Exception as ex:  # noqa
            _fallback(f"probs grad gather raised {type(ex).__name__}: {ex}; HybridEP path")
            _STATS["dispatch_plain"] += 1
            st.tap_done = True
            return fused_a2a.HybridEPDispatch.backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle)
        ref_rows = None
        if SELFCHECK and not _TAP_CHECKED[0] and gp is not None:
            _TAP_CHECKED[0] = True
            try:
                ref_rows, ref_probs = fused_a2a._hybrid_ep_buffer.combine_with_unpermute(
                    hidden=grad_x, probs=grad_probs, handle=handle, pad_multiple=ctx.pad_multiple)
                same = torch.equal(d_probs.reshape(ref_probs.shape), ref_probs)
                _say(f"dispatch-bwd self-check: probs grad bitwise HybridEP = {same} ({tuple(ref_probs.shape)})")
                if not same:
                    _TAP_OK[0] = False
                    st.tap_done = True
                    _say("dispatch-bwd self-check FAILED: router grad accumulate stays with autograd for this run")
                    return (ref_rows, None, ref_probs, None, None, None, None, None, None, None, None, None, None)
            except Exception as ex:  # noqa
                _TAP_OK[0] = False
                st.tap_done = True
                _say(f"dispatch-bwd self-check raised {type(ex).__name__}: {ex}; HybridEP path for this run")
                return fused_a2a.HybridEPDispatch.backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle)
        st.stash = (grad_x, handle, ctx.pad_multiple, ref_rows)
        _STATS["dispatch_stashed"] += 1
        if _STATS["dispatch_stashed"] <= MAX_LOG:
            _say(f"dispatch bwd: probs grad only; {tuple(grad_x.shape)} row grads parked for the router-grad tap")
        return (None, None, d_probs, None, None, None, None, None, None, None, None, None, None)


def _hybrid_ep_dispatch(x, routing_map, probs, group, num_local_experts, num_sms_dispatch_api=None,
                        num_sms_combine_api=None, num_blocks_permute=None, num_blocks_unpermute=None, fused=False,
                        num_permuted_tokens=None, pad_multiple=None, num_sms_preprocessing_api=108):
    if _NEXT_DISPATCH[0] is None or not TAP or not _TAP_OK[0]:
        return _ORIG["hybrid_ep_dispatch"](x, routing_map, probs, group, num_local_experts, num_sms_dispatch_api,
                                           num_sms_combine_api, num_blocks_permute, num_blocks_unpermute, fused,
                                           num_permuted_tokens, pad_multiple, num_sms_preprocessing_api)
    return _DispatchTap.apply(x, routing_map, probs, group, num_local_experts, num_sms_dispatch_api, num_sms_combine_api,
                              num_blocks_permute, num_blocks_unpermute, fused, num_permuted_tokens, pad_multiple,
                              num_sms_preprocessing_api)


def _wrap_manager_dispatch(orig):
    def dispatch(self, hidden_states, *args, **kwargs):
        st = _TAP_SLOT.pop(id(self), None)
        if st is not None and getattr(self, "_padded_num_tokens", None) not in (None, getattr(self, "_original_num_tokens", None)):
            st = None
        _NEXT_DISPATCH[0] = st
        try:
            return orig(self, hidden_states, *args, **kwargs)
        finally:
            _NEXT_DISPATCH[0] = None
    return dispatch


def _wrap_route(orig):
    def route(self, hidden_states, *args, **kwargs):
        if TAP and _TAP_OK[0] and torch.is_tensor(hidden_states) and hidden_states.requires_grad and torch.is_grad_enabled() \
                and hidden_states.dtype == torch.bfloat16 and getattr(self, "training", False):
            try:
                mgr = _hybridep_manager(self)
                cfg = getattr(self, "config", None)
                if mgr is not None and not getattr(cfg, "overlap_dispatch_backward_with_experts_wgrad", False):
                    st = _TapState(hidden_states.numel())
                    if len(_TAP_SLOT) >= 64:
                        _TAP_SLOT.clear()
                    _TAP_SLOT[id(mgr)] = st
                    hidden_states = _Tap.apply(hidden_states, st)
            except Exception as ex:  # noqa
                _fallback(f"route tap raised {type(ex).__name__}: {ex}")
        return orig(self, hidden_states, *args, **kwargs)
    return route


# ------------------------------------------------------------------------------------------------------------
def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if not ENABLED:
        _say("disabled by RESIDUAL_BIAS_FUSION=0 (control path: proj bias returned and added by Megatron, residual adds in the "
             "bias-dropout-adds, autograd add at the MoE fan-out)")
        return False
    parts = []
    try:
        import importlib
        try:
            importlib.import_module("custom_kernels.unpermute_add")
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            sys.modules["custom_kernels.unpermute_add"] = importlib.import_module("unpermute_add")
    except Exception as e:  # noqa
        _say(f"kernel import failed, hooks NOT installed: {type(e).__name__}: {e}")
        return False

    if BIAS:
        try:
            from megatron.core.extensions.transformer_engine import TERowParallelLinear
            _install_proj_bias(TERowParallelLinear)
            parts.append("TERowParallelLinear.__init__(proj bias -> epilogue)")
        except Exception as ex:  # noqa
            _fallback(f"TERowParallelLinear patch failed: {type(ex).__name__}: {ex}")
        try:
            from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
            prev = TransformerEngineBaseModule.grad_output_preprocess
            _ORIG["grad_output_preprocess"] = prev
            TransformerEngineBaseModule.grad_output_preprocess = staticmethod(_make_grad_output_preprocess(prev))
            parts.append("TransformerEngineBaseModule.grad_output_preprocess(+dbias)")
        except Exception as ex:  # noqa
            _fallback(f"grad_output_preprocess patch failed: {type(ex).__name__}: {ex}")
    if BIAS or COMBINE:
        try:
            from megatron.core.fusions import fused_bias_dropout as fbd
            _ORIG["_bias_dropout_add_func"] = fbd._bias_dropout_add_func
            fbd._bias_dropout_add_func = _bias_dropout_add_func
            parts.append("fused_bias_dropout._bias_dropout_add_func")
        except Exception as ex:  # noqa
            _fallback(f"bias-dropout-add patch failed: {type(ex).__name__}: {ex}")
    if COMBINE or TAP:
        try:
            from megatron.core.transformer.moe import token_dispatcher as TD
            if TD.hybrid_ep_combine is None or TD.hybrid_ep_dispatch is None:
                raise RuntimeError("HybridEP is not available in this container")
            if COMBINE:
                _ORIG["hybrid_ep_combine"] = TD.hybrid_ep_combine
                TD.hybrid_ep_combine = _hybrid_ep_combine
                _ORIG["_HybridEPManager.combine"] = TD._HybridEPManager.combine
                TD._HybridEPManager.combine = _wrap_manager_combine(TD._HybridEPManager.combine)
                from megatron.core.transformer.transformer_layer import TransformerLayer
                name = "_pre_mlp_layernorm_and_residual"
                if not hasattr(TransformerLayer, name):
                    raise RuntimeError(f"TransformerLayer has no {name}")
                _ORIG[name] = getattr(TransformerLayer, name)
                setattr(TransformerLayer, name, _wrap_pre_mlp(_ORIG[name]))
                parts.append("token_dispatcher.hybrid_ep_combine(+residual), _HybridEPManager.combine, TransformerLayer._pre_mlp_layernorm_and_residual(record)")
            if TAP:
                _ORIG["hybrid_ep_dispatch"] = TD.hybrid_ep_dispatch
                TD.hybrid_ep_dispatch = _hybrid_ep_dispatch
                _ORIG["_HybridEPManager.dispatch"] = TD._HybridEPManager.dispatch
                TD._HybridEPManager.dispatch = _wrap_manager_dispatch(TD._HybridEPManager.dispatch)
                from megatron.core.transformer.moe.moe_layer import MoELayer
                _ORIG["MoELayer.route"] = MoELayer.route
                MoELayer.route = _wrap_route(MoELayer.route)
                parts.append("token_dispatcher.hybrid_ep_dispatch(tap), _HybridEPManager.dispatch, MoELayer.route(tap)")
        except Exception as ex:  # noqa
            _fallback(f"MoE patches failed: {type(ex).__name__}: {ex}")

    _INSTALLED = True
    _say(f"installed: {'; '.join(parts) if parts else 'nothing'}; BIAS={BIAS} COMBINE={COMBINE} TAP={TAP} SELFCHECK={SELFCHECK} "
         f"DGLU(probability in the combine)={DGLU}")
    return True


def stats():
    return dict(_STATS, combine_ok=_COMBINE_OK[0], tap_ok=_TAP_OK[0], bda_reached=_BDA_REACHED[0], dglu_bwd_ok=_DGLU_BWD_OK[0],
                res_slots=len(_MLP_RES_SLOT), fused_combine_pending=len(_FUSED_COMBINE), tap_slots=len(_TAP_SLOT))
