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

"""Fused RMSNorm backward for the gpt-oss-20b transformer layer, installed by rebinding TE/Megatron entry points:
  1. transformer_engine_torch.rmsnorm_bwd (+ copies in te.pytorch.cpp_extensions / ops.basic.rmsnorm) -> one fused pass
     for dx and dgamma instead of cuDNN's ln_tma_bwd + ln_bwd_finalize; norm1 folds in the residual grad from (3) (FUSE1).
  2. Megatron's TransformerLayer._pre_mlp_layernorm_and_residual -> one autograd Function for norm2: forward = TE's own
     rmsnorm_fwd (bitwise-identical), backward = fused pass + residual grad + MXFP8 row/col proj grad + proj dbias (FUSE2).
  3. Megatron's _bias_dropout_add_func (attention BDA): same forward; the backward hands the proj Linear the MXFP8 grad
     and dbias from (2) and stashes the residual gradient for (1) instead of returning it.
  4. TE's grad_output_preprocess: accepts the MXFP8 grad from (2) as-is (no re-cast), else uses its bf16 twin.
  5. TE's rmsnorm_fwd: records norm1's input pointers so (3) can tell the layer input from other residuals.
Backward sites call the single-sweep CUDA kernel of rmsnorm_bwd_cuda.py (NORM_BWD_CUDA=1, default) when it is
loaded and eligible, else the Triton pair of rmsnorm_bwd_fused.py; NORM_BWD_MAIN_GRAD=1 (default) lets the CUDA
kernel write dgamma/dbias straight into the parameters' main_grad.  NORM_BWD=0 leaves the reference path; NORM_BWD_FUSE2=0
turns off the MXFP8/dbias path of (2)+(3), NORM_BWD_FUSE1=0 the residual stash of (3); NORM_BWD_BLOCK_N / _NUM_WARPS pick
the Triton tile.  Other dtypes/strides, recompute, offload, fp32 residual or a non-MXFP8 recipe use the original code.
"""
import inspect
import os
import re
import sys

import torch

_INSTALLED = False
_ORIG = {}
_STATS = {"bwd_plain": 0, "bwd_res1": 0, "norm2_fwd": 0, "norm2_bwd": 0, "bda_fast": 0, "bda_plain": 0,
          "gop_fast": 0, "gop_bf16": 0, "fallback": 0, "norm2_checked_ok": 0, "norm2_checked_bad": 0,
          "fwd_rec": 0, "side_unconsumed": 0,
          "cuda_calls": 0, "triton_calls": 0, "dgamma_main_grad": 0, "dgamma_tensor": 0, "dbias_main_grad": 0,
          "dbias_tensor": 0}
_PENDING_RES1 = {}     # layer-input data_ptr -> d_hidden (bf16) awaiting norm1's fused backward (FUSE1)
_NORM1_INPUTS = {}     # data_ptr -> numel: grad-requiring inputs seen by rmsnorm_fwd (LayerNormLinear forward) this step
_GRAD_SIDE = {}        # d_hidden data_ptr -> (MXFP8Tensor grad for proj, dbias [N]) produced by norm2's backward
_GRAD_QUANTIZERS = {}  # TE fp8 dtype -> MXFP8Quantizer carried by the MXFP8Tensor grad (one per dtype)

ENABLED = os.getenv("NORM_BWD", "1") == "1"
FUSE1 = os.getenv("NORM_BWD_FUSE1", "1") == "1"
FUSE2 = os.getenv("NORM_BWD_FUSE2", "1") == "1"
# the single-sweep CUDA C++ kernel (rmsnorm_bwd_cuda.py) and the direct main_grad write
CUDA_WANTED = os.getenv("NORM_BWD_CUDA", "1") == "1"
MAIN_GRAD = os.getenv("NORM_BWD_MAIN_GRAD", "1") == "1"
_CUDA = {"tried": False, "ok": False, "mod": None}
# Default Triton tile; keep in step with DEFAULT_BLOCK_N / DEFAULT_NUM_WARPS in rmsnorm_bwd_fused.py (imported lazily).
BLOCK_N = int(os.getenv("NORM_BWD_BLOCK_N", "256"))
NUM_WARPS = int(os.getenv("NORM_BWD_NUM_WARPS", "4"))
MAX_LOG = 3


def _say(msg: str) -> None:
    # utils.py replaces builtins.print with a rank-0-only print; write through sys.__stdout__ instead
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"NORM_BWD rank={rank} {msg}\n")
    sys.__stdout__.flush()


def _fallback(what: str) -> None:
    _STATS["fallback"] += 1
    if _STATS["fallback"] <= 8:
        _say(f"fallback: {what}")


def _eligible_2d(dz, x, rsigma, gamma):
    if dz.dtype != torch.bfloat16 or x.dtype != torch.bfloat16 or gamma.dtype != torch.bfloat16:
        return None
    if rsigma.dtype != torch.float32 or x.dim() < 2 or dz.dim() < 2 or x.numel() != dz.numel():
        return None
    N = x.shape[-1]
    if gamma.numel() != N or N % 32 != 0 or dz.shape[-1] != N:
        return None
    dz2 = dz.reshape(-1, N)         # TE hands LayerNormLinear's 3-D inputmat with a 2-D GEMM dgrad
    x2 = x.reshape(-1, N)
    M = x2.shape[0]
    if M % 32 != 0 or rsigma.numel() != M:
        return None
    if not (dz2.is_contiguous() and x2.is_contiguous() and rsigma.is_contiguous() and gamma.is_contiguous()):
        return None
    return dz2, x2, M, N


# ------------------------------------------------------------------------------------------------------------
# 0. the single-sweep CUDA C++ backend and the direct main_grad targets
# ------------------------------------------------------------------------------------------------------------
def _cuda():
    """The CUDA backend module, built/loaded on first use (eagerly: the warmup steps precede the graph capture); None when
    it is unavailable or NORM_BWD_CUDA=0, in which case the Triton pair is used."""
    if not _CUDA["tried"]:
        _CUDA["tried"] = True
        if CUDA_WANTED:
            try:
                import custom_kernels.rmsnorm_bwd_cuda as C
                _CUDA["ok"] = bool(C.ensure_loaded())
                _CUDA["mod"] = C if _CUDA["ok"] else None
            except Exception as ex:  # noqa
                _say(f"CUDA backend import failed, Triton pair stays: {type(ex).__name__}: {str(ex)[:400]}")
                _CUDA["ok"] = False
        else:
            _say("CUDA backend disabled by NORM_BWD_CUDA=0 (control path: Triton rmsnorm_bwd_fused + colsum)")
    return _CUDA["mod"]


def _main_grad_target(t, N):
    """(param, main_grad [N] bf16 view, accumulate) when t is (a view of) a DDP parameter whose gradient producer may write
    main_grad directly under TE's grad_added_to_main_grad protocol; None otherwise (ordinary autograd gradient)."""
    if not MAIN_GRAD or t is None:
        return None
    p = t if isinstance(t, torch.nn.Parameter) else getattr(t, "_base", None)
    if p is None or not isinstance(p, torch.Tensor) or not p.requires_grad or p.numel() != N:
        return None
    mg = getattr(p, "main_grad", None)
    if mg is None or not hasattr(p, "grad_added_to_main_grad") or getattr(p, "zero_out_wgrad", False):
        return None
    if getattr(p, "grad_added_to_main_grad", False):
        return None                                   # already written this step: never double-count
    if mg.dtype != torch.bfloat16 or mg.numel() != N or not mg.is_contiguous():
        return None
    return p, mg.view(-1), not getattr(p, "overwrite_main_grad", False)


def _cuda_bwd(C, dz2, x2, rsigma, gamma, dres, zcg, want_mxfp8, fp8_code, gamma_param_src, dbias_param, what):
    """One single-sweep launch.  Returns (out, dgamma_for_autograd, dbias_for_autograd, rd, rs, cd, cs): the two gradient
    tensors are fresh dummies when the kernel wrote the parameters' main_grad slots, else the freshly written values."""
    N = gamma.numel()
    g1 = gamma.reshape(N)
    tgt = _main_grad_target(gamma_param_src, N)
    if tgt is not None:
        dg_out, acc = tgt[1], tgt[2]
    else:
        dg_out, acc = torch.empty(N, dtype=torch.bfloat16, device=gamma.device), False
    db_out = None
    btgt = None
    if want_mxfp8:
        btgt = _main_grad_target(dbias_param, N)
        if btgt is not None and btgt[2] != acc:
            btgt = None                              # one accumulate flag per launch: keep the ordinary path for the bias
        db_out = btgt[1] if btgt is not None else torch.empty(N, dtype=torch.bfloat16, device=gamma.device)
    out, rd, rs, cd, cs = C.rmsnorm_bwd(dz2, x2, rsigma, g1, dres, zcg, want_mxfp8, fp8_code, dg_out, db_out, acc)
    if tgt is not None:
        tgt[0].grad_added_to_main_grad = True
        dgamma = torch.empty(N, dtype=torch.bfloat16, device=gamma.device)   # fresh, uniquely owned: stolen, never cloned
        _STATS["dgamma_main_grad"] += 1
    else:
        dgamma = dg_out
        _STATS["dgamma_tensor"] += 1
    dbias = None
    if want_mxfp8:
        if btgt is not None:
            btgt[0].grad_added_to_main_grad = True
            dbias = torch.empty(N, dtype=torch.bfloat16, device=gamma.device)
            _STATS["dbias_main_grad"] += 1
        else:
            dbias = db_out
            _STATS["dbias_tensor"] += 1
    _STATS["cuda_calls"] += 1
    if _STATS["cuda_calls"] <= 3 * MAX_LOG:
        _say(f"{what}: single-sweep CUDA kernel [{dz2.shape[0]}, {N}] res={dres is not None} mxfp8={want_mxfp8} "
             f"dgamma->{'main_grad' if tgt is not None else 'tensor'}"
             + (f" dbias->{'main_grad' if btgt is not None else 'tensor'}" if want_mxfp8 else "")
             + f" accumulate={acc}")
    return out, dgamma, dbias, rd, rs, cd, cs


# ------------------------------------------------------------------------------------------------------------
# 1. transformer_engine_torch.rmsnorm_bwd
# ------------------------------------------------------------------------------------------------------------
def _rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma):
    import custom_kernels.rmsnorm_bwd_fused as K  # noqa (imported lazily: the kernel module is heavy)

    e = _eligible_2d(dz, x, rsigma, gamma)
    if e is None:
        _fallback(f"rmsnorm_bwd shapes/dtypes ({tuple(dz.shape)}, {dz.dtype}, {tuple(x.shape)}, {rsigma.dtype})")
        return _ORIG["rmsnorm_bwd"](dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
    dz2, x2, M, N = e
    dres = _PENDING_RES1.pop(x2.data_ptr(), None) if FUSE1 else None
    dres_add = None
    if dres is not None:
        dres = dres.reshape(-1, N)
        if dres.shape != dz2.shape or dres.dtype != torch.bfloat16 or not dres.is_contiguous():
            # never silently drop a residual gradient: add it the plain way
            _fallback("stashed residual grad incompatible; adding separately")
            dres_add, dres = dres, None
    C = _cuda()
    if C is not None and C.eligible(M, N):
        out, dgamma, _, _, _, _, _ = _cuda_bwd(C, dz2, x2, rsigma, gamma, dres, bool(zero_centered_gamma), False, 7,
                                               gamma, None, "rmsnorm_bwd (norm1 / final norm)")
        _STATS["bwd_res1" if dres is not None else "bwd_plain"] += 1
    elif dres is not None:
        out, dgamma = K.rmsnorm_bwd_plain(dz2, x2, rsigma, gamma, dres, bool(zero_centered_gamma), BLOCK_N, NUM_WARPS)
        _STATS["triton_calls"] += 1
        _STATS["bwd_res1"] += 1
        if _STATS["bwd_res1"] <= MAX_LOG:
            _say(f"rmsnorm_bwd fused with the stashed residual grad (Triton pair): [{M}, {N}]")
    else:
        out, dgamma = K.rmsnorm_bwd_plain(dz2, x2, rsigma, gamma, None, bool(zero_centered_gamma), BLOCK_N, NUM_WARPS)
        _STATS["triton_calls"] += 1
        _STATS["bwd_plain"] += 1
        if _STATS["bwd_plain"] <= MAX_LOG:
            _say(f"rmsnorm_bwd fused (Triton pair: dx + dgamma partials + reduce): [{M}, {N}]")
    if dres_add is not None:
        out = out + dres_add
    return out.view(x.shape), dgamma.view(gamma.shape)      # TE's rmsnorm_bwd returns dx shaped like x


# ------------------------------------------------------------------------------------------------------------
# 5. transformer_engine_torch.rmsnorm_fwd recorder (LayerNormLinear forward -> norm1 inputs)
# ------------------------------------------------------------------------------------------------------------
def _rmsnorm_fwd_recording(*args, **kwargs):
    # rmsnorm_fwd(inputmat, ln_weight, eps, ln_out, quantizer, out_dtype, sm_margin, zero_centered_gamma); runs inside
    # _LayerNormLinear.forward (grad mode off there), so the tensor's own requires_grad flag tells training from eval.
    if FUSE1 and args and isinstance(args[0], torch.Tensor) and args[0].requires_grad:
        if len(_NORM1_INPUTS) >= 64:
            _NORM1_INPUTS.clear()
        _NORM1_INPUTS[args[0].data_ptr()] = args[0].numel()
        _STATS["fwd_rec"] += 1
    return _ORIG["rmsnorm_fwd"](*args, **kwargs)


# ------------------------------------------------------------------------------------------------------------
# 2. norm2: pre_mlp_layernorm forward + fused backward with residual / MXFP8 / dbias
# ------------------------------------------------------------------------------------------------------------
def _grad_quantizer(fp8_dtype):
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    q = _GRAD_QUANTIZERS.get(fp8_dtype)
    if q is None:
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=True)
        _GRAD_QUANTIZERS[fp8_dtype] = q
    return q


def _mxfp8_grad_wanted():
    """(True, fp8_code) when the layer's linears run the MXFP8 recipe in this forward (TE DType code of the grad:
    8 = kFloat8E5M2 under Format.HYBRID, 7 = kFloat8E4M3 otherwise)."""
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
        if not FP8GlobalStateManager.is_fp8_enabled():
            return False, 7
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe is None or not recipe.mxfp8():
            return False, 7
        from transformer_engine.common.recipe import Format
        if recipe.fp8_format == Format.HYBRID:
            return True, 8
        return True, 7
    except Exception as ex:  # noqa
        _fallback(f"recipe query failed: {type(ex).__name__}: {ex}")
        return False, 7


class _FusedNorm2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, eps, zero_centered_gamma, sm_fwd, want_mxfp8, fp8_code, st=None):
        from transformer_engine.pytorch.constants import TE_DType
        shape = hidden.shape
        N = shape[-1]
        x2 = hidden.contiguous().view(-1, N)
        y, _, rstd = _ORIG["rmsnorm_fwd"](x2, weight.view(N), eps, None, None, TE_DType[hidden.dtype], sm_fwd,
                                          zero_centered_gamma)
        ctx.save_for_backward(x2, rstd, weight)
        ctx.set_materialize_grads(False)          # an unused residual output arrives as None, not as zeros
        ctx.meta = (shape, eps, bool(zero_centered_gamma), bool(want_mxfp8), int(fp8_code))
        ctx.st = st                               # the layer's cached state (holds the proj bias Parameter for its main_grad)
        _STATS["norm2_fwd"] += 1
        if _STATS["norm2_fwd"] <= MAX_LOG:
            _say(f"norm2 fwd: {tuple(shape)} -> (y, residual) want_mxfp8={want_mxfp8}")
        # The residual output is a fresh view of the input, never the input object itself: autograd then treats it as a
        # view-of-input output of this Function, so its gradient arrives in backward() as `dres` and the kernel folds it
        # into d_hidden instead of the engine summing the two.  _install_megatron_norm2 checks it shares hidden_states' storage.
        return y.view(shape), hidden.view_as(hidden)

    @staticmethod
    def backward(ctx, dy, dres):
        import custom_kernels.rmsnorm_bwd_fused as K
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        x2, rstd, weight = ctx.saved_tensors
        shape, eps, zcg, want_mxfp8, fp8_code = ctx.meta
        N = shape[-1]
        if dy is None:
            dy = torch.zeros(shape, dtype=x2.dtype, device=x2.device)
        dy2 = dy.contiguous().view(-1, N)
        dres2 = dres.contiguous().view(-1, N) if dres is not None else None
        if dres2 is not None and (dres2.dtype != torch.bfloat16 or dres2.shape != dy2.shape):
            _fallback("norm2 residual grad dtype/shape; adding separately")
            dres_add, dres2 = dres2, None
        else:
            dres_add = None
        if _GRAD_SIDE:
            _STATS["side_unconsumed"] += 1
            if _STATS["side_unconsumed"] <= MAX_LOG:
                _say(f"warning: {len(_GRAD_SIDE)} MXFP8 side grads were never consumed (proj took the bf16 path)")
            _GRAD_SIDE.clear()
        C = _cuda()
        M = x2.shape[0]
        if C is not None and C.eligible(M, N):
            st = getattr(ctx, "st", None) or {}
            out, dgamma, dbias, rd, rs, cd, cs = _cuda_bwd(C, dy2, x2, rstd, weight, dres2, zcg, want_mxfp8, fp8_code,
                                                           weight, st.get("proj_bias"), "norm2 bwd")
            dgamma = dgamma.view(weight.shape)
            how = "single-sweep CUDA kernel"
        else:
            out, part, rd, rs, cd, cs = torch.ops.gpt_oss.rmsnorm_bwd_fused(
                dy2, x2, rstd, weight.view(N), dres2, zcg, want_mxfp8, want_mxfp8, fp8_code, BLOCK_N, NUM_WARPS)
            red = torch.ops.gpt_oss.colsum(part, weight.dtype)
            dgamma = red[:N].view(weight.shape)
            dbias = red[N:] if want_mxfp8 else None
            _STATS["triton_calls"] += 1
            how = "Triton pair"
        if dres_add is not None:
            out = out + dres_add
        d_hidden = out.view(shape)
        if want_mxfp8:
            fp8_dtype = tex.DType.kFloat8E4M3 if fp8_code == 7 else tex.DType.kFloat8E5M2
            grad = MXFP8Tensor(
                shape=shape, dtype=dy.dtype,
                rowwise_data=rd.view(shape), rowwise_scale_inv=rs,
                columnwise_data=cd.view(shape), columnwise_scale_inv=cs,
                fp8_dtype=fp8_dtype, quantizer=_grad_quantizer(fp8_dtype), with_gemm_swizzled_scales=True,
                requires_grad=False, device=dy.device,
            )
            grad._fused_mxfp8 = True
            grad._fused_bf16 = d_hidden
            _GRAD_SIDE[d_hidden.data_ptr()] = (grad, dbias)
        _STATS["norm2_bwd"] += 1
        if _STATS["norm2_bwd"] <= MAX_LOG:
            _say(f"norm2 bwd fused ({how}): dx + residual grad{' + MXFP8 row/col + dbias' if want_mxfp8 else ''} "
                 f"+ dgamma, [{M}, {N}]")
        return d_hidden, dgamma, None, None, None, None, None, None


def _norm2_state(layer):
    """Per-layer cached description of pre_mlp_layernorm when it holds exactly one te.ops RMSNorm (else None).
    The structure check is deliberately loose: the first-call self-check in _install_megatron_norm2 compares the
    original module's output bitwise with the fused forward, which rejects any wrapper that does more than RMSNorm."""
    st = getattr(layer, "_fused_norm2", "unset")
    if st != "unset":
        return st
    st = None
    try:
        from transformer_engine.pytorch.ops.basic import RMSNorm as OpRMSNorm
        mod = layer.pre_mlp_layernorm
        roots = [mod]
        impl = getattr(mod, "_get_fused_impl", None)
        if callable(impl):
            try:
                roots.append(impl())
            except Exception as ex:  # noqa
                _fallback(f"_get_fused_impl() raised {type(ex).__name__}: {ex}")
        ops = []
        for r in roots:
            if isinstance(r, torch.nn.Module):
                for m in r.modules():
                    if isinstance(m, OpRMSNorm) and all(m is not o for o in ops):
                        ops.append(m)
        if not ops and all(hasattr(mod, a) for a in ("weight", "eps", "zero_centered_gamma")):
            ops = [mod]                     # a plain te.pytorch.RMSNorm module carries the same attributes
        if len(ops) != 1:
            _fallback(f"layer {getattr(layer, 'layer_number', '?')}: pre_mlp_layernorm {type(mod).__name__} holds {len(ops)} RMSNorm ops")
        elif getattr(layer, "recompute_pre_mlp_layernorm", False) or getattr(layer, "offload_mlp_norm", False):
            _fallback("pre_mlp_layernorm recompute/offload is on")
        elif getattr(getattr(layer, "config", None), "fp32_residual_connection", False):
            _fallback("fp32_residual_connection")
        else:
            op = ops[0]
            margins = getattr(op, "_sm_margins", None)
            sm_fwd = int(margins.get("forward", 0)) if isinstance(margins, dict) else int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
            # the proj bias Parameter: its gradient is the column sum of norm2's residual-stream gradient, which the
            # fused backward computes; with a resolvable main_grad the kernel writes it there
            pb = None
            try:
                pb = layer.self_attention.linear_proj.bias
            except AttributeError:
                pb = None
            if not (isinstance(pb, torch.nn.Parameter) and pb.requires_grad and pb.numel() == op.weight.numel()
                    and pb.dtype == torch.bfloat16):
                pb = None
            st = {"op": op, "checked": False, "sm_fwd": sm_fwd, "proj_bias": pb}
    except Exception as ex:  # noqa
        _fallback(f"norm2 introspection failed: {type(ex).__name__}: {ex}")
        st = None
    layer._fused_norm2 = st
    return st


def _norm2_apply(layer, hidden_states):
    """(y, residual) through the fused Function; None when this call must take the original path."""
    st = _norm2_state(layer)
    if st is None:
        return None
    op = st["op"]
    w = op.weight
    if hidden_states.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or hidden_states.shape[-1] != w.numel():
        _fallback(f"norm2 dtypes/shape {hidden_states.dtype} {tuple(hidden_states.shape)} {w.dtype}")
        return None
    if not (torch.is_grad_enabled() and hidden_states.requires_grad):
        return None
    if (hidden_states.numel() // hidden_states.shape[-1]) % 32 != 0 or w.numel() % 32 != 0:
        _fallback(f"norm2 shape {tuple(hidden_states.shape)} not 32-aligned")
        return None
    want_mxfp8, fp8_code = _mxfp8_grad_wanted() if FUSE2 else (False, 7)
    return _FusedNorm2.apply(hidden_states, w, float(op.eps), bool(op.zero_centered_gamma), st["sm_fwd"],
                               want_mxfp8, fp8_code, st)


def _install_megatron_norm2(TransformerLayer):
    """Rebind TransformerLayer._pre_mlp_layernorm_and_residual; the first call per layer cross-checks the forward
    against the original module (same TE kernel -> bitwise equal) before the fast path is used."""
    name = "_pre_mlp_layernorm_and_residual"
    if not hasattr(TransformerLayer, name):
        _fallback(f"TransformerLayer has no {name}; norm2 fusion off")
        return False
    orig = getattr(TransformerLayer, name)
    try:
        src = inspect.getsource(orig)
    except (OSError, TypeError) as ex:
        _fallback(f"cannot read {name} source ({ex}); norm2 fusion off")
        return False
    m = re.search(r"\n\s*return\s+pre_mlp_layernorm_output\s*,\s*residual\s*(,\s*(\w+))?\s*\n", src)
    if m is None:
        _fallback(f"{name} return statement not recognised; norm2 fusion off")
        return False
    arity = 3 if m.group(1) else 2
    _ORIG[name] = orig

    def _pre_mlp_layernorm_and_residual(self, hidden_states, *args, **kwargs):
        if args or any(v is not None for v in kwargs.values()):
            return orig(self, hidden_states, *args, **kwargs)
        st = _norm2_state(self)
        if st is None:
            return orig(self, hidden_states, *args, **kwargs)
        if not st["checked"]:
            # first (eager warmup) call: run the original and compare its y with the fused forward bitwise
            res = orig(self, hidden_states, *args, **kwargs)
            try:
                with torch.enable_grad():
                    mine = _norm2_apply(self, hidden_states.detach().requires_grad_(True))
                ok = (mine is not None and len(res) == arity and torch.equal(res[0].detach(), mine[0].detach())
                      and res[1].data_ptr() == hidden_states.data_ptr() and res[1].shape == hidden_states.shape)
                del mine
            except Exception as ex:  # noqa
                ok = False
                _fallback(f"norm2 self-check raised {type(ex).__name__}: {ex}")
            st["checked"] = True
            if ok:
                _STATS["norm2_checked_ok"] += 1
                if _STATS["norm2_checked_ok"] <= MAX_LOG:
                    _say(f"norm2 self-check passed on layer {getattr(self, 'layer_number', '?')}: fused path active from the next call")
            else:
                _STATS["norm2_checked_bad"] += 1
                _say(f"norm2 self-check FAILED on layer {getattr(self, 'layer_number', '?')}: original path kept")
                self._fused_norm2 = None
            return res
        out = _norm2_apply(self, hidden_states)
        if out is None:
            return orig(self, hidden_states, *args, **kwargs)
        y, residual = out
        return (y, residual) if arity == 2 else (y, residual, ())

    setattr(TransformerLayer, name, _pre_mlp_layernorm_and_residual)
    return True


# ------------------------------------------------------------------------------------------------------------
# 3. attention bias-dropout-add
# ------------------------------------------------------------------------------------------------------------
class _FusedAttnBDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, residual, res_key):
        out = x + bias                 # Megatron's two kernels, unchanged (dropout p = 0 is the identity)
        out = residual + out
        ctx.res_key = int(res_key)
        ctx.bias_shape = bias.shape
        ctx.ndim = x.dim()
        return out

    @staticmethod
    def backward(ctx, d_out):
        side = _GRAD_SIDE.pop(d_out.data_ptr(), None)
        if side is not None:
            d_x, dbias = side          # the MXFP8 grad is counted in grad_output_preprocess when the proj accepts it
            d_bias = dbias.view(ctx.bias_shape)
        else:
            d_x = d_out
            d_bias = d_out.sum(dim=tuple(range(ctx.ndim - 1))).view(ctx.bias_shape)
        if ctx.res_key:
            if _PENDING_RES1:
                raise RuntimeError("NORM_BWD: a stashed residual gradient was never consumed by norm1's "
                                   f"fused backward (keys {list(_PENDING_RES1)}); refusing to continue")
            _PENDING_RES1[ctx.res_key] = d_out
            d_res = None
        else:
            d_res = d_out
        _STATS["bda_fast"] += 1
        if _STATS["bda_fast"] <= MAX_LOG:
            _say(f"attention BDA bwd: proj grad {'MXFP8 (precomputed) + dbias partials' if side is not None else 'bf16 + dbias sum'}"
                 f", residual grad {'stashed for the fused norm1 pass' if ctx.res_key else 'returned'}")
        return d_x, d_bias, d_res, None


def _bias_dropout_add_func(x_with_bias, residual, prob, training, *rest):
    x, bias = x_with_bias
    fast = (training and not rest and bias is not None and prob == 0.0 and torch.is_grad_enabled()
            and x.dtype == residual.dtype == torch.bfloat16 and x.shape == residual.shape
            and bias.dim() == 1 and bias.shape[0] == x.shape[-1]
            and (x.requires_grad or residual.requires_grad or bias.requires_grad))
    if not fast:
        _STATS["bda_plain"] += 1
        return _ORIG["_bias_dropout_add_func"](x_with_bias, residual, prob, training, *rest)
    res_key = 0
    if FUSE1 and residual.requires_grad and _NORM1_INPUTS.pop(residual.data_ptr(), None) == residual.numel():
        res_key = residual.data_ptr()
    if _PENDING_RES1:
        raise RuntimeError("NORM_BWD: a stashed residual gradient from the previous backward was never "
                           f"consumed (keys {list(_PENDING_RES1)}); refusing to continue")
    return _FusedAttnBDA.apply(x, bias, residual, res_key)


# ------------------------------------------------------------------------------------------------------------
# 4. grad_output_preprocess: accept the MXFP8 grad
# ------------------------------------------------------------------------------------------------------------
def _make_grad_output_preprocess(prev):
    def grad_output_preprocess(ctx, grad_output, row_parallel_mode, quantizer):
        if getattr(grad_output, "_fused_mxfp8", False) and hasattr(grad_output, "_fused_bf16"):
            gather = row_parallel_mode and ctx.sequence_parallel
            use_fp8_bwd = ctx.fp8 and ctx.backward_override is None
            recipe_ok = False
            try:
                from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
                recipe_ok = isinstance(quantizer, MXFP8Quantizer) and quantizer.dtype == grad_output._fp8_dtype
            except Exception:  # noqa
                recipe_ok = False
            if (use_fp8_bwd and not gather and not getattr(ctx, "debug", False)
                    and not getattr(ctx, "ub_overlap_ag", False) and recipe_ok and not ctx.use_bias):
                _STATS["gop_fast"] += 1
                if _STATS["gop_fast"] <= MAX_LOG:
                    _say(f"grad_output_preprocess fast path: MXFP8 grad {tuple(grad_output.shape)} passed to the proj GEMMs")
                return grad_output, None
            _STATS["gop_bf16"] += 1
            if _STATS["gop_bf16"] <= MAX_LOG:
                _say(f"grad_output_preprocess: outside the fast path (fp8={ctx.fp8} gather={gather} recipe_ok={recipe_ok} "
                     f"use_bias={ctx.use_bias}); using the bf16 twin")
            grad_output = grad_output._fused_bf16
        return prev(ctx, grad_output, row_parallel_mode, quantizer)
    return grad_output_preprocess


# ------------------------------------------------------------------------------------------------------------
def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if not ENABLED:
        _say("disabled by NORM_BWD=0 (control path: TE/cuDNN rmsnorm_bwd, autograd residual adds, proj re-cast)")
        return False
    try:
        import importlib
        try:
            importlib.import_module("custom_kernels.rmsnorm_bwd_fused")
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            sys.modules["custom_kernels.rmsnorm_bwd_fused"] = importlib.import_module("rmsnorm_bwd_fused")
    except Exception as e:  # noqa
        _say(f"kernel import failed, hooks NOT installed: {type(e).__name__}: {e}")
        return False

    import transformer_engine.pytorch  # noqa: F401  (loads the transformer_engine_torch extension)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import cpp_extensions as te_cpp   # `from transformer_engine_torch import *` copy
    from transformer_engine.pytorch.ops.basic import rmsnorm as te_ops_rmsnorm
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    # 1 + 5: TE norm entry points.  LayerNormLinear's forward reaches rmsnorm_fwd through module/_common.py, whose `tex`
    # is transformer_engine.pytorch.cpp_extensions (a snapshot of the extension's names), so that copy is rebound as
    # well; its backward imports transformer_engine_torch directly.
    _ORIG["rmsnorm_bwd"] = tex.rmsnorm_bwd
    _ORIG["rmsnorm_fwd"] = tex.rmsnorm_fwd
    tex.rmsnorm_bwd = _rmsnorm_bwd
    tex.rmsnorm_fwd = _rmsnorm_fwd_recording
    te_cpp.rmsnorm_bwd = _rmsnorm_bwd
    te_cpp.rmsnorm_fwd = _rmsnorm_fwd_recording
    te_ops_rmsnorm.rmsnorm_bwd = _rmsnorm_bwd
    parts = ["tex.rmsnorm_bwd", "cpp_extensions.rmsnorm_bwd", "ops.basic.rmsnorm.rmsnorm_bwd",
             "tex.rmsnorm_fwd(recorder)", "cpp_extensions.rmsnorm_fwd(recorder)"]

    # 4: TE grad_output_preprocess
    _ORIG["grad_output_preprocess"] = TransformerEngineBaseModule.grad_output_preprocess
    TransformerEngineBaseModule.grad_output_preprocess = staticmethod(
        _make_grad_output_preprocess(_ORIG["grad_output_preprocess"]))
    parts.append("TransformerEngineBaseModule.grad_output_preprocess")

    # 2 + 3: Megatron
    if FUSE1 or FUSE2:
        try:
            from megatron.core.fusions import fused_bias_dropout as fbd
            _ORIG["_bias_dropout_add_func"] = fbd._bias_dropout_add_func
            fbd._bias_dropout_add_func = _bias_dropout_add_func
            parts.append("fused_bias_dropout._bias_dropout_add_func")
        except Exception as ex:  # noqa
            _fallback(f"megatron bias-dropout-add patch failed: {type(ex).__name__}: {ex}")
    if FUSE2:
        try:
            from megatron.core.transformer.transformer_layer import TransformerLayer
            if _install_megatron_norm2(TransformerLayer):
                parts.append("TransformerLayer._pre_mlp_layernorm_and_residual")
        except Exception as ex:  # noqa
            _fallback(f"megatron norm2 patch failed: {type(ex).__name__}: {ex}")

    _INSTALLED = True
    _say(f"installed: {', '.join(parts)}; FUSE1={FUSE1} FUSE2={FUSE2} BLOCK_N={BLOCK_N} NUM_WARPS={NUM_WARPS} "
         f"CUDA_WANTED={CUDA_WANTED} MAIN_GRAD={MAIN_GRAD} (the CUDA backend loads at the first backward, eagerly)")
    return True


def stats():
    return dict(_STATS, pending_res1=len(_PENDING_RES1), norm1_inputs=len(_NORM1_INPUTS), grad_side=len(_GRAD_SIDE),
                cuda_loaded=_CUDA["ok"])
