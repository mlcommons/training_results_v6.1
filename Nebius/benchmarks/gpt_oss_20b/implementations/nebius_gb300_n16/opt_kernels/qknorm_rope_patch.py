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

import os

import torch

_ENABLED = os.environ.get("OPT_QKNORM_ROPE", "1") == "1"
_SELFCHECK = int(os.environ.get("OPT_QKNORM_ROPE_SELFCHECK", "1"))
_STRICT = os.environ.get("OPT_QKNORM_ROPE_STRICT", "0") == "1"
ACTIVE = False
_CONFIGURED = False
_FAST_OK = True
_ORIG = {}
_STATS = {"retyped": 0, "fused": 0, "fallback": 0}
_announced = set()
_QKNorm = None


def _rank():
    return int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))


def _announce(key, msg):
    if key in _announced:
        return
    _announced.add(key)
    if _rank() == 0:
        print(f"[opt qknorm_rope] {msg}", flush=True)


def _retypable(norm):
    import transformer_engine.pytorch as te

    return (
        isinstance(norm, te.RMSNorm)
        and norm.weight.numel() == 64
        and norm.weight.device.type != "meta"
        and not getattr(norm, "returns_residual", False)
    )


def _init(self, *args, **kwargs):
    global _CONFIGURED
    _ORIG["init"](self, *args, **kwargs)
    if not getattr(self.config, "qk_layernorm", False):
        return
    _CONFIGURED = True
    qn, kn = self.q_layernorm, self.k_layernorm
    if _retypable(qn) and _retypable(kn):
        qn.__class__ = _QKNorm
        kn.__class__ = _QKNorm
        _STATS["retyped"] += 2
        return
    if _STRICT:
        raise RuntimeError("[opt qknorm_rope] q/k norm modules not retypable")
    _announce("noretype", f"norm modules not retypable ({type(qn).__name__}); split path kept")


def _gqkv(self, *args, **kwargs):
    out = _ORIG["gqkv"](self, *args, **kwargs)
    qn = getattr(self, "q_layernorm", None)
    if isinstance(qn, _QKNorm) and isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
        kn = self.k_layernorm
        qn(None)
        kn(None)
        out[0]._opt_qknorm = (
            qn,
            kn,
            float(self.config.layernorm_epsilon),
            bool(self.config.layernorm_zero_centered_gamma),
        )
    return out


def _unsupported(qkv, q_freqs, k_freqs, split, tensor_format, start_positions, interleaved, cp_size):
    if qkv.dim() != 4 or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return "qkv layout"
    s, _, ng, gsz = qkv.shape
    nq = gsz // 64 - 2
    if nq < 1 or gsz != (nq + 2) * 64 or list(split) != [nq * 64, 64, 64]:
        return "split"
    if ng & (ng - 1) or nq & (nq - 1):
        return "group count"
    if tensor_format != "sbhd" or start_positions is not None or interleaved or cp_size != 1:
        return "rope mode"
    for f in (q_freqs, k_freqs):
        if f.dtype != torch.float32 or not f.is_contiguous() or f.shape[-1] != 64 or f.numel() < s * 64:
            return "freqs"
    return None


def _composed(qkv, q_freqs, k_freqs, split, qn, kn):
    import transformer_engine.pytorch as te
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb

    s, b, ng, _ = qkv.shape
    nq = split[0] // 64
    q = qkv[..., : split[0]].reshape(s, b, ng * nq, 64)
    k = qkv[..., split[0] : split[0] + split[1]].contiguous()
    v = qkv[..., split[0] + split[1] :].contiguous()
    q = te.RMSNorm.forward(qn, q)
    k = te.RMSNorm.forward(kn, k)
    q = fused_apply_rotary_pos_emb(q, q_freqs, interleaved=False)
    k = fused_apply_rotary_pos_emb(k, k_freqs, interleaved=False)
    return q, k, v


def _rel(a, b):
    d = (a.float() - b.float()).abs()
    return (d.max() / b.float().abs().max().clamp_min(1e-30)).item(), (
        d.mean() / b.float().abs().mean().clamp_min(1e-30)
    ).item()


def _selfcheck(qkv, q_freqs, k_freqs, split, qn, kn, eps, zc):
    from opt_kernels import qknorm_rope as kr

    global _FAST_OK
    with torch.no_grad():
        q, k, v = kr.qknorm_rope(qkv, q_freqs, k_freqs, qn.weight, kn.weight, eps, zc)
        rq, rk, rv = _composed(qkv, q_freqs, k_freqs, split, qn, kn)
    q_rel, k_rel = _rel(q, rq), _rel(k, rk)
    ok = (
        q_rel[0] <= 2**-6
        and k_rel[0] <= 2**-6
        and q_rel[1] <= 2**-9
        and k_rel[1] <= 2**-9
        and torch.equal(v, rv)
    )
    msg = (
        f"selfcheck fwd q rel {q_rel[0]:.2e}/{q_rel[1]:.2e} k rel {k_rel[0]:.2e}/{k_rel[1]:.2e} "
        f"v equal {torch.equal(v, rv)}"
    )
    if ok and _SELFCHECK >= 2:
        x = qkv.detach().clone().requires_grad_(True)
        gq = qn.weight.detach().clone().requires_grad_(True)
        gk = kn.weight.detach().clone().requires_grad_(True)
        q, k, v = kr.qknorm_rope(x, q_freqs, k_freqs, gq, gk, eps, zc)
        gen = torch.Generator(device=qkv.device).manual_seed(0)
        dq, dk, dv = (torch.randn(t.shape, generator=gen, device=qkv.device).to(t.dtype) for t in (q, k, v))
        dx, dgq, dgk = torch.autograd.grad((q, k, v), (x, gq, gk), (dq, dk, dv))
        rdx, rdgq, rdgk = kr.reference_backward(
            qkv, q_freqs, k_freqs, qn.weight, kn.weight, eps, zc, dq, dk, dv
        )
        dx_rel, gq_rel, gk_rel = _rel(dx, rdx), _rel(dgq, rdgq), _rel(dgk, rdgk)
        ok = dx_rel[0] <= 2**-6 and gq_rel[0] <= 2**-6 and gk_rel[0] <= 2**-6
        msg += f"; bwd dx rel {dx_rel[0]:.2e} dgq rel {gq_rel[0]:.2e} dgk rel {gk_rel[0]:.2e}"
    _announce("selfcheck", msg + (" ok" if ok else " FAILED"))
    if not ok:
        if _STRICT:
            raise RuntimeError("[opt qknorm_rope] self-check failed")
        _FAST_OK = False


def _dispatch(
    qkv,
    q_freqs,
    k_freqs,
    qkv_split_arg_list,
    tensor_format="sbhd",
    start_positions=None,
    interleaved=False,
    cu_seqlens=None,
    cp_size=1,
    cp_rank=0,
):
    tag = qkv.__dict__.pop("_opt_qknorm", None)
    if tag is None:
        if _CONFIGURED:
            raise RuntimeError("[opt qknorm_rope] qkv reached the fused RoPE without the norm tag")
        return _ORIG["te_apply"](
            qkv,
            q_freqs,
            k_freqs,
            qkv_split_arg_list,
            tensor_format,
            start_positions,
            interleaved,
            cu_seqlens,
            cp_size,
            cp_rank,
        )
    qn, kn, eps, zc = tag
    why = _unsupported(
        qkv, q_freqs, k_freqs, qkv_split_arg_list, tensor_format, start_positions, interleaved, cp_size
    )
    if (
        why is None
        and _FAST_OK
        and _SELFCHECK
        and "selfcheck" not in _announced
        and not torch.cuda.is_current_stream_capturing()
    ):
        _selfcheck(qkv, q_freqs, k_freqs, qkv_split_arg_list, qn, kn, eps, zc)
    if why is None and _FAST_OK:
        from opt_kernels import qknorm_rope as kr

        _count("fused")
        return kr.qknorm_rope(qkv, q_freqs, k_freqs, qn.weight, kn.weight, eps, zc)
    if _STRICT:
        raise RuntimeError(f"[opt qknorm_rope] fused path unavailable: {why or 'self-check failed'}")
    _count("fallback")
    _announce("fallback", f"composed fallback in use: {why or 'self-check failed'}")
    return _composed(qkv, q_freqs, k_freqs, qkv_split_arg_list, qn, kn)


def _count(kind):
    _STATS[kind] += 1
    if _STATS["fused"] + _STATS["fallback"] == max(1, _STATS["retyped"] // 2):
        _announce(
            "stats",
            f"first forward: retyped={_STATS['retyped']} fused={_STATS['fused']} "
            f"fallback={_STATS['fallback']}",
        )


def stats():
    return dict(_STATS)


def _install():
    global ACTIVE, _QKNorm
    try:
        import transformer_engine.pytorch as te
        from megatron.core.transformer import attention as mcore_attention
        from megatron.core.transformer.identity_op import IdentityOp

        from opt_kernels import qknorm_rope  # noqa: F401
    except Exception as e:
        _announce("import", f"disabled: {e!r}")
        return
    if mcore_attention.apply_fused_qkv_rotary_pos_emb is None:
        _announce("import", "disabled: TE fused QKV RoPE unavailable")
        return

    class _QKNormImpl(te.RMSNorm, IdentityOp):
        def forward(self, x=None, *args, **kwargs):
            if x is None:
                return None
            return te.RMSNorm.forward(self, x, *args, **kwargs)

    _QKNorm = _QKNormImpl
    _ORIG["init"] = mcore_attention.SelfAttention.__init__
    _ORIG["gqkv"] = mcore_attention.SelfAttention.get_query_key_value_tensors
    _ORIG["te_apply"] = mcore_attention.apply_fused_qkv_rotary_pos_emb
    mcore_attention.SelfAttention.__init__ = _init
    mcore_attention.SelfAttention.get_query_key_value_tensors = _gqkv
    mcore_attention.apply_fused_qkv_rotary_pos_emb = _dispatch
    ACTIVE = True
    _announce("installed", "installed")


if _ENABLED:
    _install()
