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

"""Unit tests + microbenchmark for hs_lora_epilogue.py (fix #2: LoRA adds/scales folded into GEMM epilogues).
Runs inside the recipe image on one GPU (a few minutes; the attention block is the production shape):

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_lora_epilogue.py [--bench | --bench-only]      # or
      ... -m pytest -s -q /p/test_hs_lora_epilogue.py
  The import/wiring check alone needs no GPU: docker run --rm -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_lora_epilogue.py --wiring-only

Checks (production shapes M=8192 tokens, 8192 -> 10240 / 8192, rank 16, alpha 32, dropout 0.1):
  1. the two Triton kernels against an fp32 reference and against the current BF16 op sequence;
  2. the hipBLASLt beta=1 variant (one kernel) and TE's general_gemm(accumulate=True) on FP8 inputs;
  3. hs.lora_apply against NeMo's own ParallelLinearAdapter forward (outputs and A/B/input grads);
  4. the dgrad fold through a real TE LayerNormLinear (qkv) and TE Linear + residual (proj), FP8 frozen weights:
     outputs, grads, and the elementwise kernel counts of the backward with and without the fold;
  5. the whole Megatron SelfAttention block with the NeMo mixin and both LoRA adapters, patched vs the original
     NeMo methods, for every item configuration;
  6. wiring: the patched functions are what NeMo/Megatron/TE resolve at call time.
The benchmark (CUDA events) times the attention block forward and backward per configuration and the individual
kernels, and extrapolates to a step (forward x 91 layer-forwards, backward x 80).
"""

import os
import sys

# the recipe's TE settings that matter here (config_MI300X_1x8x1.sh)
for _k, _v in {"NVTE_USE_RMSNORM_TRITON": "1", "NVTE_FUSED_ATTN": "1", "NVTE_FUSED_ATTN_CK": "1",
               "NVTE_FUSED_ATTN_AOTRITON": "1", "NVTE_CK_USES_BWD_V3": "1", "NVTE_CK_USES_FWD_V3": "1", "NVTE_CK_IS_V3_ATOMIC_FP32": "0",
               "NVTE_USE_HIPBLASLT": "1", "NVTE_USE_CAST_TRANSPOSE_TRITON": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
               "HS_LORA_EPILOGUE": "1"}.items():
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402
import transformer_engine_torch as tex  # noqa: E402
from transformer_engine.common.recipe import DelayedScaling, Format  # noqa: E402

import hs_lora_epilogue as hs  # noqa: E402  (installs the TE hooks, the bda patch and the NeMo mixin patch)

DEV = "cuda"
BF16 = torch.bfloat16
M, H, QKV, R = 8192, 8192, 10240, 16
ALPHA, P_DROP = 32.0, 0.1
SCALE = ALPHA / R
N_FWD, N_BWD = 91, 80  # layer-forwards (80 + 11 recomputed) and layer-backwards per step in recipe A
STEP_MS = 3693.0
RECIPE = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=4, amax_compute_algo="most_recent")


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _randn(shape, std=1.0, seed=0, dtype=BF16):
    return (torch.randn(shape, generator=_gen(seed), device=DEV) * std).to(dtype)


def _report(out, ref, what, mag=None, ulp_tol=2.0 ** -7, abs_tol=2.0 ** -16, extra=None, assert_tol=True):
    """Compare out (bf16) with ref (bf16 or wider). Bound: |out - ref| <= ulp_tol * |ref| + abs_tol * mag + extra;
    the mag part covers fp32 reduction-order noise where base and LoRA term cancel (mag = magnitude of the terms,
    default max |ref|), extra the reference path's own intermediate BF16 roundings (e.g. one ulp of the LoRA term
    for the current mm -> mul -> add sequence). Prints max abs, the error in bf16 ulps of ref, mean abs and the
    bit-identical fraction."""
    o, r = out.double(), ref.double()
    d = (o - r).abs()
    rb = ref.to(BF16).double() if ref.dtype != BF16 else r
    eq = (o == rb).double().mean().item()
    if mag is None:
        mag = r.abs().max()
    floor = (abs_tol * mag).clamp_min(1e-30) if isinstance(mag, torch.Tensor) else max(abs_tol * mag, 1e-30)
    ulp = torch.ldexp(torch.ones_like(r), torch.floor(torch.log2(torch.maximum(torch.maximum(r.abs(), o.abs()), torch.as_tensor(floor, device=r.device, dtype=r.dtype)))) - 7)
    max_ulp = (d / ulp).max().item()
    tol = ulp_tol * r.abs() + abs_tol * mag + (extra if extra is not None else 0.0)
    worst = (d - tol).max().item()
    tag = "bit-identical" if eq == 1.0 else f"differs by rounding ({100 * (1 - eq):.4f}% of elements)"
    print(f"  {what}: max abs {d.max().item():.3e}, max {max_ulp:.2f} bf16-ulp, mean abs {d.mean().item():.2e}, "
          f"equal {100 * eq:.4f}% -> {tag}")
    if assert_tol:
        assert worst <= 0, (what, d.max().item(), max_ulp, eq, worst)
    return eq == 1.0


def _report_mod(out, ref, what, ulp_tol=2.0 ** -6):
    """Module-level comparison against the current NeMo path, which rounds its intermediates (the LoRA term, and
    base + lora before the residual add) to BF16: allow one BF16 ulp of the largest value (2^-7 * max |ref| * 2)
    on top of ulp_tol * |ref|. The kernel-level tests carry the tight bounds."""
    return _report(out, ref, what, ulp_tol=ulp_tol, abs_tol=2.0 ** -6)


def _report_noisy(out, ref, noise, what, tol):
    """Attention-block comparison: fused attention's backward is not deterministic and 1-ulp qkv differences are
    amplified through the softmax over 8192 keys, so per-element ulp bounds do not apply. Judge the relative L1
    error (mean |out - ref| / mean |ref|) against max(tol, 32 x the reference-vs-reference noise floor)."""
    o, r, n = out.double(), ref.double(), noise.double()
    d = (o - r).abs()
    rel = d.mean().item() / max(r.abs().mean().item(), 1e-30)
    rel_noise = (n - r).abs().mean().item() / max(r.abs().mean().item(), 1e-30)
    eq = (o == r).double().mean().item()
    print(f"  {what}: max abs {d.max().item():.3e}, rel-L1 {rel:.2e} (noise floor {rel_noise:.2e}), "
          f"equal {100 * eq:.2f}%")
    assert rel <= max(tol, 32 * rel_noise), (what, rel, rel_noise)


def _set_flags(add=True, residual=True, bwd=True, impl="triton"):
    hs.FLAGS.master, hs.FLAGS.add, hs.FLAGS.residual, hs.FLAGS.bwd, hs.FLAGS.impl = True, add, residual, bwd, impl


def _kernel_counts(fn):
    """GPU kernel launches of fn() by family (elementwise add / scalar mul / masked_scale / dropout / GEMM / other)."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    fam = {"add": 0, "mul": 0, "masked_scale": 0, "dropout": 0, "gemm": 0, "triton_lora": 0, "other": 0}
    for e in p.key_averages():
        if e.self_device_time_total <= 0:  # GPU kernels / memcpys only (runtime launch events have no device time)
            continue
        n = e.key
        c = e.count
        if "CUDAFunctor_add" in n:
            fam["add"] += c
        elif "AUnaryFunctor" in n or "BUnaryFunctor" in n or "CUDAFunctor_mul" in n:
            fam["mul"] += c
        elif "masked_scale" in n:
            fam["masked_scale"] += c
        elif "fused_dropout" in n:
            fam["dropout"] += c
        elif n.startswith("Cijk_") or "gemm" in n.lower():
            fam["gemm"] += c
        elif "_lora_" in n:
            fam["triton_lora"] += c
        else:
            fam["other"] += c
    return fam


# ------------------------------------------------------------------------------------------------------ kernels


def _up_cases():
    for (m, n), seed in (((M, QKV), 1), ((M, H), 2), ((37, 203), 3)):
        for b_std in (0.05, 0.5):  # LoRA term ~5% of the base, and comparable to it
            base = _randn((m, n), 1.8, seed)
            a = _randn((m, R), 0.3, seed + 10)
            b = _randn((n, R), b_std, seed + 20)
            res = _randn((m, n), 1.0, seed + 30)
            yield m, n, b_std, base, a, b, res


def test_up_kernel():
    for m, n, b_std, base, a, b, res in _up_cases():
        for with_res in (False, True):
            lora64 = SCALE * (a.double() @ b.double().t())
            ref64 = base.double() + lora64 + (res.double() if with_res else 0.0)
            mag = base.double().abs() + SCALE * (a.double().abs() @ b.double().abs().t()) + (res.double().abs() if with_res else 0.0)
            out = hs.lora_up(base, a, b, SCALE, res if with_res else None)
            cur = (torch.matmul(a, b.t()) * SCALE + base)  # the current op order: mm (bf16), scalar mul, add
            if with_res:
                cur = res + cur
            what = f"up [{m}x{n}] B~{b_std}{' +res' if with_res else ''}"
            _report(out, ref64, what + " vs fp64 ref", mag)
            # the current path rounds the LoRA term (1 ulp) and, with the residual, also base + lora (0.5 ulp)
            extra = 2.0 ** -7 * lora64.abs() + (2.0 ** -8 * (base.double() + lora64).abs() if with_res else 0.0)
            _report(out, cur, what + " vs current bf16 path", mag, extra=extra)
            e_new, e_cur = (out.double() - ref64).abs().mean().item(), (cur.double() - ref64).abs().mean().item()
            print(f"    mean abs error vs fp64: fused {e_new:.3e}, current path {e_cur:.3e}")
            assert e_new <= e_cur


def test_dgrad_kernel():
    for (m, n), seed in (((M, H), 4), ((M, QKV), 5), ((37, 203), 6)):
        dgrad = _randn((m, n), 0.02, seed)
        ga = _randn((m, R), 0.05, seed + 10)
        a_w = _randn((R, n), 0.01, seed + 20)  # A weight [rank, in]
        mask = torch.rand((m, n), generator=_gen(seed + 30), device=DEV) >= P_DROP
        for with_mask in (True, False):
            scale = 1.0 / (1.0 - P_DROP) if with_mask else 1.0  # as _LoraDown.backward passes it
            mk = mask if with_mask else None
            mf = mask.double() if with_mask else 1.0
            ref64 = dgrad.double() + mf * scale * (ga.double() @ a_w.double())
            mag = dgrad.double().abs() + scale * (ga.double().abs() @ a_w.double().abs())
            out = dgrad.clone()
            hs.lora_dgrad_acc_(out, ga, a_w, mk, scale)
            t = torch.matmul(ga, a_w)  # current path: mm (bf16), dropout backward (masked_scale), add
            if with_mask:
                t = torch.ops.aten.native_dropout_backward(t, mask, scale)
            cur = dgrad + t
            what = f"dgrad-acc [{m}x{n}]{' mask' if with_mask else ''}"
            _report(out, ref64, what + " vs fp64 ref", mag)
            _report(out, cur, what + " vs current bf16 path", mag, extra=2.0 ** -6 * (ref64 - dgrad.double()).abs())
            e_new, e_cur = (out.double() - ref64).abs().mean().item(), (cur.double() - ref64).abs().mean().item()
            print(f"    mean abs error vs fp64: fused {e_new:.3e}, current path {e_cur:.3e}")
            assert e_new <= e_cur


def test_blas_variant_and_te_accumulate():
    base = _randn((M, QKV), 1.8, 7)
    a = _randn((M, R), 0.3, 8)
    b = _randn((QKV, R), 0.05, 9)
    ref32 = base.float() + SCALE * (a.float() @ b.float().t())
    x = base.clone()
    fam = _kernel_counts(lambda: x.copy_(base).view(M, QKV).addmm_(a, b.t(), alpha=SCALE))
    x.copy_(base).addmm_(a, b.t(), alpha=SCALE)
    _report(x, ref32.to(BF16), "hipBLASLt addmm_ (beta=1, alpha) vs fp32 ref",
            base.double().abs() + SCALE * (a.double().abs() @ b.double().abs().t()))
    assert fam["gemm"] == 1 and fam["add"] == 0 and fam["mul"] == 0, fam
    print(f"  addmm_ launches: {fam}")
    # TE general_gemm(accumulate=True): FP8 x FP8 -> BF16 with beta = 1 on the BF16 out (hipBLASLt honours it)
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.module.base import get_workspace
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    q = lambda: Float8Quantizer(scale=torch.ones(1, device=DEV), amax=torch.zeros(1, device=DEV),  # noqa: E731
                                fp8_dtype=tex.DType.kFloat8E4M3)
    w = q()(_randn((QKV, H), 0.02, 10))
    xq = q()(_randn((M, H), 1.0, 11))
    out0, *_ = general_gemm(w, xq, get_workspace(), out_dtype=BF16, layout="TN")
    c = _randn((M, QKV), 1.0, 12)
    d = c.clone()
    general_gemm(w, xq, get_workspace(), out_dtype=BF16, layout="TN", out=d, accumulate=True)
    _report(d, (c.float() + out0.float()).to(BF16), "TE FP8 GEMM accumulate=True vs GEMM + add (beta=1 honoured)",
            c.double().abs() + out0.double().abs(), extra=2.0 ** -7 * out0.double().abs())


# ------------------------------------------------------------------------------------------- Megatron/NeMo setup


def _init_parallel():
    import megatron.core.parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    if not ps.model_parallel_is_initialized():
        ps.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(1234)


def _config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=64, num_query_groups=8, kv_channels=128,
        ffn_hidden_size=28672, use_cpu_initialization=False, bf16=True, params_dtype=BF16, normalization="RMSNorm",
        add_bias_linear=False, gated_linear_unit=True, activation_func=torch.nn.functional.silu, hidden_dropout=0.0,
        attention_dropout=0.0, apply_rope_fusion=True, bias_dropout_fusion=False, bias_activation_fusion=True,
        fp8="hybrid", fp8_margin=0, fp8_amax_history_len=4, fp8_amax_compute_algo="most_recent",
        disable_parameter_transpose_cache=True, gradient_accumulation_fusion=False, layernorm_epsilon=1e-5,
        attention_softmax_in_fp32=True, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        sequence_parallel=False, init_method_std=0.02,
    )


def _mp_config():
    """What NeMo passes as model_parallel_config to add_adapter (build_model_parallel_config): a ModelParallelConfig,
    not the TransformerConfig (hydra's instantiate wraps dataclass kwargs in OmegaConf, and TransformerConfig has
    int fields holding None)."""
    from megatron.core import ModelParallelConfig

    return ModelParallelConfig(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, bf16=True,
                               params_dtype=BF16, gradient_accumulation_fusion=False, sequence_parallel=False,
                               use_cpu_initialization=False)


def _make_adapter(in_f, out_f, kind, cfg, seed):
    """A NeMo LoRA adapter as the recipe builds it (peft_config._create_lora_config), with a non-zero B."""
    from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
        LoraDenseAttentionAdapter, LoraKQVAdapter)

    cls = LoraKQVAdapter if kind == "qkv" else LoraDenseAttentionAdapter
    ad = cls(in_features=in_f, out_features=out_f, dim=R, norm_position=None, norm_type=None, activation="identity",
             column_init_method="kaiming", row_init_method="zero", gather_output=False,
             input_is_parallel=(kind == "proj"), dropout=P_DROP, model_parallel_config=cfg, alpha=ALPHA,
             dropout_position="pre", a2a_experimental=False)
    with torch.no_grad():
        ad.linear_out.weight.copy_(_randn(ad.linear_out.weight.shape, 0.05, seed))
    assert ad.linear_in.weight.shape == (R, in_f) and ad.linear_out.weight.shape == (out_f, R)
    assert hs.adapter_foldable(ad)
    return ad.train()


def _fp8_state(module):
    ts = []
    for m in module.modules():
        meta = getattr(m, "fp8_meta", None)
        if not meta:
            continue
        for k in ("scaling_fwd", "scaling_bwd"):
            if k in meta:
                for attr in ("scale", "scale_inv", "amax_history"):
                    t = getattr(meta[k], attr, None)
                    if isinstance(t, torch.Tensor):
                        ts.append((t, t.clone()))
    return ts


def _restore_fp8(state):
    for t, c in state:
        t.copy_(c)


def _grads(*params):
    out = []
    for p in params:
        out.append(None if p.grad is None else p.grad.detach().clone())
        p.grad = None
    return out


# ---------------------------------------------------------------------------------- lora_apply vs NeMo adapter


def test_lora_apply_vs_nemo_adapter():
    _init_parallel()
    cfg = _config()
    for kind, in_f, out_f in (("qkv", H, QKV), ("proj", H, H)):
        ad = _make_adapter(in_f, out_f, kind, cfg, seed=21)
        x0 = _randn((M, 1, in_f), 1.0, 22)
        base0 = _randn((M, 1, out_f), 1.8, 23)
        g = _randn((M, 1, out_f), 0.01, 24)
        a_w, b_w = ad.linear_in.weight, ad.linear_out.weight

        def run(path, impl="triton"):
            torch.cuda.manual_seed(77)  # same dropout mask in both paths
            x = x0.clone().requires_grad_()
            base_leaf = base0.clone().requires_grad_()
            base = base_leaf * 1.0  # non-leaf, like TE's output (the blas variant modifies it in place)
            if path == "nemo":
                y = base + ad(x)
            else:
                _set_flags(add=path != "plain", residual=False, bwd=False, impl=impl)
                y, _ = hs.lora_apply(ad, x, base, None, None)
            y.backward(g)
            return y.detach(), *_grads(x, base_leaf, a_w, b_w)

        ref = run("nemo")
        for path, impl in (("plain", "triton"), ("fused", "triton"), ("fused", "blas")):
            out = run(path, impl)
            print(f" {kind} lora_apply {path}/{impl} vs NeMo adapter:")
            same = _report_mod(out[0], ref[0], "  out")
            if path == "plain":
                assert same, "ADD off must reproduce NeMo's op sequence bit for bit"
            assert _report(out[1], ref[1], "  grad x (through dropout backward)")
            assert _report(out[2], ref[2], "  grad base")
            assert _report(out[3], ref[3], "  grad A"), "grad A must be bit-identical (scale is a power of two)"
            assert _report(out[4], ref[4], "  grad B"), "grad B must be bit-identical (scale is a power of two)"
    _set_flags()


# ------------------------------------------------------------------------------------------ TE module folds


def test_te_layernorm_linear_fold():
    """qkv: TE LayerNormLinear (FP8, frozen) returning ln_out; the LoRA A dgrad folded before TE's RMSNorm backward."""
    _init_parallel()
    cfg = _config()
    with te.fp8_model_init(enabled=True, recipe=RECIPE):
        lnl = te.LayerNormLinear(H, QKV, eps=1e-5, bias=False, normalization="RMSNorm", return_layernorm_output=True,
                                 params_dtype=BF16, device=DEV)
    for p in lnl.parameters():
        p.requires_grad_(False)
    ad = _make_adapter(H, QKV, "qkv", cfg, seed=31)
    a_w, b_w = ad.linear_in.weight, ad.linear_out.weight
    h0 = _randn((M, 1, H), 1.0, 32)
    g = _randn((M, 1, QKV), 0.01, 33)
    # warm the FP8 scales (DelayedScaling: two iterations to settle), then snapshot
    for _ in range(2):
        with te.fp8_autocast(enabled=True, fp8_recipe=RECIPE):
            out, _ = lnl(h0.clone().requires_grad_())
        out.backward(g)
    state = _fp8_state(lnl)

    def run(path):
        _restore_fp8(state)
        torch.cuda.manual_seed(78)
        h = h0.clone().requires_grad_()
        with te.fp8_autocast(enabled=True, fp8_recipe=RECIPE):
            base, ln_out = lnl(h)
        assert ln_out.requires_grad and hs.find_base_node(ln_out, "_LayerNormLinearBackward") is not None
        if path == "nemo":
            y = base + ad(ln_out)
        else:
            _set_flags(add="add" in path, residual=False, bwd="bwd" in path)
            y, _ = hs.lora_apply(ad, ln_out, base, None, "_LayerNormLinearBackward")
        y.backward(g)
        return y.detach(), *_grads(h, a_w, b_w)

    ref = run("nemo")
    for path in ("bwd", "add", "add+bwd"):
        before = dict(hs.STATS)
        out = run(path)
        print(f" LayerNormLinear + LoRA, {path} vs NeMo:")
        _report_mod(out[0], ref[0], "  out")
        _report_mod(out[1], ref[1], "  grad h (through RMSNorm backward)", ulp_tol=2.0 ** -5)
        assert _report(out[2], ref[2], "  grad A")
        assert _report(out[3], ref[3], "  grad B")
        if "bwd" in path:
            assert hs.STATS["bwd_folded"] == before["bwd_folded"] + 1, "the dgrad fold did not happen"
    # kernel accounting of forward + backward, current path vs all folds
    def fb(path):
        def f():
            run(path)
        return f
    k_ref, k_new = _kernel_counts(fb("nemo")), _kernel_counts(fb("add+bwd"))
    print(f"  kernels fwd+bwd NeMo: {k_ref}\n  kernels fwd+bwd add+bwd: {k_new}")
    assert k_new["add"] == k_ref["add"] - 2 and k_new["mul"] == k_ref["mul"] - 2  # 2 adds (fwd add, TE's dgrad add)
    assert k_new["masked_scale"] == k_ref["masked_scale"] - 1 and k_new["gemm"] == k_ref["gemm"] - 2
    _set_flags()


def test_te_linear_fold_with_residual():
    """proj: TE Linear (FP8, frozen); LoRA A dgrad folded into _Linear's dgrad; the residual folded in the forward."""
    _init_parallel()
    cfg = _config()
    with te.fp8_model_init(enabled=True, recipe=RECIPE):
        lin = te.Linear(H, H, bias=False, params_dtype=BF16, device=DEV)
    lin.weight.requires_grad_(False)
    ad = _make_adapter(H, H, "proj", cfg, seed=41)
    a_w, b_w = ad.linear_in.weight, ad.linear_out.weight
    x0 = _randn((M, 1, H), 1.0, 42)
    r0 = _randn((M, 1, H), 1.0, 43)
    g = _randn((M, 1, H), 0.01, 44)
    for _ in range(2):
        with te.fp8_autocast(enabled=True, fp8_recipe=RECIPE):
            out = lin(x0.clone().requires_grad_())
        out.backward(g)
    state = _fp8_state(lin)
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

    bda = get_bias_dropout_add(True, False)  # what TransformerLayer uses with bias_dropout_fusion False

    def run(path):
        _restore_fp8(state)
        torch.cuda.manual_seed(79)
        x = x0.clone().requires_grad_()
        res = r0.clone().requires_grad_()
        with te.fp8_autocast(enabled=True, fp8_recipe=RECIPE):
            base = lin(x)
        assert hs.find_base_node(base, "_LinearBackward") is not None
        if path == "nemo":
            y = base + ad(x)
        else:
            _set_flags(add="add" in path, residual="res" in path, bwd="bwd" in path)
            y, folded = hs.lora_apply(ad, x, base, res, "_LinearBackward")
            assert folded == ("res" in path and "add" in path)
            if folded:
                y._hs_residual_folded = True
        hidden = bda((y, None), res, 0.0)  # the layer's residual add (a pass-through when folded)
        hidden.backward(g)
        return hidden.detach(), *_grads(x, res, a_w, b_w)

    ref = run("nemo")
    for path in ("bwd", "add", "add+res", "add+res+bwd"):
        before = dict(hs.STATS)
        out = run(path)
        print(f" Linear + LoRA + residual, {path} vs NeMo:")
        _report_mod(out[0], ref[0], "  hidden")
        _report_mod(out[1], ref[1], "  grad x (base dgrad + folded LoRA dgrad)")
        assert _report(out[2], ref[2], "  grad residual")
        assert _report(out[3], ref[3], "  grad A")
        assert _report(out[4], ref[4], "  grad B")
        if "bwd" in path:
            assert hs.STATS["bwd_folded"] == before["bwd_folded"] + 1
        if "res" in path:
            assert hs.STATS["residual_passthrough"] == before["residual_passthrough"] + 1
    def fb(path):
        def f():
            run(path)
        return f
    k_ref, k_new = _kernel_counts(fb("nemo")), _kernel_counts(fb("add+res+bwd"))
    print(f"  kernels fwd+bwd NeMo: {k_ref}\n  kernels fwd+bwd add+res+bwd: {k_new}")
    assert k_new["add"] == k_ref["add"] - 3 and k_new["mul"] == k_ref["mul"] - 2  # LoRA add, residual add, dgrad add_
    assert k_new["masked_scale"] == k_ref["masked_scale"] - 1 and k_new["gemm"] == k_ref["gemm"] - 2
    _set_flags()


# ------------------------------------------------------------------------------------------ attention block


_ATTN = {}


def _attention_block():
    """Megatron SelfAttention (TE spec, FP8 frozen weights) with NeMo's mixin and both LoRA adapters, as in the
    recipe, plus the TransformerLayer's residual add; returns (attn, step_fn)."""
    if _ATTN:
        return _ATTN["attn"], _ATTN["step"]
    _init_parallel()
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import build_module
    from nemo.collections.nlp.modules.common.megatron.adapters.mcore_mixins import (
        MCoreSelfAttentionMixin, swap_mcore_mixin)
    from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
        AdapterName, LoraDenseAttentionAdapterConfig, LoraKQVAdapterConfig)

    cfg = _config()
    spec = get_gpt_layer_with_transformer_engine_spec().submodules.self_attention
    with te.fp8_model_init(enabled=True, recipe=RECIPE):
        attn = build_module(spec, config=cfg, layer_number=1)
    for p in attn.parameters():
        p.requires_grad_(False)
    swap_mcore_mixin(attn, MCoreSelfAttentionMixin)
    lora_kw = dict(dim=R, norm_position=None, norm_type=None, activation="identity", column_init_method="kaiming",
                   row_init_method="zero", gather_output=False, dropout=P_DROP, alpha=ALPHA, dropout_position="pre",
                   a2a_experimental=False)
    mp_cfg = _mp_config()
    attn.add_adapter(name=AdapterName.LORA_KQV_ADAPTER, cfg=LoraKQVAdapterConfig(in_features=H, out_features=QKV,
                                                                                **lora_kw), model_parallel_config=mp_cfg)
    attn.add_adapter(name=AdapterName.LORA_DENSE_ATTENTION_ADAPTER,
                     cfg=LoraDenseAttentionAdapterConfig(in_features=H, out_features=H, **lora_kw),
                     model_parallel_config=mp_cfg)
    for i, ad in enumerate(attn.adapter_layer.values()):
        with torch.no_grad():
            ad.linear_out.weight.copy_(_randn(ad.linear_out.weight.shape, 0.05, 50 + i))
        assert hs.adapter_foldable(ad)
    attn.train()
    assert type(attn).forward is MCoreSelfAttentionMixin.forward and hs._HOOKED["mixin"]
    rot = RotaryEmbedding(kv_channels=128, rotary_percent=1.0)(M)
    bda = get_bias_dropout_add(True, False)
    h0 = _randn((M, 1, H), 1.0, 60)
    g = _randn((M, 1, H), 0.01, 61)
    params = [ad.linear_in.weight for ad in attn.adapter_layer.values()] + \
             [ad.linear_out.weight for ad in attn.adapter_layer.values()]

    def step(path, timing=None, seed=80):
        """One layer-forward + backward of the attention block (+ residual add); path 'nemo' uses the original
        NeMo methods, otherwise the patched ones with the given item set."""
        Mixin = MCoreSelfAttentionMixin
        if path == "nemo":
            Mixin.forward, Mixin.get_query_key_value_tensors = Mixin._hs_orig_forward, \
                Mixin._hs_orig_get_query_key_value_tensors
        else:
            Mixin.forward, Mixin.get_query_key_value_tensors = _ATTN["patched"]
            _set_flags(add="add" in path, residual="res" in path, bwd="bwd" in path,
                       impl="blas" if "blas" in path else "triton")
        try:
            torch.cuda.manual_seed(seed)
            h = h0.clone().requires_grad_()
            if timing:
                timing[0].record()
            with te.fp8_autocast(enabled=True, fp8_recipe=RECIPE):
                out, bias = attn(h, attention_mask=None, rotary_pos_emb=rot)
                hidden = bda((out, bias), h, 0.0)
            if timing:
                timing[1].record()
            hidden.backward(g)
            if timing:
                timing[2].record()
            return hidden.detach(), h.grad.detach().clone(), [p.grad.detach().clone() for p in params]
        finally:
            for p in params:
                p.grad = None
            h.grad = None
            Mixin.forward, Mixin.get_query_key_value_tensors = _ATTN["patched"]
            _set_flags()

    _ATTN.update(attn=attn, step=step, rot=rot, patched=(MCoreSelfAttentionMixin.forward,
                                                         MCoreSelfAttentionMixin.get_query_key_value_tensors))
    return attn, step


def test_attention_block():
    attn, step = _attention_block()
    for _ in range(2):
        step("nemo")
    state = _fp8_state(attn)
    _restore_fp8(state)
    ref = step("nemo")
    _restore_fp8(state)
    ref2 = step("nemo")
    repro = torch.equal(ref[0], ref2[0]) and all(torch.equal(a, b) for a, b in zip(ref[2], ref2[2]))
    print(f"  reference reproducible bit for bit (FP8 state restored, same dropout seed): {repro}"
          + ("" if repro else "  (fused-attention backward is not deterministic: bit-equality checks below are skipped)"))
    for path in ("plain", "add", "add+res", "bwd", "add+bwd", "add+res+bwd", "blas-add"):
        _restore_fp8(state)
        before = dict(hs.STATS)
        out = step(path)
        d = {k: hs.STATS[k] - before[k] for k in hs.STATS}
        print(f" attention block {path} vs NeMo:  (stats {d})")
        _report_noisy(out[0], ref[0], ref2[0], "  hidden (after residual)", 2.0 ** -6)
        _report_noisy(out[1], ref[1], ref2[1], "  grad hidden_states", 2.0 ** -4)
        names = ["A qkv", "A proj", "B qkv", "B proj"]
        for n_, o, r, n2 in zip(names, out[2], ref[2], ref2[2]):
            _report_noisy(o, r, n2, f"  grad {n_}", 2.0 ** -4)
        if path == "plain":
            if repro:
                assert torch.equal(out[0], ref[0]) and all(torch.equal(a, b) for a, b in zip(out[2], ref[2])), \
                    "all items off must reproduce NeMo's path bit for bit"
            assert d["up_plain"] == 2 and d["bwd_plain"] == 2 and d["fallback_adapter"] == 0
        if "add" in path:  # the blas impl falls back to the Triton kernel on TE outputs (views)
            assert d["up_fused"] + d["up_blas"] == 2 and d["up_plain"] == 0
        if "res" in path:
            assert d["residual_folded"] == 1 and d["residual_passthrough"] == 1
        else:
            assert d["residual_passthrough"] == 0
        if "bwd" in path:
            assert d["bwd_folded"] == 2 and d["bwd_plain"] == 0
        else:
            assert d["bwd_plain"] == 2 and d["bwd_folded"] == 0
    # no-grad forward (what Megatron's checkpoint does for the recomputed layers): no fold targets, plain kernels
    before = dict(hs.STATS)
    with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=RECIPE):
        attn(_randn((M, 1, H), 1.0, 62), attention_mask=None, rotary_pos_emb=_ATTN["rot"])
    d = {k: hs.STATS[k] - before[k] for k in hs.STATS}
    assert d["up_fused"] == 2 and d["residual_folded"] == 1 and d["bwd_folded"] == 0, d
    print("  no-grad forward: fused kernels, no fold targets", d)


def test_wiring():
    import megatron.core.fusions.fused_bias_dropout as fbd
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from nemo.collections.nlp.modules.common.megatron.adapters import mcore_mixins as mm
    from transformer_engine.pytorch.module import layernorm_linear as lnl_mod
    from transformer_engine.pytorch.module import linear as lin_mod

    assert all(hs._HOOKED.values()), hs._HOOKED
    Mixin = mm.MCoreSelfAttentionMixin
    assert Mixin.forward.__module__ == hs.__name__ and Mixin.get_query_key_value_tensors.__module__ == hs.__name__
    assert hasattr(Mixin, "_hs_orig_forward")
    assert lnl_mod._LayerNormLinear.backward.__name__ == "lnl_backward"
    assert lin_mod._Linear.backward.__name__ == "linear_backward"
    assert hasattr(tex.rmsnorm_bwd, "_hs_orig") and hasattr(lnl_mod.te_rmsnorm_bwd_triton, "_hs_orig")
    assert fbd._bias_dropout_add_func.__module__ == hs.__name__
    # the layer's bda (unfused closure) resolves the patched function at call time
    x = torch.randn(4, 1, 8, dtype=BF16)  # CPU: this test also runs without a GPU
    res = torch.randn(4, 1, 8, dtype=BF16)
    bda = get_bias_dropout_add(True, False)
    assert torch.equal(bda((x, None), res, 0.0), x + res)
    x._hs_residual_folded = True
    assert bda((x, None), res, 0.0) is x
    # a fresh import in the recipe's order: patch module first (train.py hook), NeMo afterwards -> same objects
    assert sys.modules["hs_lora_epilogue"] is hs
    print("  wiring: mixin methods, TE backward wrappers, norm-backward hooks and the bda pass-through resolve")


# ------------------------------------------------------------------------------------------------ benchmark


def _time_ms(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def test_benchmark():
    print("\n kernel-level (ms, one MI300X, 20 iters):")
    rows = []
    for (m, n), tag in (((M, QKV), "qkv [8192x10240]"), ((M, H), "proj [8192x8192]")):
        base, a, b, res = _randn((m, n), 1.8, 1), _randn((m, R), 0.3, 2), _randn((n, R), 0.05, 3), _randn((m, n), 1, 4)
        ga, a_w, dg = _randn((m, R), 0.05, 5), _randn((R, n), 0.01, 6), _randn((m, n), 0.02, 7)
        mask = torch.rand((m, n), generator=_gen(8), device=DEV) >= P_DROP
        x = _randn((m, n), 1.0, 9)
        t = {
            "mm K=16 (B-GEMM write)": _time_ms(lambda: torch.matmul(a, b.t())),
            "scalar mul": _time_ms(lambda: base * SCALE),
            "add": _time_ms(lambda: base + res),
            "fused up": _time_ms(lambda: hs.lora_up(base, a, b, SCALE)),
            "fused up + residual": _time_ms(lambda: hs.lora_up(base, a, b, SCALE, res)),
            "blas addmm_ (beta=1)": _time_ms(lambda: base.addmm_(a, b.t(), alpha=SCALE)),
            "dropout fwd": _time_ms(lambda: torch.native_dropout(x, P_DROP, True)),
            "mm K=16 (A dgrad write)": _time_ms(lambda: torch.matmul(ga, a_w)),
            "masked_scale (dropout bwd)": _time_ms(lambda: torch.ops.aten.native_dropout_backward(dg, mask, 1 / 0.9)),
            "fused dgrad-acc": _time_ms(lambda: hs.lora_dgrad_acc_(dg, ga, a_w, mask, 1 / 0.9)),
        }
        for k, v in t.items():
            print(f"  {tag:18s} {k:28s} {v:7.3f}")
        rows.append((tag, t))
    (qt, pt) = rows[0][1], rows[1][1]
    fwd_cur = (qt["mm K=16 (B-GEMM write)"] + qt["scalar mul"] + qt["add"]) + \
              (pt["mm K=16 (B-GEMM write)"] + pt["scalar mul"] + pt["add"] + pt["add"])
    fwd_new = qt["fused up"] + pt["fused up + residual"]
    bwd_cur = qt["scalar mul"] + pt["scalar mul"] + 2 * (pt["mm K=16 (A dgrad write)"] + pt["masked_scale (dropout bwd)"] + pt["add"])
    bwd_new = 2 * pt["fused dgrad-acc"]
    print(f"  kernel sums per layer: forward {fwd_cur:.3f} -> {fwd_new:.3f} ms, backward {bwd_cur:.3f} -> {bwd_new:.3f} ms;"
          f" per step (x{N_FWD} fwd, x{N_BWD} bwd): {(fwd_cur - fwd_new) * N_FWD + (bwd_cur - bwd_new) * N_BWD:.1f} ms"
          f" of {STEP_MS:.0f}")

    print("\n attention block forward + backward per layer (ms, CUDA events, 20 iters), production shape:")
    attn, step = _attention_block()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    res = {}
    for rep in range(2):  # two interleaved repeats, best of both (run-to-run noise is ~0.05 ms per layer)
        for path in ("nemo", "plain", "add", "add+res", "bwd", "add+bwd", "add+res+bwd"):
            f = b = 0.0
            for _ in range(3):
                step(path, ev)
            torch.cuda.synchronize()
            n = 20
            for _ in range(n):
                step(path, ev)
                torch.cuda.synchronize()
                f += ev[0].elapsed_time(ev[1])
                b += ev[1].elapsed_time(ev[2])
            prev = res.get(path, (1e9, 1e9))
            res[path] = (min(prev[0], f / n), min(prev[1], b / n))
    f0, b0 = res["nemo"]
    print(f"  {'config':14s} {'fwd':>8s} {'bwd':>8s} {'fwd+bwd':>8s}   {'d fwd':>7s} {'d bwd':>7s}   step est. (ms)")
    for path, (f, b) in res.items():
        dstep = (f0 - f) * N_FWD + (b0 - b) * N_BWD
        print(f"  {path:14s} {f:8.3f} {b:8.3f} {f + b:8.3f}   {f0 - f:7.3f} {b0 - b:7.3f}   {dstep:7.1f}"
              f"  ({100 * dstep / STEP_MS:4.1f}% of {STEP_MS:.0f})")
    print("  (fwd column x 91 layer-forwards incl. 11 recomputes, bwd x 80; 'plain' = patched functions, all items off)")


if __name__ == "__main__":
    tests = [test_up_kernel, test_dgrad_kernel, test_blas_variant_and_te_accumulate, test_lora_apply_vs_nemo_adapter,
             test_te_layernorm_linear_fold, test_te_linear_fold_with_residual, test_attention_block, test_wiring]
    if "--bench-only" in sys.argv:
        tests = []
    if "--wiring-only" in sys.argv:  # CPU-only import/wiring check (no GPU needed)
        tests = [test_wiring]
    if "--bench" in sys.argv or "--bench-only" in sys.argv:
        tests.append(test_benchmark)
    for t in tests:
        print(f"\n== {t.__name__}")
        t()
        print(f"   {t.__name__}: OK")
    print("\nALL OK")
