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

"""Unit tests + microbenchmark for the fused FP8 act-store SwiGLU kernels (hs_swiglu_fp8_triton.py) and the
HS_ACT_STORE_FUSED=1 level of hs_act_store.py. Runs inside the recipe image on one GPU:

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs -m pytest -s -q /p/test_hs_swiglu_fp8.py        # or
      ... --entrypoint /opt/venv/bin/python hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_swiglu_fp8.py [--bench]

Checks: e4m3fn bytes bit-identical to torch's cast (all 65536 BF16 patterns, specials, random data incl. |x| > 448,
subnormals, -0.0); dequantize exact over the 256 codes; forward == te.ops.SwiGLU within BF16 rounding (bit-exact
in practice); backward == tex.dswiglu(g, x_fp8.to(bf16)) within BF16 rounding; the fused autograd.Function
against Megatron's level-1 SwiGLUFunction (the submission reference) on the production shape [8192, 57344]
and a small odd shape; the patch wiring. The benchmark (CUDA events, 20 iterations) compares per layer the
level-1 kernels (torch cast + TE gated_act, torch dequant + tex.dswiglu) with the fused kernels.
"""

import os
import sys

os.environ.setdefault("USE_TE_SWIGLU", "1")  # the recipe's setting: te.ops.SwiGLU forward, tex.dswiglu backward
os.environ["HS_ACT_STORE_FUSED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402  (loads the transformer_engine_torch extension)
import transformer_engine_torch as tex  # noqa: E402

import hs_swiglu_fp8_triton as fused  # noqa: E402

DEV = "cuda"
BF16, E4M3 = torch.bfloat16, torch.float8_e4m3fn
PROD_SHAPE = (8192, 57344)  # 8192 tokens x 2H, H = 28672 (Llama 2 70B ffn)
ODD_SHAPE = (37, 2 * 101)  # masks, non-multiple-of-block columns
N_LAYERS = 69  # kept (non-recomputed) layers per step in recipe A
STEP_MS = 3693.0


def _gen(seed=0):
    return torch.Generator(device=DEV).manual_seed(seed)


def _randn_bf16(shape, scale=1.0, seed=0):
    return (torch.randn(shape, generator=_gen(seed), device=DEV) * scale).to(BF16)


def _diff_stats(a, b):
    """max abs, max rel (over differing elements), fraction exactly equal (NaN == NaN counts as equal)."""
    a, b = a.float(), b.float()
    both_nan = a.isnan() & b.isnan()
    d = (a - b).abs()
    d = torch.where(both_nan, torch.zeros_like(d), d)
    eq = (d == 0) | both_nan
    max_abs = d[~both_nan].max().item() if (~both_nan).any() else 0.0
    rel = d / b.abs().clamp_min(torch.finfo(torch.float32).tiny)
    max_rel = rel[(d > 0) & ~both_nan].max().item() if ((d > 0) & ~both_nan).any() else 0.0
    return max_abs, max_rel, eq.float().mean().item()


def _assert_bf16_close(out, ref, what):
    max_abs, max_rel, eq = _diff_stats(out, ref)
    print(f"  {what}: max abs {max_abs:.3e}, max rel {max_rel:.3e}, exactly equal {eq * 100:.5f}%")
    # within BF16 rounding: at most 1 BF16 ulp (2^-8 relative at the mantissa edge, 2^-7 worst case)
    assert max_rel <= 2.0**-7 + 1e-9, (what, max_abs, max_rel, eq)
    assert eq > 0.999, (what, eq)


# ----------------------------------------------------------------------------------------------- cast / dequant


def _check_cast(x, what):
    _, q = fused.swiglu_fwd_store(x)
    ref = x.to(E4M3)
    assert q.dtype == E4M3 and q.shape == x.shape
    same = q.view(torch.uint8) == ref.view(torch.uint8)
    n_bad = (~same).sum().item()
    if n_bad:
        idx = (~same).nonzero()[:8]
        print("  mismatches:", [(x[i, j].item(), hex(q.view(torch.uint8)[i, j].item()),
                                 hex(ref.view(torch.uint8)[i, j].item())) for i, j in idx.tolist()])
    print(f"  cast {what}: {x.numel()} elements, {n_bad} mismatching bytes")
    assert n_bad == 0, what


def test_cast_special_values():
    vals = [0.0, -0.0, 448.0, 456.0, 464.0, 465.0, 470.0, 480.0, 1000.0, 65504.0, float("inf"), -float("inf"),
            float("nan"), -float("nan"), -448.0, -464.0, -465.0,
            2.0**-9, 2.0**-10, 1.5 * 2.0**-9, 2.5 * 2.0**-9, 3.5 * 2.0**-9, 2.0**-7, 2.0**-6, 2.0**-6 - 2.0**-10,
            2.0**-130, -2.0**-133, 1.0, 1.0625, 1.1875, 1.3125, -1.0625, 3e-3, -0.015, 0.1, 255.0, 256.0, 257.0]
    vals += [0.0] * (-len(vals) % 4)
    _check_cast(torch.tensor(vals, dtype=BF16, device=DEV).view(2, -1), "special values")


def test_cast_all_bf16_patterns():
    x = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16).view(BF16).view(256, 256)
    _check_cast(x, "all 65536 bf16 bit patterns")


def test_cast_random():
    for scale in (100.0, 1.0, 0.01, 1e-6, 1e4):  # > 448 and NaN-producing, normal, subnormal, zero-rounding
        _check_cast(_randn_bf16((512, 2048), scale, seed=int(scale * 1000) % 7919), f"randn * {scale}")


def test_dequant_all_codes():
    import triton
    import triton.language as tl

    @triton.jit
    def _k(bp, op, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(op + offs, fused._e4m3fn_bits_to_fp32(tl.load(bp + offs)))

    codes = torch.arange(256, dtype=torch.uint8, device=DEV)
    out = torch.empty(256, dtype=torch.float32, device=DEV)
    _k[(1,)](codes, out, BLOCK=256)
    ref = codes.view(E4M3).float()
    ok = (out == ref) | (out.isnan() & ref.isnan())
    print(f"  dequant: {ok.sum().item()}/256 codes exact")
    assert ok.all(), (~ok).nonzero().flatten().tolist()


# ----------------------------------------------------------------------------------------------- kernels vs TE


def _check_fwd_bwd(shape, scale, seed):
    x = _randn_bf16(shape, scale, seed)
    g = _randn_bf16((shape[0], shape[1] // 2), 1.0, seed + 1)
    out, q = fused.swiglu_fwd_store(x)
    assert out.shape == (shape[0], shape[1] // 2) and out.dtype == BF16
    _assert_bf16_close(out, te.ops.SwiGLU()(x), f"fwd {shape} scale {scale} vs te.ops.SwiGLU")
    assert torch.equal(q.view(torch.uint8), x.to(E4M3).view(torch.uint8))
    dx = fused.dswiglu_fp8in(g, q)
    assert dx.shape == shape and dx.dtype == BF16
    _assert_bf16_close(dx, tex.dswiglu(g, q.to(BF16), None), f"bwd {shape} scale {scale} vs tex.dswiglu")


def test_kernels_prod_shape():
    _check_fwd_bwd(PROD_SHAPE, 1.0, seed=10)
    _check_fwd_bwd(PROD_SHAPE, 4.0, seed=11)


def test_kernels_odd_shape():
    _check_fwd_bwd(ODD_SHAPE, 1.0, seed=20)
    _check_fwd_bwd((5, 2 * 3), 20.0, seed=21)


def test_noncontiguous_grad():
    x = _randn_bf16(ODD_SHAPE, 1.0, 30)
    _, q = fused.swiglu_fwd_store(x)
    g = _randn_bf16((ODD_SHAPE[0], ODD_SHAPE[1]), 1.0, 31)[:, : ODD_SHAPE[1] // 2]  # strided view
    assert not g.is_contiguous()
    assert torch.equal(fused.dswiglu_fp8in(g, q), fused.dswiglu_fp8in(g.contiguous(), q))


# ------------------------------------------------------------------------------------- autograd + patch wiring


def _import_patch():
    import hs_act_store  # noqa: F401  (patches megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl)
    import megatron.core.fusions.fused_bias_swiglu as fbs

    assert fbs.use_te_swiglu, "set USE_TE_SWIGLU=1 (recipe default) before importing"
    return hs_act_store, fbs


def test_autograd_vs_megatron_level1():
    """Full fused autograd.Function against Megatron's SwiGLUFunction(fp8_input_store=True), the level-1 path
    used by the submission (forward: te.ops.SwiGLU under fp8_autocast; backward: tex.dswiglu of the
    e4m3-rounded input)."""
    hs, fbs = _import_patch()
    for shape, scale, seed in ((PROD_SHAPE, 1.0, 40), (ODD_SHAPE, 3.0, 41)):
        x = _randn_bf16(shape, scale, seed)
        g = _randn_bf16((shape[0], shape[1] // 2), 1.0, seed + 1)
        x_ref = x.clone().requires_grad_()
        out_ref = fbs.SwiGLUFunction.apply(x_ref, True)
        out_ref.backward(g)
        x_f = x.clone().requires_grad_()
        out_f = hs.SwiGLUFunctionFused.apply(x_f)
        assert "SwiGLUFunctionFused" in out_f.grad_fn.name()
        out_f.backward(g)
        _assert_bf16_close(out_f, out_ref, f"autograd fwd {shape} vs Megatron level-1")
        _assert_bf16_close(x_f.grad, x_ref.grad, f"autograd dx {shape} vs Megatron level-1")


def test_patch_wiring_and_fallbacks():
    hs, fbs = _import_patch()
    import megatron.core.transformer.mlp as mlp

    assert hs.USE_FUSED
    assert fbs.bias_swiglu_impl is hs.bias_swiglu_impl
    assert mlp.bias_swiglu_impl is hs.bias_swiglu_impl, "mlp must import bias_swiglu_impl after the patch"
    x = _randn_bf16(ODD_SHAPE, 1.0, 50)
    # grad enabled + store requested -> fused function (stored tensor is the fp8 view on ctx)
    out = hs.bias_swiglu_impl(x.clone().requires_grad_(), None, True)
    assert "SwiGLUFunctionFused" in out.grad_fn.name()
    fn = out.grad_fn
    assert fn.x_fp8.dtype == E4M3 and fn.x_fp8.shape == ODD_SHAPE
    out.backward(torch.ones_like(out))
    assert fn.x_fp8 is None, "stored tensor must be released after backward"
    # 3-D input is viewed as 2-D and back
    x3 = _randn_bf16((3, 5, ODD_SHAPE[1]), 1.0, 51)
    out3 = hs.bias_swiglu_impl(x3.clone().requires_grad_(), None, True)
    assert out3.shape == (3, 5, ODD_SHAPE[1] // 2)
    assert "SwiGLUFunctionFused" in out3.grad_fn.next_functions[0][0].name()  # behind the view node
    _assert_bf16_close(out3, te.ops.SwiGLU()(x3.view(-1, ODD_SHAPE[1])).view(3, 5, -1), "3-D input fwd")
    # store not requested -> Megatron's own function (BF16 saved)
    out = hs.bias_swiglu_impl(x.clone().requires_grad_(), None, False)
    assert "SwiGLUFunction" in out.grad_fn.name() and "Fused" not in out.grad_fn.name()
    # no-grad (checkpoint forward) -> no autograd graph, plain TE forward
    with torch.no_grad():
        out = hs.bias_swiglu_impl(x, None, True)
    assert out.grad_fn is None and torch.equal(out, te.ops.SwiGLU()(x))
    # recompute phase -> Megatron's function with BF16 store (input consumed at once)
    from transformer_engine.pytorch import distributed as te_dist

    te_dist._FP8_ACTIVATION_RECOMPUTE_PHASE = True  # what activation_recompute_forward(recompute_phase=True) sets
    try:
        assert te_dist.in_fp8_activation_recompute_phase()
        out = hs.bias_swiglu_impl(x.clone().requires_grad_(), None, True)
    finally:
        te_dist._FP8_ACTIVATION_RECOMPUTE_PHASE = False
    assert "Fused" not in out.grad_fn.name()
    print("  wiring: fused under grad, Megatron path under no-grad / recompute / store=False")


# ------------------------------------------------------------------------------------------------ benchmark


def _time_ms(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def test_benchmark():
    hs, fbs = _import_patch()
    M, H2 = PROD_SHAPE
    H = H2 // 2
    x = _randn_bf16(PROD_SHAPE, 1.0, 60)
    g = _randn_bf16((M, H), 1.0, 61)
    x_fp8 = x.to(E4M3)
    x_deq = x_fp8.to(BF16)
    swiglu_op = te.ops.SwiGLU()
    t = {
        "cast x.to(float8_e4m3fn)": _time_ms(lambda: x.to(E4M3)),
        "te.ops.SwiGLU forward (gated_act)": _time_ms(lambda: swiglu_op(x)),
        "dequant x_fp8.to(bf16)": _time_ms(lambda: x_fp8.to(BF16)),
        "tex.dswiglu (dgated_act)": _time_ms(lambda: tex.dswiglu(g, x_deq, None)),
        "fused swiglu_fwd_store": _time_ms(lambda: fused.swiglu_fwd_store(x)),
        "fused dswiglu_fp8in": _time_ms(lambda: fused.dswiglu_fp8in(g, x_fp8)),
    }
    # whole autograd paths (forward + backward, includes Python / autograd overhead)
    def _ref_path():
        xr = x.requires_grad_()
        fbs.SwiGLUFunction.apply(xr, True).backward(g)

    def _fused_path():
        xr = x.requires_grad_()
        hs.SwiGLUFunctionFused.apply(xr).backward(g)

    t["level-1 autograd fwd+bwd"] = _time_ms(_ref_path)
    t["fused autograd fwd+bwd"] = _time_ms(_fused_path)
    x.requires_grad_(False)
    x.grad = None

    gb = M * H2 * 1e-9  # elements of x in 1e9
    print(f"\n  microbenchmark {PROD_SHAPE}, 20 iterations, CUDA events (ms per call):")
    for k, v in t.items():
        print(f"    {k:40s} {v:7.3f} ms")
    ref_f = t["cast x.to(float8_e4m3fn)"] + t["te.ops.SwiGLU forward (gated_act)"]
    ref_b = t["dequant x_fp8.to(bf16)"] + t["tex.dswiglu (dgated_act)"]
    fus_f, fus_b = t["fused swiglu_fwd_store"], t["fused dswiglu_fp8in"]
    print(f"    forward  per layer: level-1 {ref_f:.3f} ms -> fused {fus_f:.3f} ms, saving {ref_f - fus_f:.3f} ms"
          f"  (fused traffic {gb * 4:.2f} GB -> {gb * 4 / fus_f:.2f} TB/s)")
    print(f"    backward per layer: level-1 {ref_b:.3f} ms -> fused {fus_b:.3f} ms, saving {ref_b - fus_b:.3f} ms"
          f"  (fused traffic {gb * 4:.2f} GB -> {gb * 4 / fus_b:.2f} TB/s)")
    per_layer = (ref_f - fus_f) + (ref_b - fus_b)
    per_step = per_layer * N_LAYERS
    print(f"    per layer fwd+bwd saving {per_layer:.3f} ms; per step x{N_LAYERS} layers: {per_step:.1f} ms "
          f"= {per_step / STEP_MS * 100:.2f}% of the {STEP_MS:.0f} ms step")
    e2e = t["level-1 autograd fwd+bwd"] - t["fused autograd fwd+bwd"]
    print(f"    autograd fwd+bwd saving {e2e:.3f} ms per layer, {e2e * N_LAYERS:.1f} ms per step")
    assert fus_f < ref_f and fus_b < ref_b


if __name__ == "__main__":
    tests = [test_cast_special_values, test_cast_all_bf16_patterns, test_cast_random, test_dequant_all_codes,
             test_kernels_prod_shape, test_kernels_odd_shape, test_noncontiguous_grad,
             test_autograd_vs_megatron_level1, test_patch_wiring_and_fallbacks]
    if "--bench" in sys.argv or "--bench-only" in sys.argv:
        tests = ([] if "--bench-only" in sys.argv else tests) + [test_benchmark]
    for fn in tests:
        print(f"{fn.__name__}:")
        fn()
        print("  ok")
    print("all passed")
