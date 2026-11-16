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

"""Tests + microbenchmark for fix 8a (hs_qkvnorm_e4m3.py, HS_QKVNORM_E4M3=1; needs HS_RMSNORM_GRID=1).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_qkvnorm_e4m3.py [--bench]

1. Kernel exactness at [8192, 8192]: the BF16 ln_out and rsigma equal fix 13's pinned TE RMSNorm forward; the
   e4m3 bytes / amax / scale_inv equal TE's Float8Quantizer(E4M3, rowwise) applied to that ln_out. Several
   magnitudes, heavy tails, NaN/inf in the input, saturating scales, a pre-filled amax.
2. A TE LayerNormLinear as the recipe's linear_qkv (8192 -> 10240, RMSNorm, return_layernorm_output, FP8 primary
   frozen weight): 3 forwards + backwards with the fix off then on from the same FP8 state: out, ln_out, input
   grad and FP8 state bit-identical; the e4m3 unary_kernel cast gone, the dual kernel engaged.
3. The two-TransformerLayer check of test_hs_resgrad_fold with fixes 8b, 8c and 8a all on: bit-identical.
4. --bench: the linear_qkv forward, off vs on.
"""

import os
import sys

for _k, _v in {"HS_RMSNORM_GRID": "1", "HS_QKVNORM_E4M3": "1", "HS_GRAD_CAST_ONLY": "1",
               "NVTE_USE_RMSNORM_TRITON": "1", "NVTE_USE_CAST_TRANSPOSE_TRITON": "1"}.items():
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import test_hs_resgrad_fold as RF  # noqa: E402  (recipe env, patch stack, 2-layer helpers)
import hs_qkvnorm_e4m3 as qn  # noqa: E402
import hs_rmsnorm_grid as grid  # noqa: E402
import hs_resgrad_fold as rf  # noqa: E402
import test_hs_lora_epilogue as T  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402
import transformer_engine_torch as tex  # noqa: E402
from transformer_engine.pytorch.module import _common as _common  # noqa: E402
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer  # noqa: E402

DEV = "cuda"
M, N = 8192, 8192


def _inputs(seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*shape, std=1.0):
        return torch.randn(*shape, generator=gen, device=DEV) * std

    yield "x~N(0,1)", rn(M, N)
    yield "x~N(0,30)", rn(M, N, std=30)
    yield "x~N(0,1e-3)", rn(M, N, std=1e-3)
    u = torch.rand(M, N, generator=gen, device=DEV).clamp_(1e-6, 1 - 1e-6)
    yield "x heavy-tailed", torch.tan(torch.pi * (u - 0.5)).clamp_(-3e4, 3e4)
    x = rn(M, N)
    x.view(-1)[::9973] = float("nan")
    x.view(-1)[5::10007] = float("inf")
    yield "x with NaN/inf", x


def test_kernel_exact():
    gen = torch.Generator(device=DEV).manual_seed(3)
    gamma = (1.0 + 0.1 * torch.randn(N, generator=gen, device=DEV)).to(torch.bfloat16)
    n_cases = 0
    for name, x32 in _inputs(77):
        x = x32.to(torch.bfloat16)
        ref_out, _, ref_rs = _common.te_rmsnorm_fwd_triton(x, gamma, 1e-5, None, None, torch.bfloat16, 0, False)
        ref_amax = ref_out.float().abs().nan_to_num(nan=0.0, posinf=0.0).amax().item()
        dyn = 240.0 / ref_amax if ref_amax not in (0.0, float("inf")) else 1.0
        for scale_val in (dyn, 2.0 ** 6, dyn * 64.0, 1.0 / 256):
            for amax0 in (0.0, 1e30):
                scale = torch.tensor([scale_val], dtype=torch.float32, device=DEV)
                amax_r = torch.tensor([amax0], dtype=torch.float32, device=DEV)
                q_r = Float8Quantizer(scale=scale, amax=amax_r, fp8_dtype=tex.DType.kFloat8E4M3)
                q_r.set_usage(rowwise=True, columnwise=False)
                t = q_r(ref_out)
                amax_c = torch.tensor([amax0], dtype=torch.float32, device=DEV)
                q_c = Float8Quantizer(scale=scale.clone(), amax=amax_c, fp8_dtype=tex.DType.kFloat8E4M3)
                ln_out = torch.empty_like(x)
                out, rs, f8, sinv = qn.rmsnorm_fwd_dual(x, gamma, 1e-5, ln_out, q_c, False, grid.GRID_BF16)
                n_out = RF._bits_equal(out, ref_out)
                n_rs = RF._bits_equal(rs, ref_rs)
                nd = RF._bytes_equal(f8, t._data)
                same_amax = torch.equal(amax_c.view(torch.int32), amax_r.view(torch.int32))
                same_sinv = torch.equal(sinv.view(torch.int32), t._scale_inv.view(torch.int32))
                print(f"  {name:16s} scale {scale_val:12.5g} amax0 {amax0:7.1g}: ln_out diff {n_out}, rsigma diff "
                      f"{n_rs}, e4m3 diff bytes {nd}, amax {'=' if same_amax else '!='} ({amax_c.item():.6g} vs "
                      f"{amax_r.item():.6g}), scale_inv {'=' if same_sinv else '!='}")
                assert n_out == 0 and n_rs == 0 and nd == 0 and same_amax and same_sinv, name
                n_cases += 1
    print(f"PASS kernel exactness: {n_cases} cases at [{M}, {N}], 0 differing elements / bytes")


_QKV = {}


def _qkv():
    if not _QKV:
        with te.fp8_model_init(enabled=True, recipe=T.RECIPE):
            lin = te.LayerNormLinear(T.H, T.QKV, eps=1e-5, bias=False, normalization="RMSNorm",
                                     return_layernorm_output=True, params_dtype=torch.bfloat16, device=DEV)
        for p in lin.parameters():
            p.requires_grad_(False)
        lin.train()
        _QKV["lin"] = lin
    return _QKV["lin"]


def _run(lin, steps, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    outs, lns, grads = [], [], []
    for _ in range(steps):
        x = torch.randn(M, 1, T.H, generator=gen, device=DEV).to(torch.bfloat16).requires_grad_()
        g = (torch.randn(M, 1, T.QKV, generator=gen, device=DEV) * 0.01).to(torch.bfloat16)
        with te.fp8_autocast(enabled=True, fp8_recipe=T.RECIPE):
            out, ln = lin(x)
        (out + ln.sum() * 0.0).backward(g)   # keep ln in the graph as the LoRA would
        outs.append(out.detach().clone()); lns.append(ln.detach().clone()); grads.append(x.grad.detach().clone())
    return outs, lns, grads, [t.clone() for t, _ in T._fp8_state(lin)]


def _counts(fn):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    c = {"e4m3_cast": 0, "dual": 0, "te_norm_fwd": 0}
    for e in p.key_averages():
        if e.self_device_time_total <= 0:
            continue
        n = e.key
        if "unary_kernel" in n and "e4m3" in n:
            c["e4m3_cast"] += e.count
        elif "_rmsnorm_fwd_dual_kernel" in n:
            c["dual"] += e.count
        elif "_rmsnorm_fwd_triton" in n:
            c["te_norm_fwd"] += e.count
    return c


def test_linear_qkv_bit_identical_and_engaged():
    lin = _qkv()
    qn.set_enabled(False)
    _run(lin, 1, seed=1)
    saved = T._fp8_state(lin)
    o0, l0, g0, s0 = _run(lin, 3, seed=2)
    T._restore_fp8(saved)
    c_off = _counts(lambda: _run(lin, 1, seed=9))
    T._restore_fp8(saved)
    qn.set_enabled(True)
    st = dict(qn.STATS)
    o1, l1, g1, s1 = _run(lin, 3, seed=2)
    d = {k: qn.STATS[k] - st[k] for k in qn.STATS}
    T._restore_fp8(saved)
    c_on = _counts(lambda: _run(lin, 1, seed=9))
    T._restore_fp8(saved)
    for i in range(3):
        no, nl, ng = RF._bits_equal(o0[i], o1[i]), RF._bits_equal(l0[i], l1[i]), RF._bits_equal(g0[i], g1[i])
        print(f"  step {i}: out differing {no}, ln_out differing {nl}, input grad differing {ng}")
        assert no == 0 and nl == 0 and ng == 0
    for i, (a, b) in enumerate(zip(s0, s1)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"FP8 state tensor {i} differs"
    print(f"  stats for 3 forwards: {d}; kernels per fwd+bwd: off {c_off}, on {c_on}")
    assert d["dual"] == 3 and d["handed"] == 3 and d["skipped"] == 0, d
    assert c_on["dual"] == 1 and c_off["dual"] == 0 and c_off["e4m3_cast"] - c_on["e4m3_cast"] == 1, (c_off, c_on)
    assert c_off["te_norm_fwd"] - c_on["te_norm_fwd"] == 1, (c_off, c_on)
    print("PASS linear_qkv: 3 steps bit-identical with fix 8a; one e4m3 cast fewer, dual kernel engaged")


def test_layers_all_on():
    RF._layers()
    rf.set_enabled(False); rf.set_qkv_e5m2(False); qn.set_enabled(False)
    RF._step(seed=1)
    saved = RF._fp8_state()
    o0, g0, w0 = RF._step()
    RF._restore_fp8(saved) if hasattr(RF, "_restore_fp8") else T._restore_fp8(saved)
    rf.set_enabled(True); rf.set_qkv_e5m2(True); qn.set_enabled(True)
    st = dict(qn.STATS)
    o1, g1, w1 = RF._step()
    d = {k: qn.STATS[k] - st[k] for k in qn.STATS}
    T._restore_fp8(saved)
    no, ng = RF._bits_equal(o0, o1), RF._bits_equal(g0, g1)
    nw = [RF._bits_equal(a, b) for a, b in zip(w0, w1)]
    print(f"  2 layers, fixes 8b + 8c + 8a: stats {d}; output differing {no}, input grad differing {ng}, "
          f"LoRA grads differing {nw}")
    assert no == 0 and ng == 0 and all(n == 0 for n in nw)
    assert d["dual"] == 2 and d["handed"] == 2 and d["skipped"] == 0, d
    print("PASS 2 layers with 8b + 8c + 8a: bit-identical; both qkv norms fused")


def test_benchmark(iters=20, warmup=3):
    lin = _qkv()
    saved = T._fp8_state(lin)
    x = torch.randn(M, 1, T.H, device=DEV).to(torch.bfloat16).requires_grad_()
    res = {}
    for rnd in range(2):
        for flag in (False, True):
            qn.set_enabled(flag)
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            tf = 0.0
            for i in range(warmup + iters):
                with te.fp8_autocast(enabled=True, fp8_recipe=T.RECIPE):
                    if i >= warmup:
                        ev[0].record()
                    out, ln = lin(x)
                    if i >= warmup:
                        ev[1].record()
                        ev[1].synchronize()
                        tf += ev[0].elapsed_time(ev[1])
            res.setdefault(flag, []).append(tf / iters)
            T._restore_fp8(saved)
    print(f"\n  linear_qkv forward [{M} tokens], ms per call (CUDA events, {iters} iters, 2 interleaved rounds):")
    for flag in (False, True):
        print(f"    fix 8a {'on ' if flag else 'off'}: {min(res[flag]):.3f}")
    print(f"  saving {1000 * (min(res[False]) - min(res[True])):.1f} us per layer forward (80 per step)")


if __name__ == "__main__":
    tests = [test_kernel_exact, test_linear_qkv_bit_identical_and_engaged, test_layers_all_on]
    if "--bench" in sys.argv:
        tests.append(test_benchmark)
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
