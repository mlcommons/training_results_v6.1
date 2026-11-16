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

"""Tests + microbenchmark for fix 3b (hs_swiglu_fwd_e4m3.py, wired through hs_act_store.py HS_SWIGLU_FWD_FP8=1).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_swiglu_fwd_e4m3.py [--bench | --bench-only]

1. Kernel exactness: the act-store bytes, the e4m3 fc2-input bytes, amax and scale_inv of the fused kernel equal fix
   #1's swiglu_fwd_store (act-store + BF16 out) followed by TE's Float8Quantizer(scale, amax, kFloat8E4M3), rowwise
   only (what fc2's _Linear.forward runs). Exhaustive over all 65,536 BF16 patterns in y1 and in y2, plus the
   production shape [8192, 57344] at several magnitudes, heavy tails and special values; delayed-scaling, power-of-two,
   saturating and tiny scales; fresh and pre-filled amax. NaN-vs-NaN bytes count as equal (FNUZ NaN is one code).
2. Megatron TE MLP (FP8 frozen weights, delayed scaling HYBRID, fp8 act-store, fused SwiGLU, fix 3a on): 3 fwd+bwd
   steps with fix 3b off, then on from the same FP8 state. Outputs, input gradients, amax histories and scales must
   be bit-identical; the fused path must engage 3/3 and one e4m3 unary_kernel cast per forward must disappear.
   A freshly built MLP's first forward falls back (fc2 not FP8-initialised yet) and still matches.
3. --bench: MLP forward, off vs on, CUDA events.
"""

import os
import sys

for _k, _v in {"USE_TE_SWIGLU": "1", "HS_ACT_STORE_FUSED": "1", "HS_SWIGLU_BWD_FP8": "1", "HS_SWIGLU_FWD_FP8": "1",
               "HS_GRAD_CAST_ONLY": "1"}.items():
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import hs_act_store as hs  # noqa: E402  (must precede megatron's mlp import)
import test_hs_lora_epilogue as T  # noqa: E402  (recipe env, Megatron config, FP8 recipe, fp8 state helpers)
import transformer_engine.pytorch as te  # noqa: E402
import transformer_engine_torch as tex  # noqa: E402
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer  # noqa: E402

import hs_swiglu_fp8_triton as sw  # noqa: E402
from hs_swiglu_fwd_e4m3 import swiglu_fwd_e4m3_into  # noqa: E402

DEV = "cuda"
M, H = 8192, 28672
FNUZ_NAN = 0x80


def _bytes_diff(a, b, nan_code=None):
    a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    diff = a != b
    if nan_code is not None:
        diff &= ~((a == nan_code) & (b == nan_code))
    return int(diff.sum().item())


def _reference(x, scale_val, amax0):
    out, q = sw.swiglu_fwd_store(x)
    scale = torch.tensor([scale_val], dtype=torch.float32, device=DEV)
    amax = torch.tensor([amax0], dtype=torch.float32, device=DEV)
    qz = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E4M3)
    qz.set_usage(rowwise=True, columnwise=False)
    t = qz(out)
    return q.view(torch.uint8), t._data, amax, t._scale_inv


def _candidate(x, scale_val, amax0):
    m, h2 = x.shape
    q = torch.empty((m, h2), dtype=torch.uint8, device=DEV)
    f8 = torch.empty((m, h2 // 2), dtype=torch.uint8, device=DEV)
    scale = torch.tensor([scale_val], dtype=torch.float32, device=DEV)
    amax = torch.tensor([amax0], dtype=torch.float32, device=DEV)
    sinv = torch.empty(1, dtype=torch.float32, device=DEV)
    swiglu_fwd_e4m3_into(x, q, f8, scale, amax, sinv)
    return q, f8, amax, sinv


def _all_bf16():
    return torch.arange(-32768, 32768, dtype=torch.int32, device=DEV).to(torch.int16).view(torch.bfloat16)


def _inputs():
    gen = torch.Generator(device=DEV).manual_seed(4321)
    # exhaustive: every BF16 pattern as y1 (rows 0-3, y2 random / 1 / -1 / heavy) and as y2 (rows 4-7)
    allp = _all_bf16()
    other = [torch.randn(65536, generator=gen, device=DEV), torch.ones(65536, device=DEV),
             -torch.ones(65536, device=DEV), torch.randn(65536, generator=gen, device=DEV) * 300]
    rows = [torch.cat([allp, o.to(torch.bfloat16)]) for o in other]
    rows += [torch.cat([o.to(torch.bfloat16), allp]) for o in other]
    yield "exhaustive BF16 y1 / y2", torch.stack(rows).contiguous()
    for s in (0.25, 2.0, 16.0):
        yield f"N(0,{s}) [{M},{2 * H}]", (torch.randn(M, 2 * H, generator=gen, device=DEV) * s).to(torch.bfloat16)
    u = torch.rand(M, 2 * H, generator=gen, device=DEV).clamp_(1e-6, 1 - 1e-6)
    yield "heavy tail", torch.tan(torch.pi * (u - 0.5)).clamp_(-3e38, 3e38).to(torch.bfloat16)
    x = torch.randn(M, 2 * H, generator=gen, device=DEV)
    f = x.view(-1)
    f[0::997] = 1e-40; f[1::1009] = 3e38; f[2::1013] = -0.0; f[3::1019] = float("inf"); f[4::1021] = -float("inf")
    f[5::1031] = float("nan"); f[6::1033] = 470.0; f[7::1039] = -500.0
    yield "special values", x.to(torch.bfloat16)


def test_kernel_exact():
    n = 0
    for name, x in _inputs():
        out, _ = sw.swiglu_fwd_store(x)
        a = out.float().abs().nan_to_num(nan=0.0, posinf=0.0).amax().item()
        dyn = 240.0 / a if a > 0 else 1.0
        for scale_val in (dyn, 2.0 ** 4, dyn * 64.0, 1.0 / 1024):
            for amax0 in (0.0, 1e30):
                r = _reference(x, scale_val, amax0)
                c = _candidate(x, scale_val, amax0)
                nq = _bytes_diff(c[0], r[0])                 # OCP act-store: NaN codes 0x7f/0xff must match exactly
                nf = _bytes_diff(c[1], r[1], FNUZ_NAN)
                same_amax = torch.equal(c[2].view(torch.int32), r[2].view(torch.int32))
                same_sinv = torch.equal(c[3].view(torch.int32), r[3].view(torch.int32))
                print(f"  {name:26s} scale {scale_val:11.4g} amax0 {amax0:7.1g}: act-store diff {nq}, fc2-input diff "
                      f"{nf}, amax {'=' if same_amax else '!='} ({c[2].item():.6g} vs {r[2].item():.6g}), "
                      f"scale_inv {'=' if same_sinv else '!='}")
                assert nq == 0 and nf == 0 and same_amax and same_sinv, name
                n += 1
    print(f"PASS kernel exactness: {n} cases, 0 differing bytes, amax and scale_inv equal")


# ---------------------------------------------------------------------------------------------------- MLP level


def _build_mlp():
    T._init_parallel()
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import build_module
    cfg = T._config()
    cfg.activation_func_fp8_input_store = True
    spec = get_gpt_layer_with_transformer_engine_spec().submodules.mlp
    with te.fp8_model_init(enabled=True, recipe=T.RECIPE):
        mlp = build_module(spec, config=cfg)
    for p in mlp.parameters():
        p.requires_grad_(False)
    mlp.train()
    return mlp


_MLP = {}


def _mlp():
    if not _MLP:
        _MLP["mlp"] = _build_mlp()
    return _MLP["mlp"]


def _run_steps(mlp, steps, seed=7):
    """`steps` forward+backward passes; returns outputs, input grads and the final FP8 state."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    outs, grads = [], []
    for _ in range(steps):
        x = (torch.randn(M, 1, T.H, generator=gen, device=DEV)).to(torch.bfloat16).requires_grad_()
        g = (torch.randn(M, 1, T.H, generator=gen, device=DEV) * 0.01).to(torch.bfloat16)
        with te.fp8_autocast(enabled=True, fp8_recipe=T.RECIPE):
            out, _ = mlp(x)
        out.backward(g)
        outs.append(out.detach().clone())
        grads.append(x.grad.detach().clone())
    state = [t.clone() for t, _ in T._fp8_state(mlp)]
    return outs, grads, state


def _neq(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


def test_mlp_bit_identical():
    mlp = _mlp()
    hs.FWD_FP8 = False
    _run_steps(mlp, 1, seed=1)                  # FP8-initialise fc1/fc2 (first forward), then snapshot the state
    saved = T._fp8_state(mlp)
    s0 = dict(hs.STATS)
    o_off, g_off, st_off = _run_steps(mlp, 3)
    assert hs.STATS["fp8_fwd"] == s0["fp8_fwd"], "e4m3 forward ran while off"
    T._restore_fp8(saved)
    hs.FWD_FP8 = True
    s1 = dict(hs.STATS)
    o_on, g_on, st_on = _run_steps(mlp, 3)
    n_fp8 = hs.STATS["fp8_fwd"] - s1["fp8_fwd"]
    assert n_fp8 == 3, f"e4m3 forward engaged {n_fp8} of 3 times (stats {hs.STATS})"
    for i in range(3):
        no, ng = _neq(o_off[i], o_on[i]), _neq(g_off[i], g_on[i])
        print(f"  step {i}: output differing {no}, input grad differing {ng} of {o_off[i].numel()}")
        assert no == 0 and ng == 0
    for i, (a, b) in enumerate(zip(st_off, st_on)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"FP8 state tensor {i} differs"
    print(f"PASS MLP: 3 steps bit-identical (outputs, input grads, {len(st_on)} FP8 state tensors), "
          f"e4m3 forward engaged 3/3 (fix 3a backward engaged too: {hs.STATS['fp8_bwd'] - s1['fp8_bwd']}/3)")
    T._restore_fp8(saved)


def test_first_forward_fallback():
    hs.FWD_FP8 = True
    fresh = _build_mlp()
    s0 = dict(hs.STATS)
    _run_steps(fresh, 1, seed=3)
    fb = hs.STATS["bf16_fwd"] - s0["bf16_fwd"]
    on = hs.STATS["fp8_fwd"] - s0["fp8_fwd"]
    print(f"  fresh MLP, first forward: fallback {fb}, fused {on}")
    assert fb == 1 and on == 0
    s1 = dict(hs.STATS)
    _run_steps(fresh, 1, seed=4)
    assert hs.STATS["fp8_fwd"] - s1["fp8_fwd"] == 1, "second forward did not engage"
    print("PASS first forward falls back (fc2 not FP8-initialised), the second engages")
    del fresh


def _count_e4m3_casts(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return len([e.name for e in prof.events() if "unary_kernel" in e.name and "e4m3" in e.name])


def test_mlp_engagement():
    mlp = _mlp()
    saved = T._fp8_state(mlp)
    counts = {}
    for flag in (False, True):
        hs.FWD_FP8 = flag
        counts[flag] = _count_e4m3_casts(lambda: _run_steps(mlp, 1))
        T._restore_fp8(saved)
    print(f"  e4m3 unary_kernel launches per fwd+bwd: off {counts[False]}, on {counts[True]}")
    assert counts[True] == counts[False] - 1, counts
    print("PASS engagement: the fc2 input cast is gone (one e4m3 cast fewer per MLP forward)")


# ---------------------------------------------------------------------------------------------------- benchmark


def test_benchmark(iters=20, warmup=3):
    mlp = _mlp()
    saved = T._fp8_state(mlp)
    x = torch.randn(M, 1, T.H, device=DEV).to(torch.bfloat16).requires_grad_()
    g = (torch.randn(M, 1, T.H, device=DEV) * 0.01).to(torch.bfloat16)
    res = {}
    for rnd in range(2):
        for flag in (False, True):
            hs.FWD_FP8 = flag
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            tf = tb = 0.0
            for i in range(warmup + iters):
                with te.fp8_autocast(enabled=True, fp8_recipe=T.RECIPE):
                    if i >= warmup:
                        ev[0].record()
                    out, _ = mlp(x)
                if i >= warmup:
                    ev[1].record()
                out.backward(g)
                if i >= warmup:
                    ev[2].record()
                    ev[2].synchronize()
                    tf += ev[0].elapsed_time(ev[1])
                    tb += ev[1].elapsed_time(ev[2])
                x.grad = None
            res.setdefault(flag, []).append((tf / iters, tb / iters))
            T._restore_fp8(saved)
    print(f"\n  MLP block [{M} tokens], ms per call (CUDA events, {iters} iters, 2 interleaved rounds):")
    for flag in (False, True):
        f = min(r[0] for r in res[flag]); b = min(r[1] for r in res[flag])
        print(f"    fix 3b {'on ' if flag else 'off'}: fwd {f:.3f}  bwd {b:.3f}  total {f + b:.3f}")
    f_off = min(r[0] for r in res[False]); f_on = min(r[0] for r in res[True])
    print(f"  forward saving {1000 * (f_off - f_on):.1f} us per layer; x 80 layers (ACL 0) = "
          f"{80 * (f_off - f_on):.1f} ms/step (isolated; screens decide)")


if __name__ == "__main__":
    tests = [test_kernel_exact, test_mlp_bit_identical, test_first_forward_fallback, test_mlp_engagement]
    if "--bench-only" in sys.argv:
        tests = [test_benchmark]
    elif "--bench" in sys.argv:
        tests.append(test_benchmark)
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
