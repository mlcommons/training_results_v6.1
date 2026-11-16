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

"""Tests + microbenchmark for fix 3a (hs_dswiglu_e5m2.py, wired through hs_act_store.py HS_SWIGLU_BWD_FP8=1).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_dswiglu_e5m2.py [--bench | --bench-only]

1. Kernel exactness at the production shape [8192, 57344]: the e5m2 bytes, amax and scale_inv of the fused kernel
   equal fix #1's BF16 dx passed through TE's Float8Quantizer(scale, amax, kFloat8E5M2), rowwise only (the recipe's
   HS_GRAD_CAST_ONLY=1 path, TE unary_kernel). Inputs: several grad magnitudes, heavy tails, NaN/inf, saturating
   scales, a pre-filled amax (max-accumulation semantics). NaN-vs-NaN bytes count as equal (FNUZ NaN is one code).
2. Megatron TE MLP (FP8 frozen weights, delayed scaling HYBRID, fp8 act-store, fused SwiGLU): 3 fwd+bwd steps with
   the e5m2 backward off, then on from the same FP8 state. Input gradients, amax histories and scales must be
   bit-identical; the fc1 grad cast kernel must disappear (one e5m2 unary_kernel fewer per backward).
3. --bench: MLP fwd+bwd and the backward alone, off vs on, CUDA events.
"""

import os
import sys

for _k, _v in {"USE_TE_SWIGLU": "1", "HS_ACT_STORE_FUSED": "1", "HS_SWIGLU_BWD_FP8": "1",
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
from hs_dswiglu_e5m2 import dswiglu_e5m2_into  # noqa: E402

DEV = "cuda"
M, H = 8192, 28672
FNUZ_NAN = 0x80


def _bytes_equal(a, b):
    a, b = a.view(torch.uint8), b.view(torch.uint8)
    diff = (a != b) & ~((a == FNUZ_NAN) & (b == FNUZ_NAN))
    return int(diff.sum().item())


def _reference(g, x_fp8, scale_val, amax0):
    """fix #1's BF16 dx through TE's quantizer, rowwise only (what the recipe runs today)."""
    dx = sw.dswiglu_fp8in(g, x_fp8)
    scale = torch.tensor([scale_val], dtype=torch.float32, device=DEV)
    amax = torch.tensor([amax0], dtype=torch.float32, device=DEV)
    q = Float8Quantizer(scale=scale, amax=amax, fp8_dtype=tex.DType.kFloat8E5M2)
    q.set_usage(rowwise=True, columnwise=False)
    t = q(dx)
    return t._data, amax, t._scale_inv


def _candidate(g, x_fp8, scale_val, amax0):
    data = torch.empty(x_fp8.shape, dtype=torch.uint8, device=DEV)
    scale = torch.tensor([scale_val], dtype=torch.float32, device=DEV)
    amax = torch.tensor([amax0], dtype=torch.float32, device=DEV)
    sinv = torch.empty(1, dtype=torch.float32, device=DEV)
    dswiglu_e5m2_into(g, x_fp8, data, scale, amax, sinv)
    return data, amax, sinv


def _inputs(seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    yield "g~N(0,0.1) x~N(0,2)", (torch.randn(M, H, generator=gen, device=DEV) * 0.1), \
        (torch.randn(M, 2 * H, generator=gen, device=DEV) * 2)
    yield "g~N(0,1e-3) x~N(0,4)", (torch.randn(M, H, generator=gen, device=DEV) * 1e-3), \
        (torch.randn(M, 2 * H, generator=gen, device=DEV) * 4)
    u = torch.rand(M, H, generator=gen, device=DEV).clamp_(1e-6, 1 - 1e-6)
    cauchy = torch.tan(torch.pi * (u - 0.5)).clamp_(-3e4, 3e4)
    yield "g heavy-tailed x~N(0,8)", cauchy, torch.randn(M, 2 * H, generator=gen, device=DEV) * 8
    g = torch.randn(M, H, generator=gen, device=DEV)
    g.view(-1)[::9973] = float("nan")
    g.view(-1)[5::10007] = float("inf")
    g.view(-1)[7::10009] = -float("inf")
    yield "g with NaN/inf x~N(0,1)", g, torch.randn(M, 2 * H, generator=gen, device=DEV)


def test_kernel_exact():
    n_cases = 0
    for name, g32, x32 in _inputs(1234):
        g = g32.to(torch.bfloat16)
        x_fp8 = x32.to(torch.float8_e4m3fn)
        ref_dx_amax = sw.dswiglu_fp8in(g, x_fp8).float().abs().nan_to_num(nan=0.0).amax().item()
        # scales: the delayed-scaling value for this data, a power of two, saturating, tiny; amax0: fresh and pre-filled
        dyn = 57344.0 / ref_dx_amax if ref_dx_amax not in (0.0, float("inf")) else 1.0
        for scale_val in (dyn, 2.0 ** 10, dyn * 64.0, 1.0 / 1024):
            for amax0 in (0.0, 1e30):
                r = _reference(g, x_fp8, scale_val, amax0)
                c = _candidate(g, x_fp8, scale_val, amax0)
                nd = _bytes_equal(c[0], r[0])
                same_amax = torch.equal(c[1].view(torch.int32), r[1].view(torch.int32))
                same_sinv = torch.equal(c[2].view(torch.int32), r[2].view(torch.int32))
                print(f"  {name:28s} scale {scale_val:12.5g} amax0 {amax0:7.1g}: diff bytes {nd}, "
                      f"amax {'=' if same_amax else '!='} ({c[1].item():.6g} vs {r[1].item():.6g}), "
                      f"scale_inv {'=' if same_sinv else '!='}")
                assert nd == 0 and same_amax and same_sinv, name
                n_cases += 1
    print(f"PASS kernel exactness: {n_cases} cases at [{M}, {2 * H}], 0 differing bytes, amax and scale_inv equal")


# ---------------------------------------------------------------------------------------------------- MLP level


_MLP = {}


def _mlp():
    if _MLP:
        return _MLP["mlp"]
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
    _MLP["mlp"] = mlp
    return mlp


def _run_steps(mlp, steps, seed=7):
    """`steps` forward+backward passes; returns the input grads and the final FP8 state (amax history, scales)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    grads = []
    for _ in range(steps):
        x = (torch.randn(M, 1, T.H, generator=gen, device=DEV)).to(torch.bfloat16).requires_grad_()
        g = (torch.randn(M, 1, T.H, generator=gen, device=DEV) * 0.01).to(torch.bfloat16)
        with te.fp8_autocast(enabled=True, fp8_recipe=T.RECIPE):
            out, _ = mlp(x)
        out.backward(g)
        grads.append(x.grad.detach().clone())
    state = [t.clone() for t, _ in T._fp8_state(mlp)]
    return grads, state


def _count_e5m2_casts(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if "unary_kernel" in e.name and "e5m2" in e.name]
    return len(names)


def test_mlp_bit_identical():
    mlp = _mlp()
    saved = T._fp8_state(mlp)
    hs.BWD_FP8 = False
    s0 = dict(hs.STATS)
    g_off, st_off = _run_steps(mlp, 3)
    assert hs.STATS["fp8_bwd"] == s0["fp8_bwd"], "e5m2 path ran while off"
    T._restore_fp8(saved)
    hs.BWD_FP8 = True
    s1 = dict(hs.STATS)
    g_on, st_on = _run_steps(mlp, 3)
    n_fp8 = hs.STATS["fp8_bwd"] - s1["fp8_bwd"]
    assert n_fp8 == 3, f"e5m2 backward engaged {n_fp8} of 3 times (stats {hs.STATS})"
    for i, (a, b) in enumerate(zip(g_off, g_on)):
        nd = int((a.view(torch.int16) != b.view(torch.int16)).sum().item())
        print(f"  step {i}: input grad differing elements {nd} of {a.numel()}")
        assert nd == 0
    for i, (a, b) in enumerate(zip(st_off, st_on)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"FP8 state tensor {i} differs"
    print(f"PASS MLP: 3 steps bit-identical (input grads, {len(st_on)} FP8 state tensors), e5m2 backward engaged 3/3")
    T._restore_fp8(saved)


def test_mlp_engagement():
    mlp = _mlp()
    saved = T._fp8_state(mlp)
    counts = {}
    for flag in (False, True):
        hs.BWD_FP8 = flag
        counts[flag] = _count_e5m2_casts(lambda: _run_steps(mlp, 1))
        T._restore_fp8(saved)
    print(f"  e5m2 unary_kernel launches per fwd+bwd: off {counts[False]}, on {counts[True]}")
    assert counts[True] == counts[False] - 1, counts
    print("PASS engagement: the fc1 grad cast is gone (one e5m2 cast fewer per MLP backward)")


# ---------------------------------------------------------------------------------------------------- benchmark


def test_benchmark(iters=20, warmup=3):
    mlp = _mlp()
    saved = T._fp8_state(mlp)
    x = torch.randn(M, 1, T.H, device=DEV).to(torch.bfloat16).requires_grad_()
    g = (torch.randn(M, 1, T.H, device=DEV) * 0.01).to(torch.bfloat16)
    res = {}
    for rnd in range(2):                       # two interleaved rounds per setting (clock drift)
        for flag in (False, True):
            hs.BWD_FP8 = flag
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
        print(f"    e5m2 bwd {'on ' if flag else 'off'}: fwd {f:.3f}  bwd {b:.3f}  total {f + b:.3f}")
    b_off = min(r[1] for r in res[False]); b_on = min(r[1] for r in res[True])
    print(f"  backward saving {1000 * (b_off - b_on):.1f} us per layer; x 77 non-recomputed layers = "
          f"{77 * (b_off - b_on):.1f} ms/step (isolated; screens decide)")


if __name__ == "__main__":
    tests = [test_kernel_exact, test_mlp_bit_identical, test_mlp_engagement]
    if "--bench-only" in sys.argv:
        tests = [test_benchmark]
    elif "--bench" in sys.argv:
        tests.append(test_benchmark)
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
