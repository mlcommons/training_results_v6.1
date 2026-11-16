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

"""Tests + microbenchmark for fix 6b (hs_transpose_cache.py, HS_TCACHE).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_transpose_cache.py [--bench]

1. Classification: the attention block (NeMo LoRA mixin, FP8 primary weights) caches exactly its qkv and proj base
   projections; the LoRA adapter linears and the MLP's fc1 / fc2 are classified correctly.
2. Megatron TE MLP (fc1 = LayerNormLinear, fc2 = Linear, FP8 primary weights, recipe patches): 3 fwd+bwd steps with
   the cache off (keep_fp8_weight_transpose_cache False, today's path) then on, from the same FP8 state. Outputs,
   input gradients and FP8 state bit-identical; with the cache warm, no weight transpose kernel runs.
3. --bench: MLP fwd+bwd, cache off vs on.
"""

import os
import sys

for _k, _v in {"USE_TE_SWIGLU": "1", "HS_ACT_STORE_FUSED": "1", "HS_SWIGLU_BWD_FP8": "1", "HS_SWIGLU_FWD_FP8": "1",
               "HS_GRAD_CAST_ONLY": "1", "HS_TCACHE": "qkv,proj,fc1,fc2"}.items():
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import hs_transpose_cache as tc  # noqa: E402  (before any TE module is built)
import hs_act_store as hs  # noqa: E402
import test_hs_lora_epilogue as T  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402

DEV = "cuda"
M = 8192


def _te_modules(root):
    return [(n, m) for n, m in root.named_modules() if isinstance(m, (te.Linear, te.LayerNormLinear))]


def test_classification():
    attn, _ = T._attention_block()
    rows = [(n, tc._kind(m), m.keep_fp8_weight_transpose_cache, getattr(m, "primary_weights_in_fp8", False))
            for n, m in _te_modules(attn)]
    for r in rows:
        print(f"  attention {r[0]:40s} kind {r[1]:5s} cache {r[2]}  fp8 primary {r[3]}")
    cached = sorted(n for n, k, c, _ in rows if c)
    assert any("linear_qkv" in n for n in cached) and any("linear_proj" in n for n in cached), cached
    assert all(("linear_qkv" in n or "linear_proj" in n) for n in cached), cached
    print(f"PASS classification: attention caches {cached}")


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
    kinds = {tc._kind(m): m.keep_fp8_weight_transpose_cache for _, m in _te_modules(mlp)}
    assert kinds == {"fc1": True, "fc2": True}, kinds
    return mlp


_MLP = {}


def _mlp():
    if not _MLP:
        _MLP["mlp"] = _build_mlp()
    return _MLP["mlp"]


def _set_cache(mlp, on):
    for _, m in _te_modules(mlp):
        m.keep_fp8_weight_transpose_cache = on
        # start each phase from a weight with no transpose, as built (TE's per-backward clear leaves an empty
        # tensor in _transpose that _create_transpose cannot write into; a module cached from init never has one)
        w = m.weight
        w._transpose, w._transpose_invalid = None, True


def _run_steps(mlp, steps, seed=7):
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
    return outs, grads, [t.clone() for t, _ in T._fp8_state(mlp)]


def _neq(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


def _count_transposes(fn):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return len([e.name for e in prof.events() if "transpose" in e.name.lower() and "cast" not in e.name.lower()])


def test_mlp_bit_identical_and_engaged():
    mlp = _mlp()
    _set_cache(mlp, False)
    _run_steps(mlp, 1, seed=1)                 # FP8-initialise
    saved = T._fp8_state(mlp)
    o0, g0, s0 = _run_steps(mlp, 3)
    n_off = _count_transposes(lambda: _run_steps(mlp, 1, seed=9))
    T._restore_fp8(saved)
    _set_cache(mlp, True)
    o1, g1, s1 = _run_steps(mlp, 3)
    n_on = _count_transposes(lambda: _run_steps(mlp, 1, seed=9))
    for i in range(3):
        no, ng = _neq(o0[i], o1[i]), _neq(g0[i], g1[i])
        print(f"  step {i}: output differing {no}, input grad differing {ng} of {o0[i].numel()}")
        assert no == 0 and ng == 0
    for i, (a, b) in enumerate(zip(s0, s1)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"FP8 state tensor {i} differs"
    print(f"  weight transpose kernels per fwd+bwd: off {n_off}, on (warm) {n_on}")
    assert n_off >= 2 and n_on == 0, (n_off, n_on)
    print("PASS MLP: 3 steps bit-identical with the cache on; no weight transpose once warm")
    T._restore_fp8(saved)


def test_benchmark(iters=20, warmup=3):
    mlp = _mlp()
    saved = T._fp8_state(mlp)
    x = torch.randn(M, 1, T.H, device=DEV).to(torch.bfloat16).requires_grad_()
    g = (torch.randn(M, 1, T.H, device=DEV) * 0.01).to(torch.bfloat16)
    res = {}
    for rnd in range(2):
        for flag in (False, True):
            _set_cache(mlp, flag)
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
        print(f"    transpose cache {'on ' if flag else 'off'}: fwd {f:.3f}  bwd {b:.3f}  total {f + b:.3f}")
    t0 = min(sum(r) for r in res[False]); t1 = min(sum(r) for r in res[True])
    print(f"  fc1 + fc2 saving {1000 * (t0 - t1):.1f} us per layer (isolated)")


if __name__ == "__main__":
    tests = [test_classification, test_mlp_bit_identical_and_engaged]
    if "--bench" in sys.argv:
        tests.append(test_benchmark)
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
