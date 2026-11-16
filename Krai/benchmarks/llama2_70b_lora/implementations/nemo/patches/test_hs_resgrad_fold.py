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

"""Tests + microbenchmark for fix 8b (hs_resgrad_fold.py, HS_RESGRAD_FOLD=1).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_resgrad_fold.py [--bench]

1. Kernel exactness at [8192, 8192]: (dx + res) in BF16, dgamma, and the e5m2 bytes / amax / scale_inv of the sum
   equal TE's te_rmsnorm_bwd_triton -> torch BF16 add -> TE Float8Quantizer(E5M2, rowwise). Inputs: several
   magnitudes, heavy tails, NaN/inf in the residual grad, saturating scales, a pre-filled amax.
2. Two Megatron TransformerLayers (TE spec, FP8 frozen weights, NeMo LoRA mixin on attention, the recipe's patch
   stack, unfused attention so the backward is deterministic): fwd+bwd with the fold off then on from the same
   FP8 state. Output, input gradient, the 8 LoRA weight
   gradients and the FP8 state must be bit-identical; kernel counts: 3 folds (fc1 norm of both layers, qkv norm
   of layer 1), 3 CUDAFunctor_add and 3 e5m2 unary_kernel launches fewer.
3. --bench: 2-layer fwd+bwd, off vs on.
"""

import os
import sys

if "--fused-attn" in sys.argv:   # timing only: the fused attention backward is nondeterministic (atomics)
    os.environ["NVTE_FUSED_ATTN"] = "1"
for _k, _v in {"USE_TE_SWIGLU": "1", "HS_ACT_STORE_FUSED": "1", "HS_SWIGLU_BWD_FP8": "1", "HS_SWIGLU_FWD_FP8": "1",
               "HS_GRAD_CAST_ONLY": "1", "HS_ROPE_FUSED": "1", "HS_LORA_EPILOGUE_FAST": "1",
               "NVTE_USE_RMSNORM_TRITON": "1", "NVTE_USE_CAST_TRANSPOSE_TRITON": "1", "HS_RMSNORM_GRID": "1",
               "HS_RESGRAD_FOLD": "1", "HS_QKVGRAD_E5M2": "1",
               # the fused attention backward accumulates with atomics (run-to-run nondeterministic, kb
               # training-nondeterministic-run-to-run); the unfused path makes the layer comparison exact
               "NVTE_FUSED_ATTN": "0"}.items():
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import hs_act_store as hs_as  # noqa: E402  (must precede megatron's mlp import)
import test_hs_lora_epilogue as T  # noqa: E402  (recipe env, Megatron config, FP8 recipe, fp8 state helpers)
import hs_rmsnorm_grid  # noqa: E402,F401
import hs_resgrad_fold as rf  # noqa: E402
import hs_rope_qkv as rq  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402
import transformer_engine_torch as tex  # noqa: E402
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer  # noqa: E402
from transformer_engine.pytorch.triton_kernels import rmsnorm as _rms  # noqa: E402

DEV = "cuda"
M, N = 8192, 8192
FNUZ_NAN = 0x80


def _bytes_equal(a, b):
    a, b = a.view(torch.uint8), b.view(torch.uint8)
    diff = (a != b) & ~((a == FNUZ_NAN) & (b == FNUZ_NAN))
    return int(diff.sum().item())


def _bits_equal(a, b):
    """BF16 / fp32 tensors equal bit for bit; NaN equals NaN (torch's BF16 rounding canonicalises NaN to 0x7FC0,
    Triton's conversion keeps the payload: both are NaN, the e5m2 byte and the amax are the same either way)."""
    ai = a.view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32)
    bi = b.view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32)
    diff = (ai != bi) & ~(torch.isnan(a) & torch.isnan(b))
    return int(diff.sum().item())


def _inputs(seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*shape, std=1.0):
        return torch.randn(*shape, generator=gen, device=DEV) * std

    yield "dz~N(0,0.01) res~N(0,0.01)", rn(M, N), rn(M, N, std=0.01), rn(M, N, std=0.01)
    yield "dz~N(0,1) res~N(0,3)", rn(M, N, std=4), rn(M, N), rn(M, N, std=3)
    u = torch.rand(M, N, generator=gen, device=DEV).clamp_(1e-6, 1 - 1e-6)
    cauchy = torch.tan(torch.pi * (u - 0.5)).clamp_(-3e4, 3e4)
    yield "res heavy-tailed", rn(M, N), rn(M, N, std=0.1), cauchy
    res = rn(M, N, std=0.1)
    res.view(-1)[::9973] = float("nan")
    res.view(-1)[5::10007] = float("inf")
    res.view(-1)[7::10009] = -float("inf")
    yield "res with NaN/inf", rn(M, N), rn(M, N, std=0.1), res


def test_kernel_exact():
    gen = torch.Generator(device=DEV).manual_seed(99)
    gamma = (1.0 + 0.1 * torch.randn(N, generator=gen, device=DEV)).to(torch.bfloat16)
    n_cases = 0
    for name, x32, dz32, res32 in _inputs(1234):
        x, dz, res = x32.to(torch.bfloat16), dz32.to(torch.bfloat16), res32.to(torch.bfloat16)
        _, _, rsigma = _rms.te_rmsnorm_fwd_triton(x, gamma, 1e-5, None, None, torch.bfloat16, 0, False,
                                                  autotune=False)
        ref_dx, ref_dg = _rms.te_rmsnorm_bwd_triton(dz, x, rsigma, gamma, 0, False)
        ref_sum = ref_dx + res
        ref_amax = ref_sum.float().abs().nan_to_num(nan=0.0).amax().item()
        dyn = 57344.0 / ref_amax if ref_amax not in (0.0, float("inf")) else 1.0
        for scale_val in (dyn, 2.0 ** 10, dyn * 64.0, 1.0 / 1024):
            for amax0 in (0.0, 1e30):
                scale = torch.tensor([scale_val], dtype=torch.float32, device=DEV)
                amax_r = torch.tensor([amax0], dtype=torch.float32, device=DEV)
                q_r = Float8Quantizer(scale=scale, amax=amax_r, fp8_dtype=tex.DType.kFloat8E5M2)
                q_r.set_usage(rowwise=True, columnwise=False)
                t = q_r(ref_sum)
                amax_c = torch.tensor([amax0], dtype=torch.float32, device=DEV)
                q_c = Float8Quantizer(scale=scale.clone(), amax=amax_c, fp8_dtype=tex.DType.kFloat8E5M2)
                out, dg, f8, sinv = rf.rmsnorm_bwd_resfold(dz, x, rsigma, gamma, 0, False, res, q_c)
                n_sum = _bits_equal(out, ref_sum)
                n_dg = _bits_equal(dg, ref_dg)
                nd = _bytes_equal(f8, t._data)
                same_amax = torch.equal(amax_c.view(torch.int32), amax_r.view(torch.int32))
                same_sinv = torch.equal(sinv.view(torch.int32), t._scale_inv.view(torch.int32))
                print(f"  {name:28s} scale {scale_val:12.5g} amax0 {amax0:7.1g}: sum diff {n_sum}, dgamma diff "
                      f"{n_dg}, e5m2 diff bytes {nd}, amax {'=' if same_amax else '!='} ({amax_c.item():.6g} vs "
                      f"{amax_r.item():.6g}), scale_inv {'=' if same_sinv else '!='}")
                assert n_sum == 0 and n_dg == 0 and nd == 0 and same_amax and same_sinv, name
                n_cases += 1
    print(f"PASS kernel exactness: {n_cases} cases at [{M}, {N}], 0 differing elements / bytes")


def test_rope_e5m2_exact():
    """Fix 8c kernel: the e5m2 bytes / amax / scale_inv written by the fused RoPE backward equal TE's quantizer on
    the BF16 d_mixed it writes, and that BF16 output is unchanged by the emission."""
    T._init_parallel()
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    freqs = RotaryEmbedding(kv_channels=128, rotary_percent=1.0)(M).contiguous()
    gen = torch.Generator(device=DEV).manual_seed(7)
    n_cases = 0
    for name, std in (("N(0,1)", 1.0), ("N(0,1e-3)", 1e-3), ("N(0,64)", 64.0), ("special", 1.0)):
        dq = (torch.randn(M, 1, 64, 128, generator=gen, device=DEV) * std).to(torch.bfloat16)
        dk = (torch.randn(M, 1, 8, 128, generator=gen, device=DEV) * std).to(torch.bfloat16)
        dv = (torch.randn(M, 1, 8, 128, generator=gen, device=DEV) * std).to(torch.bfloat16)
        if name == "special":
            dq.view(-1)[7::1039] = float("nan"); dk.view(-1)[8::1049] = float("inf"); dv.view(-1)[9::1051] = -0.0
        ref = rq.rope_qkv_bwd(dq, dk, dv, freqs)
        ref_amax = ref.float().abs().nan_to_num(nan=0.0).amax().item()
        dyn = 57344.0 / ref_amax if ref_amax not in (0.0, float("inf")) else 1.0
        for scale_val in (dyn, 2.0 ** 10, 1.0 / 1024):
            for amax0 in (0.0, 1e30):
                scale = torch.tensor([scale_val], dtype=torch.float32, device=DEV)
                amax_r = torch.tensor([amax0], dtype=torch.float32, device=DEV)
                q_r = Float8Quantizer(scale=scale, amax=amax_r, fp8_dtype=tex.DType.kFloat8E5M2)
                q_r.set_usage(rowwise=True, columnwise=False)
                t = q_r(ref)
                amax_c = torch.tensor([amax0], dtype=torch.float32, device=DEV)
                f8 = torch.empty(ref.shape, dtype=torch.uint8, device=DEV)
                sinv = torch.empty(1, dtype=torch.float32, device=DEV)
                out = rq.rope_qkv_bwd(dq, dk, dv, freqs, e5m2=(f8, scale, amax_c.view(torch.int32), sinv))
                n_out = _bits_equal(out, ref)
                nd = _bytes_equal(f8, t._data)
                same_amax = torch.equal(amax_c.view(torch.int32), amax_r.view(torch.int32))
                same_sinv = torch.equal(sinv.view(torch.int32), t._scale_inv.view(torch.int32))
                print(f"  {name:10s} scale {scale_val:12.5g} amax0 {amax0:7.1g}: d_mixed diff {n_out}, e5m2 diff "
                      f"bytes {nd}, amax {'=' if same_amax else '!='} ({amax_c.item():.6g} vs {amax_r.item():.6g}), "
                      f"scale_inv {'=' if same_sinv else '!='}")
                assert n_out == 0 and nd == 0 and same_amax and same_sinv, name
                n_cases += 1
    print(f"PASS RoPE backward e5m2 emission: {n_cases} cases at [{M}, 1, 10240], 0 differing elements / bytes")


# ------------------------------------------------------------------------------------------ 2 transformer layers

_LAYERS = {}


def _build_layers(n=2):
    T._init_parallel()
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import build_module
    from nemo.collections.nlp.modules.common.megatron.adapters.mcore_mixins import (
        MCoreSelfAttentionMixin, swap_mcore_mixin)
    from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
        AdapterName, LoraDenseAttentionAdapterConfig, LoraKQVAdapterConfig)

    cfg = T._config()
    cfg.activation_func_fp8_input_store = True
    spec = get_gpt_layer_with_transformer_engine_spec()
    layers = []
    with te.fp8_model_init(enabled=True, recipe=T.RECIPE):
        for i in range(n):
            layers.append(build_module(spec, config=cfg, layer_number=i + 1))
    lora_kw = dict(dim=T.R, norm_position=None, norm_type=None, activation="identity", column_init_method="kaiming",
                   row_init_method="zero", gather_output=False, dropout=T.P_DROP, alpha=T.ALPHA,
                   dropout_position="pre", a2a_experimental=False)
    mp_cfg = T._mp_config()
    params = []
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad_(False)
        attn = layer.self_attention
        swap_mcore_mixin(attn, MCoreSelfAttentionMixin)
        attn.add_adapter(name=AdapterName.LORA_KQV_ADAPTER,
                         cfg=LoraKQVAdapterConfig(in_features=T.H, out_features=T.QKV, **lora_kw),
                         model_parallel_config=mp_cfg)
        attn.add_adapter(name=AdapterName.LORA_DENSE_ATTENTION_ADAPTER,
                         cfg=LoraDenseAttentionAdapterConfig(in_features=T.H, out_features=T.H, **lora_kw),
                         model_parallel_config=mp_cfg)
        for j, ad in enumerate(attn.adapter_layer.values()):
            with torch.no_grad():
                ad.linear_out.weight.copy_(T._randn(ad.linear_out.weight.shape, 0.05, 50 + 10 * i + j))
            assert T.hs.adapter_foldable(ad)
            params += [ad.linear_in.weight, ad.linear_out.weight]
        layer.train()
    assert T.hs._HOOKED["mixin"] and T.hs._HOOKED["bda"]
    rot = RotaryEmbedding(kv_channels=128, rotary_percent=1.0)(M)
    return layers, rot, params


def _layers():
    if not _LAYERS:
        layers, rot, params = _build_layers()
        _LAYERS.update(layers=layers, rot=rot, params=params, h0=T._randn((M, 1, T.H), 1.0, 60),
                       g=T._randn((M, 1, T.H), 0.01, 61))
    return _LAYERS


def _step(seed=80, timing=None):
    L = _layers()
    torch.cuda.manual_seed(seed)
    h = L["h0"].clone().requires_grad_()
    if timing:
        timing[0].record()
    with te.fp8_autocast(enabled=True, fp8_recipe=T.RECIPE):
        x = h
        for layer in L["layers"]:
            x, _ = layer(x, attention_mask=None, rotary_pos_emb=L["rot"])
    if timing:
        timing[1].record()
    x.backward(L["g"])
    if timing:
        timing[2].record()
    grads = [p.grad.detach().clone() for p in L["params"]]
    for p in L["params"]:
        p.grad = None
    hg = h.grad.detach().clone()
    h.grad = None
    return x.detach(), hg, grads


def _fp8_state():
    st = []
    for layer in _layers()["layers"]:
        st += T._fp8_state(layer)
    return st


def _kernel_counts(fn):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    c = {"add": 0, "e5m2_cast": 0, "resfold": 0, "te_norm_bwd": 0, "rope_bwd": 0}
    for e in p.key_averages():
        if e.self_device_time_total <= 0:
            continue
        n = e.key
        if "CUDAFunctor_add" in n:
            c["add"] += e.count
        elif "unary_kernel" in n and "e5m2" in n:
            c["e5m2_cast"] += e.count
        elif "_rmsnorm_bwd_resfold_kernel" in n:
            c["resfold"] += e.count
        elif "_rmsnorm_bwd_triton" in n:
            c["te_norm_bwd"] += e.count
        elif "_rope_qkv_bwd_kernel" in n:
            c["rope_bwd"] += e.count
    return c


def test_layers_bit_identical_and_engaged():
    _layers()
    rf.set_enabled(False)
    rf.set_qkv_e5m2(False)
    _step(seed=1)                       # FP8-initialise everything
    saved = _fp8_state()
    o0, g0, w0 = _step()
    T._restore_fp8(saved)
    c_off = _kernel_counts(lambda: _step(seed=5))
    T._restore_fp8(saved)
    rf.set_enabled(True)
    stats0 = dict(rf.STATS)
    o1, g1, w1 = _step()
    delta = {k: rf.STATS[k] - stats0[k] for k in rf.STATS}
    print(f"  fold stats for one 2-layer step: {delta}")
    T._restore_fp8(saved)
    c_on = _kernel_counts(lambda: _step(seed=5))
    T._restore_fp8(saved)
    s0 = [c for _, c in saved]
    s1 = [t.clone() for t, _ in _fp8_state()]
    no, ng = _bits_equal(o0, o1), _bits_equal(g0, g1)
    nw = [_bits_equal(a, b) for a, b in zip(w0, w1)]
    print(f"  output differing {no}, input grad differing {ng} of {o0.numel()}; LoRA weight grads differing {nw}")
    print(f"  kernels per 2-layer fwd+bwd: off {c_off}, on {c_on}")
    assert no == 0 and ng == 0 and all(n == 0 for n in nw)
    # the FP8 state after one step: on vs off (both ran from `saved`)
    rf.set_enabled(False)
    _step()
    s_off = [t.clone() for t, _ in _fp8_state()]
    T._restore_fp8(saved)
    rf.set_enabled(True)
    _step()
    s_on = [t.clone() for t, _ in _fp8_state()]
    T._restore_fp8(saved)
    for i, (a, b) in enumerate(zip(s_off, s_on)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"FP8 state tensor {i} differs"
    # 3 folds; the e5m2 miss is the last layer's MLP add (no LayerNormLinear consumes its output); no fallback
    assert delta["fold"] == 3 and delta["marked"] == 3 and delta["e5m2_taken"] == 3 and delta["e5m2_missed"] == 1 \
        and delta["fallback_add"] == 0, delta
    assert c_on["resfold"] == 3 and c_off["resfold"] == 0, (c_off, c_on)
    assert c_off["add"] - c_on["add"] == 3 and c_off["e5m2_cast"] - c_on["e5m2_cast"] == 3, (c_off, c_on)
    assert c_off["te_norm_bwd"] - c_on["te_norm_bwd"] == 3, (c_off, c_on)
    print("PASS 2 layers: bit-identical with the fold on; 3 folds, 3 adds and 3 e5m2 casts fewer, FP8 state equal")
    # fix 8c on top: both layers' qkv grads pre-quantized by the RoPE backward (2 more e5m2 casts gone)
    rf.set_qkv_e5m2(True)
    s1 = dict(rf.STATS)
    o2, g2, w2 = _step()
    d2 = {k: rf.STATS[k] - s1[k] for k in rf.STATS}
    T._restore_fp8(saved)
    c_on2 = _kernel_counts(lambda: _step(seed=5))
    T._restore_fp8(saved)
    _step()
    s_on2 = [t.clone() for t, _ in _fp8_state()]
    T._restore_fp8(saved)
    no, ng = _bits_equal(o0, o2), _bits_equal(g0, g2)
    nw = [_bits_equal(a, b) for a, b in zip(w0, w2)]
    print(f"  + qkv e5m2: stats {d2}; output differing {no}, input grad differing {ng}; LoRA grads differing {nw}")
    print(f"  kernels per 2-layer fwd+bwd: on + qkv e5m2 {c_on2}")
    assert no == 0 and ng == 0 and all(n == 0 for n in nw)
    for i, (a, b) in enumerate(zip(s_off, s_on2)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"FP8 state tensor {i} differs (qkv e5m2)"
    assert d2["qkv_e5m2"] == 2 and d2["e5m2_taken"] == 5 and d2["fold"] == 3, d2
    assert c_on2["e5m2_cast"] == c_on["e5m2_cast"] - 2 and c_on2["rope_bwd"] == c_on["rope_bwd"] == 2, (c_on, c_on2)
    print("PASS 2 layers + qkv e5m2: bit-identical; 2 more e5m2 casts gone, FP8 state equal")


def test_benchmark(iters=10, warmup=3):
    _layers()
    saved = _fp8_state()
    res = {}
    for rnd in range(2):
        for flag in (False, True):
            rf.set_enabled(flag)
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            tf = tb = 0.0
            for i in range(warmup + iters):
                timing = ev if i >= warmup else None
                _step(seed=100 + i, timing=timing)
                if timing:
                    ev[2].synchronize()
                    tf += ev[0].elapsed_time(ev[1])
                    tb += ev[1].elapsed_time(ev[2])
            res.setdefault(flag, []).append((tf / iters, tb / iters))
            T._restore_fp8(saved)
    print(f"\n  2 layers [{M} tokens], ms per call (CUDA events, {iters} iters, 2 interleaved rounds):")
    for flag in (False, True):
        f = min(r[0] for r in res[flag]); b = min(r[1] for r in res[flag])
        print(f"    fold {'on ' if flag else 'off'}: fwd {f:.3f}  bwd {b:.3f}  total {f + b:.3f}")
    t0 = min(sum(r) for r in res[False]); t1 = min(sum(r) for r in res[True])
    print(f"  saving {1000 * (t0 - t1):.1f} us per 2 layers (3 folds; a step has 159)")


if __name__ == "__main__":
    tests = [test_kernel_exact, test_rope_e5m2_exact, test_layers_bit_identical_and_engaged]
    if "--fused-attn" in sys.argv:
        tests = [test_benchmark]
    elif "--bench" in sys.argv:
        tests.append(test_benchmark)
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
