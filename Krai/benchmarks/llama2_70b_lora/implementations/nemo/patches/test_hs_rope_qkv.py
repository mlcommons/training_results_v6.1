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

"""Tests + microbenchmark for fix 4 (hs_rope_qkv.py, wired into hs_lora_epilogue.py's mixin with HS_ROPE_FUSED=1).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_rope_qkv.py [--bench | --bench-only]

1. Function level, production shape (s 8192, b 1, mixed_qkv [s,1,10240]): the fused forward equals Megatron's chain
   (view -> TE SplitAlongDim -> q reshape -> apply_rotary_pos_emb with apply_rope_fusion (apex) on q and k, v) and the
   fused backward equals autograd through that chain (apex RoPE backward + SplitAlongDim cat), bit for bit, on normal
   inputs at 5 scales, heavy tails and special values (bf16 subnormals, overflow, +-0, +-inf, NaN). NaN-vs-NaN equal.
2. Attention block (Megatron SelfAttention + NeMo LoRA mixin + fix #2 folds, FP8): forward output bit-identical with
   the flag off and on, and the fused path engaged. (The block backward runs through CK attention's atomic dq,
   which is nondeterministic run to run, so backward exactness is proven at the function level in test 1.)
3. --bench: attention block fwd and bwd, off vs on, CUDA events.
"""

import os
import sys

os.environ.setdefault("HS_ROPE_FUSED", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import test_hs_lora_epilogue as T  # noqa: E402  (recipe env; imports hs_lora_epilogue = fix #2 + the fix 4 hook)
import hs_lora_epilogue as hs  # noqa: E402
import hs_rope_qkv as rq  # noqa: E402

DEV = "cuda"
S = 8192


def _neq(a, b):
    ia, ib = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    nan = torch.isnan(a) & torch.isnan(b)
    return int(((ia != ib) & ~nan).sum().item())


def _freqs():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    return RotaryEmbedding(kv_channels=128, rotary_percent=1.0)(S).contiguous()


def _chain(mixed, freqs, cfg):
    from megatron.core.extensions.transformer_engine import SplitAlongDim
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    m = mixed.view(S, 1, 8, 1280)
    q, k, v = SplitAlongDim(m, 3, [1024, 128, 128])
    q = q.reshape(S, 1, -1, 128)
    q = apply_rotary_pos_emb(q, freqs, config=cfg)
    k = apply_rotary_pos_emb(k, freqs, config=cfg)
    return q, k, v


def _inputs():
    g = torch.Generator(device=DEV).manual_seed(5)
    for s in (0.25, 1.0, 4.0, 16.0, 64.0):
        yield f"normal x{s}", (torch.randn(S, 1, 10240, generator=g, device=DEV) * s).to(torch.bfloat16)
    u = torch.rand(S, 1, 10240, generator=g, device=DEV).clamp_(1e-6, 1 - 1e-6)
    yield "heavy tail", torch.tan(torch.pi * (u - 0.5)).clamp_(-3e38, 3e38).to(torch.bfloat16)
    x = torch.randn(S, 1, 10240, generator=g, device=DEV)
    flat = x.view(-1)
    flat[0::997] = 1e-40          # fp32 subnormal -> bf16 subnormal
    flat[1::1009] = 3e38          # overflow after the rotation
    flat[2::1013] = 0.0
    flat[3::1019] = -0.0
    flat[4::1021] = float("inf")
    flat[5::1031] = -float("inf")
    flat[6::1033] = float("nan")
    yield "special values", x.to(torch.bfloat16)


def test_function_exact():
    T._init_parallel()
    cfg = T._config()
    freqs = _freqs()
    n = 0
    for name, mixed in _inputs():
        m_ref = mixed.clone().requires_grad_()
        q0, k0, v0 = _chain(m_ref, freqs, cfg)
        m_new = mixed.clone().requires_grad_()
        q1, k1, v1 = rq.fused_qkv_rope(m_new, freqs)
        fwd = (_neq(q1, q0), _neq(k1, k0), _neq(v1, v0))
        gen = torch.Generator(device=DEV).manual_seed(n)
        dq = torch.randn(q0.shape, generator=gen, device=DEV).to(torch.bfloat16)
        dk = torch.randn(k0.shape, generator=gen, device=DEV).to(torch.bfloat16)
        dv = torch.randn(v0.shape, generator=gen, device=DEV).to(torch.bfloat16)
        if name == "special values":
            dq.view(-1)[7::1039] = float("nan"); dk.view(-1)[8::1049] = -0.0; dv.view(-1)[9::1051] = float("inf")
        (g0,) = torch.autograd.grad((q0, k0, v0), (m_ref,), (dq, dk, dv))
        (g1,) = torch.autograd.grad((q1, k1, v1), (m_new,), (dq, dk, dv))
        bwd = _neq(g1, g0)
        print(f"  {name:16s} fwd diff q/k/v {fwd}  bwd diff d_mixed {bwd}   (q contiguous: {q1.is_contiguous()})")
        assert fwd == (0, 0, 0) and bwd == 0, name
        n += 1
    print(f"PASS function exactness: {n} input sets at [{S}, 1, 10240], fwd and bwd bit-identical to Megatron/apex")


def test_block_forward_and_engagement():
    attn, step = T._attention_block()
    full = "add_res_bwd"
    saved = T._fp8_state(attn)          # delayed scaling updates scales every step: restore before each run

    def run(flag):
        T._restore_fp8(saved)
        hs.FLAGS.rope = flag
        out, _, _ = step(full)
        return out

    ctl0 = run(False)
    ctl1 = run(False)
    nd_ctl = _neq(ctl1, ctl0)
    s0 = dict(hs.STATS)
    on = run(True)
    eng = hs.STATS["rope_fused"] - s0["rope_fused"]
    nd = _neq(on, ctl0)
    T._restore_fp8(saved)
    print(f"  block forward: off vs off {nd_ctl}, off vs on {nd} differing of {ctl0.numel()}; "
          f"fused path engaged {eng} time(s)")
    assert nd_ctl == 0, "the block forward is not repeatable; the comparison is meaningless"
    assert eng == 1 and nd == 0
    print("PASS attention block: forward bit-identical with fix 4, fused path engaged")


def test_benchmark(iters=20, warmup=3):
    attn, step = T._attention_block()
    full = "add_res_bwd"
    res = {}
    saved = T._fp8_state(attn)
    for rnd in range(2):
        for flag in (False, True):
            T._restore_fp8(saved)
            hs.FLAGS.rope = flag
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            tf = tb = 0.0
            for i in range(warmup + iters):
                step(full, timing=ev if i >= warmup else None)
                if i >= warmup:
                    ev[2].synchronize()
                    tf += ev[0].elapsed_time(ev[1]); tb += ev[1].elapsed_time(ev[2])
            res.setdefault(flag, []).append((tf / iters, tb / iters))
    print(f"\n  attention block [{S} tokens], ms per call (CUDA events, {iters} iters, 2 interleaved rounds):")
    for flag in (False, True):
        f = min(r[0] for r in res[flag]); b = min(r[1] for r in res[flag])
        print(f"    fix 4 {'on ' if flag else 'off'}: fwd {f:.3f}  bwd {b:.3f}  total {f + b:.3f}")
    f0 = min(r[0] for r in res[False]); f1 = min(r[0] for r in res[True])
    b0 = min(r[1] for r in res[False]); b1 = min(r[1] for r in res[True])
    print(f"  saving fwd {1000 * (f0 - f1):.1f} us, bwd {1000 * (b0 - b1):.1f} us per layer; at ACL 3 "
          f"(83 fwd, 80 bwd) = {83 * (f0 - f1) + 80 * (b0 - b1):.1f} ms/step (isolated; screens decide)")


if __name__ == "__main__":
    tests = [test_function_exact, test_block_forward_and_engagement]
    if "--bench-only" in sys.argv:
        tests = [test_benchmark]
    elif "--bench" in sys.argv:
        tests.append(test_benchmark)
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
