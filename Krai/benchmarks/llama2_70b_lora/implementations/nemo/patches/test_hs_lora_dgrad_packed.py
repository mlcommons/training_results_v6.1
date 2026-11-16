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

"""Test + microbenchmark for v5.0 item A (hs_lora_dgrad_packed.py).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_lora_dgrad_packed.py

At both production shapes (dgrad [8192, 10240] qkv, [8192, 8192] proj; K = 16): the new kernel's in-place result
equals lora_dropout_packed's `_lora_dgrad_acc_packed_kernel` (and fix 14's bool-mask kernel) bit for bit, for
several scales, dropout masks of p = 0.1 / 0.5 / 0 / 1, heavy-tailed grads and NaN/inf in dgrad; then timing.
"""

import os
import sys

os.environ.setdefault("HS_LORA_DGRAD_PACKED_FAST", "1")
os.environ.setdefault("HS_LORA_EPILOGUE_FAST", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import hs_lora_epilogue as hle  # noqa: E402
import lora_dropout_packed as ldp  # noqa: E402
import hs_lora_dgrad_packed as fast  # noqa: E402
import triton  # noqa: E402

DEV = "cuda"
K = 16


def _ref_packed(d, ga, a_w, packed, scale):
    M, N = d.shape
    ldp._lora_dgrad_acc_packed_kernel[(triton.cdiv(M, hle._BM), triton.cdiv(N, hle._BN))](
        d, ga, a_w, packed, M, N, float(scale), K=K, BM=hle._BM, BN=hle._BN, num_warps=hle._WARPS)
    return d


def _neq(a, b):
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    return int(((ai != bi) & ~(torch.isnan(a) & torch.isnan(b))).sum())


def test_exact():
    g = torch.Generator(device=DEV).manual_seed(0)
    n = 0
    for M, N in ((8192, 10240), (8192, 8192)):
        a_w = (torch.randn(K, N, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
        for p in (0.1, 0.5, 0.0, 1.0):
            mask = torch.rand(M, N, generator=g, device=DEV) >= p
            packed = ldp.pack_mask(mask)
            for gstd, dstd in ((1e-3, 1e-2), (1.0, 1.0), (30.0, 1e-4)):
                ga = (torch.randn(M, K, generator=g, device=DEV) * gstd).to(torch.bfloat16)
                d0 = (torch.randn(M, N, generator=g, device=DEV) * dstd).to(torch.bfloat16)
                if gstd == 30.0:
                    d0.view(-1)[::9973] = float("nan")
                    d0.view(-1)[7::10007] = float("inf")
                scale = 1.0 / (1.0 - 0.1)
                r = _ref_packed(d0.clone(), ga, a_w, packed, scale)
                c = fast.dgrad_acc_packed_fast_(d0.clone(), ga, a_w, packed, scale)
                nd = _neq(c, r)
                print(f"  [{M}, {N}] p {p} ga~{gstd} d~{dstd}: differing {nd}")
                assert nd == 0
                n += 1
    print(f"PASS exactness: {n} cases, bit-identical to _lora_dgrad_acc_packed_kernel")


def test_dispatch():
    M, N = 8192, 8192
    g = torch.Generator(device=DEV).manual_seed(1)
    a_w = (torch.randn(K, N, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    ga = torch.randn(M, K, generator=g, device=DEV).to(torch.bfloat16)
    packed = ldp.pack_mask(torch.rand(M, N, generator=g, device=DEV) >= 0.1)
    d = torch.zeros(M, N, device=DEV, dtype=torch.bfloat16)
    st = dict(fast.STATS)
    hle.lora_dgrad_acc_(d, ga, a_w, packed, 1 / 0.9)
    assert fast.STATS["fast"] == st["fast"] + 1, fast.STATS
    print("PASS dispatch: hs_lora_epilogue.lora_dgrad_acc_ routes packed masks to the new kernel")


def test_bench(iters=30):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    g = torch.Generator(device=DEV).manual_seed(2)
    for M, N in ((8192, 10240), (8192, 8192)):
        a_w = (torch.randn(K, N, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
        ga = torch.randn(M, K, generator=g, device=DEV).to(torch.bfloat16)
        packed = ldp.pack_mask(torch.rand(M, N, generator=g, device=DEV) >= 0.1)
        d = torch.zeros(M, N, device=DEV, dtype=torch.bfloat16)
        res = {}
        for name, fn in (("packed (v4.2)", lambda: _ref_packed(d, ga, a_w, packed, 1.1)),
                         ("packed fast (A)", lambda: fast.dgrad_acc_packed_fast_(d, ga, a_w, packed, 1.1))):
            for _ in range(3):
                fn()
            torch.cuda.synchronize(); ev[0].record()
            for _ in range(iters):
                fn()
            ev[1].record(); torch.cuda.synchronize()
            res[name] = ev[0].elapsed_time(ev[1]) / iters * 1000
        gb = 2 * M * N * 2 / 1e9
        print(f"  [{M}, {N}]: " + ", ".join(f"{k} {v:.1f} us ({gb / v * 1e3:.2f} TB/s)" for k, v in res.items()))


if __name__ == "__main__":
    for t in (test_exact, test_dispatch, test_bench):
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
