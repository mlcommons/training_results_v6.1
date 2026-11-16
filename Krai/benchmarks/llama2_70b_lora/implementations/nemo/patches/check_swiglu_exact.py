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

"""Bit-exactness sweep for the fused act-store SwiGLU kernels vs TE's gated_act / dgated_act (fix #1 gate 1)."""
import sys, torch, time
sys.path.insert(0, "/p")
import transformer_engine.pytorch as te, transformer_engine_torch as tex
from hs_swiglu_fp8_triton import swiglu_fwd_store, dswiglu_fp8in
torch.manual_seed(0); M, H = 8192, 28672; dev = "cuda"
swiglu = te.ops.SwiGLU()
cases = [("randn*%g" % s, lambda s=s: torch.randn(M, 2*H, device=dev) * s) for s in (0.25, 1, 4, 16, 64)]
cases.append(("heavy-tail", lambda: torch.randn(M, 2*H, device=dev) * torch.exp(torch.randn(M, 2*H, device=dev))))
bad = 0
for name, gen in cases:
    x = gen().to(torch.bfloat16); g = (torch.randn(M, H, device=dev) * (float(name.split('*')[1]) if '*' in name else 1)).to(torch.bfloat16)
    ref_out = swiglu(x); out, x_fp8 = swiglu_fwd_store(x)
    ref_fp8 = x.to(torch.float8_e4m3fn)
    fwd_diff = (out.view(torch.int16) != ref_out.view(torch.int16)).sum().item()
    fp8_diff = (x_fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).sum().item()
    ref_dx = tex.dswiglu(g, ref_fp8.to(torch.bfloat16), None); dx = dswiglu_fp8in(g, x_fp8)
    bwd_diff = (dx.view(torch.int16) != ref_dx.view(torch.int16)).sum().item()
    nan = torch.isnan(ref_fp8.float()).sum().item()
    print(f"{name:12} fwd diff {fwd_diff:8d}  fp8 bytes diff {fp8_diff:8d}  bwd diff {bwd_diff:8d}  (stored NaN codes {nan})")
    bad += fwd_diff + fp8_diff + bwd_diff
# timing
x = (torch.randn(M, 2*H, device=dev)).to(torch.bfloat16); g = torch.randn(M, H, device=dev).to(torch.bfloat16)
for label, fn in (("fused fwd", lambda: swiglu_fwd_store(x)), ("fused bwd", lambda: dswiglu_fp8in(g, x_fp8)),
                  ("te fwd", lambda: swiglu(x)), ("te bwd", lambda: tex.dswiglu(g, x, None))):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record(); [fn() for _ in range(20)]; t1.record(); torch.cuda.synchronize(); print(f"{label:10} {t0.elapsed_time(t1)/20:.3f} ms")
print("RESULT:", "BIT-EXACT" if bad == 0 else f"{bad} differing elements")
