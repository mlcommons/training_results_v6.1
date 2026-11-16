"""eval_trim's no-new-benchmark rule for TE's autotuned Triton kernels, on one GPU.

Run inside the training image:
    cd /workspace/trainloom_patches && python check_eval_trim_autotune.py
After the training shape [8192, N] is tuned, the eval lengths [M, N] must run without benchmarking (bench_time not
updated) on the tuned config, and the FP8 cast must be byte-identical to an autotuned run of the same input.
"""

import sys

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch.triton_kernels import cast_transpose as ct

import eval_trim  # noqa: F401  (installs the rule)

N = 28672
ok = True


def cast(x):
    c = torch.empty(x.shape, dtype=torch.uint8, device="cuda")
    t = torch.empty(x.shape[::-1], dtype=torch.uint8, device="cuda")
    amax = torch.zeros(1, device="cuda")
    sinv = torch.empty(1, device="cuda")
    ct.te_cast_transpose_noop_triton(x, torch.empty(0, device="cuda"), torch.tensor([4.0], device="cuda"), c, t, amax, sinv, tex.DType.kFloat8E4M3)
    return c, t, amax


tuner = ct._cast_transpose_triton
cast(torch.randn(8192, N, device="cuda", dtype=torch.bfloat16))
trained = dict(tuner.cache)
print(f"tuned for training shape: {[(k[:2], str(c)) for k, c in trained.items()]}")
for M in (1536, 4096, 7680):
    tuner.bench_time = None
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    c, t, a = cast(x)
    new = [k for k in tuner.cache if k not in trained and k[0] == M]
    reused = all(str(tuner.cache[k]) == str(next(iter(trained.values()))) for k in new)
    benched = tuner.bench_time is not None
    same = True
    configs = tuner.configs
    for cfg in configs:   # reference: TE's own launcher with the tuner held to one config (no benchmark path)
        tuner.configs = [cfg]
        c2, t2, a2 = cast(x)
        same &= torch.equal(c, c2) and torch.equal(t, t2) and torch.equal(a, a2)
    tuner.configs = configs
    ok &= reused and not benched and same
    print(f"M={M}: benchmarked {benched}, reused training config {reused}, cast identical under every config {same}")
sys.exit(0 if ok else 1)
