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
"""Fix 13 (microbenchmark lane l6_attn_norm, campaign kfix_261010): TE's Triton RMSNorm forward with a larger grid.

TE 2.2's te_rmsnorm_fwd_triton launches min(rows, #CUs) = 304 programs (10% occupancy at [8192, 8192]) and autotunes
num_warps / waves_per_eu per process, which changes the row-sum tree: today the recipe's norm output is not even
reproducible run to run (12-20% of rows' rsigma differ between processes). This module (HS_RMSNORM_GRID=1 to enable;
load anywhere before the first forward):
  - launches the forward with min(rows, 4864) programs for a BF16 output and min(rows, 2432) for an FP8 output
    (HS_RMSNORM_GRID_BF16 / HS_RMSNORM_GRID_FP8), the lane's measured sweet spots (0.83x / 0.92x);
  - pins the kernel to TE's own non-autotuned launch (Triton default: 4 warps, waves_per_eu 0), so every run computes
    the same tree - one of the configurations the autotuner picks today.
Grid size is bit-exact at fixed num_warps (programs walk rows with a grid stride; each row's reduction is the same);
the FP8 amax is a max, order-free. The backward is untouched (already at ~78% of the HBM roof).
"""
import os

import transformer_engine.pytorch.module._common as _common
import transformer_engine.pytorch.triton_kernels.rmsnorm as _rms
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

ENABLED = os.getenv("HS_RMSNORM_GRID", "0") == "1"
GRID_BF16 = int(os.getenv("HS_RMSNORM_GRID_BF16", "4864"))
GRID_FP8 = int(os.getenv("HS_RMSNORM_GRID_FP8", "2432"))
STATS = {"fwd": 0}
_orig_fwd = _rms.te_rmsnorm_fwd_triton


def te_rmsnorm_fwd_triton(input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma,
                          autotune=True):
    target = GRID_FP8 if isinstance(quantizer, Float8Quantizer) else GRID_BF16
    saved = _rms.num_programs                       # shared with the backward: override for this call only
    _rms.num_programs = lambda x, sm_margin=None: min(x.shape[0], target)
    try:
        STATS["fwd"] += 1
        return _orig_fwd(input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma, autotune=False)
    finally:
        _rms.num_programs = saved


if ENABLED:
    _common.te_rmsnorm_fwd_triton = te_rmsnorm_fwd_triton   # _get_normalization_func looks it up at call time
print(f"[hyperspindle] hs_rmsnorm_grid: {'on' if ENABLED else 'off'} (grid bf16 {GRID_BF16}, fp8 {GRID_FP8}, "
      f"pinned non-autotuned launch)", flush=True)
