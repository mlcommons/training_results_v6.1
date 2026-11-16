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

"""v5.0 item A: the LoRA dgrad fold with packed dropout masks, at fix 14's bandwidth (env HS_LORA_DGRAD_PACKED_FAST=1;
load after hs_lora_epilogue and lora_dropout_packed).

lora_dropout_packed replaces hs_lora_epilogue.lora_dgrad_acc_ for packed masks with `_lora_dgrad_acc_packed_kernel`,
a copy of fix #2's original kernel (2-D grid, plain loads, 128x128 / 8 warps) that reads one mask byte per element
(`packed[offs // 8] >> (offs % 8)`). So fix 14's restructured dgrad kernel never runs with packed dropout: the v4.2
profile (docs/analysis/v42_profile_geak_eval.md) measured 490 us per call, 0.62 TB/s, 78 ms/step.

This kernel is fix 14's `_lora_dgrad_acc_fast_kernel` (N-fastest 1-D grid, 32 x 128 tiles, 4 warps, the same
transposed dot and scale, `.cg` streaming loads) with the mask read as packed bytes: one [BM, BN/8] byte tile per
program, expanded to [BM, BN] bits in registers (bit i of byte j is element 8 j + i, lora_dropout_packed's layout).
Bit-exact with the packed kernel it replaces: the dot, the scale, the where and the BF16 add are the same operations
(fix 14 is bit-identical to fix #2's kernel); only the memory access changes. Shapes that are not multiples of the
tile (or of 8 columns) fall back to the previous function.
"""

import os

import torch
import triton
import triton.language as tl

import hs_lora_epilogue as hle

ENABLED = os.getenv("HS_LORA_DGRAD_PACKED_FAST", "0") == "1"
BM, BN, WARPS = 32, 128, 4
STATS = {"fast": 0, "fallback": 0}


@triton.jit
def _lora_dgrad_acc_packed_fast_kernel(dgrad_ptr, ga_ptr, a_w_ptr, packed_ptr, M, N, scale,
                                       K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    nn = N // BN
    pid_m = tl.program_id(0) // nn
    pid_n = tl.program_id(0) % nn
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, K)
    gat = tl.load(ga_ptr + rm[None, :] * K + rk[:, None])        # [K, BM]
    awt = tl.load(a_w_ptr + rk[None, :] * N + rn[:, None])       # [BN, K]
    acc = tl.trans(tl.dot(awt, gat)) * scale                      # [BM, BN], as fix 14
    rb = pid_n * (BN // 8) + tl.arange(0, BN // 8)
    pb = tl.load(packed_ptr + rm[:, None] * (N // 8) + rb[None, :], cache_modifier=".cg").to(tl.int32)
    bits = (pb[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1   # [BM, BN/8, 8]
    keep = tl.reshape(bits, [BM, BN])                              # column 8 j + i <- bit i of byte j
    acc = tl.where(keep != 0, acc, 0.0)
    offs = rm[:, None] * N + rn[None, :]
    d = tl.load(dgrad_ptr + offs, cache_modifier=".cg").to(tl.float32)
    tl.store(dgrad_ptr + offs, (d + acc).to(tl.bfloat16))


def dgrad_acc_packed_fast_(dgrad2d, ga, a_w, packed, scale):
    """dgrad2d [M, N] += unpack(packed) * scale * ga @ a_w, in place (bf16; packed uint8 [M*N/8])."""
    M, N = dgrad2d.shape
    K = ga.shape[1]
    _lora_dgrad_acc_packed_fast_kernel[((M // BM) * (N // BN),)](
        dgrad2d, ga, a_w, packed, M, N, float(scale), K=K, BM=BM, BN=BN, num_warps=WARPS, matrix_instr_nonkdim=16)
    return dgrad2d


def _install():
    prev = hle.lora_dgrad_acc_   # lora_dropout_packed's wrapper (or fix #2's own function)

    def lora_dgrad_acc_(dgrad2d, ga, a_w, mask, scale):
        if (ENABLED and mask is not None and mask.dtype == torch.uint8 and mask.dim() == 1):
            M, N = dgrad2d.shape
            K = ga.shape[1]
            if (M % BM == 0 and N % BN == 0 and dgrad2d.is_contiguous() and ga.is_contiguous()
                    and a_w.is_contiguous() and a_w.shape == (K, N) and hle._rank_ok(K)
                    and mask.numel() == M * N // 8 and dgrad2d.dtype == ga.dtype == a_w.dtype == torch.bfloat16):
                STATS["fast"] += 1
                if STATS["fast"] in (1, 1000) and os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")) == "0":
                    print(f"[hyperspindle] hs_lora_dgrad_packed: engaged, {STATS}", flush=True)
                return dgrad_acc_packed_fast_(dgrad2d, ga, a_w, mask, scale)
            STATS["fallback"] += 1
        return prev(dgrad2d, ga, a_w, mask, scale)

    lora_dgrad_acc_._hs_prev = prev
    hle.lora_dgrad_acc_ = lora_dgrad_acc_


_install()
print(f"[hyperspindle] hs_lora_dgrad_packed: {'on' if ENABLED else 'off'} (fix 14 dgrad kernel for packed dropout "
      f"masks; HS_LORA_DGRAD_PACKED_FAST=1)", flush=True)
