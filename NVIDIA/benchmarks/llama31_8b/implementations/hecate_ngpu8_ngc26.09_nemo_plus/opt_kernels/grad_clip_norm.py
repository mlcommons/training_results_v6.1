# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


import torch
import triton
import triton.language as tl

CHUNK = 32768
_BLOCK = 4096
_RBLOCK = 2048


@triton.jit
def _grad_sumsq_kernel(
    g_ptr,
    off_ptr,
    len_ptr,
    partial_ptr,
    counter_ptr,
    out_ptr,
    n_chunks,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.load(off_ptr + pid)
    n = tl.load(len_ptr + pid)
    if ALIGNED:
        off = tl.multiple_of(off, 8)
    base = g_ptr + off
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.range(0, CHUNK, BLOCK):
        idx = s + tl.arange(0, BLOCK)
        x = tl.load(base + idx, mask=idx < n, other=0.0, eviction_policy="evict_first").to(tl.float32)
        acc += x * x
    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))
    tl.debug_barrier()
    ticket = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == n_chunks - 1:
        r = tl.zeros([RBLOCK], dtype=tl.float32)
        for s in tl.range(0, n_chunks, RBLOCK):
            i = s + tl.arange(0, RBLOCK)
            r += tl.load(partial_ptr + i, mask=i < n_chunks, other=0.0, cache_modifier=".cg")
        tl.store(out_ptr, tl.sum(r, axis=0))
        tl.store(counter_ptr, 0)


@triton.jit
def _clip_coeff_kernel(sumsq_ptr, out_ptr, max_norm):
    s = tl.load(sumsq_ptr)
    norm = tl.sqrt_rn(s)
    rcp = tl.div_rn(1.0, norm + 1.0e-6)
    coeff = tl.minimum(rcp * max_norm, 1.0)
    tl.store(out_ptr, norm)
    tl.store(out_ptr + 1, coeff)


@torch.library.custom_op(
    "optimized::grad_sumsq", mutates_args=("partials", "counter", "out"), device_types="cuda"
)
def grad_sumsq(
    grads_base: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    partials: torch.Tensor,
    counter: torch.Tensor,
    out: torch.Tensor,
    aligned: bool,
) -> None:
    n_chunks = offsets.numel()
    _grad_sumsq_kernel[(n_chunks,)](
        grads_base, offsets, lengths, partials, counter, out, n_chunks,
        CHUNK=CHUNK, BLOCK=_BLOCK, RBLOCK=_RBLOCK, ALIGNED=aligned, num_warps=8,
    )


@torch.library.custom_op("optimized::clip_coeff", mutates_args=("out",), device_types="cuda")
def clip_coeff(sumsq: torch.Tensor, out: torch.Tensor, max_norm: float) -> None:
    _clip_coeff_kernel[(1,)](sumsq, out, max_norm, num_warps=1)


class GradNormPlan:

    def __init__(self, grads):
        assert len(grads) > 0
        dev = grads[0].device
        for g in grads:
            assert g.dtype == torch.bfloat16 and g.is_contiguous() and g.device == dev
        self.base = grads[0]
        base_ptr = self.base.data_ptr()
        offs, lens = [], []
        aligned = True
        for g in grads:
            n = g.numel()
            if n == 0:
                continue
            d = g.data_ptr() - base_ptr
            assert d % 2 == 0
            if g.data_ptr() % 16 != 0:
                aligned = False
            e = d // 2
            for c in range(0, n, CHUNK):
                offs.append(e + c)
                lens.append(min(CHUNK, n - c))
        self.aligned = aligned
        self.offsets = torch.tensor(offs, dtype=torch.int64, device=dev)
        self.lengths = torch.tensor(lens, dtype=torch.int32, device=dev)
        self.partials = torch.empty(len(offs), dtype=torch.float32, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.sumsq = torch.zeros(1, dtype=torch.float32, device=dev)
        self.numel = sum(lens)
        self.ptrs = tuple(g.data_ptr() for g in grads)

    def sumsq_(self):
        grad_sumsq(self.base, self.offsets, self.lengths, self.partials, self.counter, self.sumsq,
                   self.aligned)
        return self.sumsq


def norm_and_coeff(sumsq: torch.Tensor, max_norm: float) -> torch.Tensor:
    out = torch.empty(2, dtype=torch.float32, device=sumsq.device)
    clip_coeff(sumsq, out, float(max_norm))
    return out
