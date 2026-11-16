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
#
# The AdamW / Adam update order and bias correction transcribe Transformer Engine's fused Adam
# (common/multi_tensor/adam.cu, AdamFunctorMaster; Apache-2.0, Copyright (c) NVIDIA CORPORATION
# & AFFILIATES) into Triton so that the results match bitwise.


import torch
import triton
import triton.language as tl

CHUNK = 16384
_BLOCK = 512


@triton.jit
def _adam_kernel(
    g_ptr, p_ptr, w_ptr, m_ptr, v_ptr, tab_ptr, coeff_ptr,
    lr, beta1, beta2, omb1, omb2, bc1, bc2, eps, wd,
    ADAMW: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr, ALIGNED: tl.constexpr,
    STAGES: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    row = tab_ptr + pid * 6
    og = tl.load(row + 0)
    op = tl.load(row + 1)
    ow = tl.load(row + 2)
    om = tl.load(row + 3)
    ov = tl.load(row + 4)
    n = tl.load(row + 5)
    if ALIGNED:
        og = tl.multiple_of(og, 8)
        op = tl.multiple_of(op, 8)
        ow = tl.multiple_of(ow, 4)
        om = tl.multiple_of(om, 4)
        ov = tl.multiple_of(ov, 4)
    coeff = tl.load(coeff_ptr)
    for s in tl.range(0, CHUNK, BLOCK, num_stages=STAGES):
        idx = s + tl.arange(0, BLOCK)
        mask = idx < n
        g = tl.load(g_ptr + og + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        w = tl.load(w_ptr + ow + idx, mask=mask, other=0.0, eviction_policy="evict_first")
        m = tl.load(m_ptr + om + idx, mask=mask, other=0.0, eviction_policy="evict_first")
        v = tl.load(v_ptr + ov + idx, mask=mask, other=0.0, eviction_policy="evict_first")
        g = g * coeff
        if ADAMW:
            m = beta1 * m + omb1 * g
            v = beta2 * v + omb2 * g * g
            m_hat = tl.div_rn(m, bc1)
            v_hat = tl.div_rn(v, bc2)
            denom = tl.sqrt_rn(v_hat) + eps
            update = tl.div_rn(m_hat, denom) + wd * w
        else:
            g = g + wd * w
            m = beta1 * m + omb1 * g
            v = beta2 * v + omb2 * g * g
            m_hat = tl.div_rn(m, bc1)
            v_hat = tl.div_rn(v, bc2)
            denom = tl.sqrt_rn(v_hat) + eps
            update = tl.div_rn(m_hat, denom)
        w = w - lr * update
        tl.store(w_ptr + ow + idx, w, mask=mask, eviction_policy="evict_first")
        tl.store(m_ptr + om + idx, m, mask=mask, eviction_policy="evict_first")
        tl.store(v_ptr + ov + idx, v, mask=mask, eviction_policy="evict_first")
        tl.store(p_ptr + op + idx, w.to(tl.bfloat16), mask=mask, eviction_policy="evict_first")


@torch.library.custom_op(
    "optimized::adam_bf16_master_step",
    mutates_args=("p_base", "w_base", "m_base", "v_base"),
    device_types="cuda",
)
def adam_bf16_master_step(
    g_base: torch.Tensor, p_base: torch.Tensor, w_base: torch.Tensor, m_base: torch.Tensor,
    v_base: torch.Tensor, table: torch.Tensor, coeff: torch.Tensor, aligned: bool,
    lr: float, beta1: float, beta2: float, one_minus_beta1: float, one_minus_beta2: float,
    bias_correction1: float, bias_correction2: float, eps: float, weight_decay: float,
    adam_w_mode: bool,
) -> None:
    n_chunks = table.shape[0]
    if n_chunks == 0:
        return
    _adam_kernel[(n_chunks,)](
        g_base, p_base, w_base, m_base, v_base, table, coeff,
        lr, beta1, beta2, one_minus_beta1, one_minus_beta2, bias_correction1, bias_correction2,
        eps, weight_decay,
        ADAMW=adam_w_mode, CHUNK=CHUNK, BLOCK=_BLOCK, ALIGNED=aligned, num_warps=4,
    )


def _f32(x):
    return float(torch.tensor(x, dtype=torch.float32).item())


class AdamPlan:

    def __init__(self, g_l, p_l, w_l, m_l, v_l):
        self.empty = len(g_l) == 0
        if self.empty:
            return
        dev = g_l[0].device
        lists = (g_l, p_l, w_l, m_l, v_l)
        esz = (2, 2, 4, 4, 4)
        for l, dt in zip(lists, (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32,
                                 torch.float32)):
            for t in l:
                assert t.dtype == dt and t.is_contiguous() and t.device == dev
        self.bases = [l[0] for l in lists]
        rows = []
        aligned = True
        for k in range(len(g_l)):
            n = g_l[k].numel()
            assert all(l[k].numel() == n for l in lists)
            if n == 0:
                continue
            offs = []
            for l, base, sz in zip(lists, self.bases, esz):
                d = l[k].data_ptr() - base.data_ptr()
                assert d % sz == 0
                if l[k].data_ptr() % 16 != 0:
                    aligned = False
                offs.append(d // sz)
            for c in range(0, n, CHUNK):
                rows.append([o + c for o in offs] + [min(CHUNK, n - c)])
        self.aligned = aligned
        self.table = torch.tensor(rows, dtype=torch.int64, device=dev).reshape(-1, 6)
        self.numel = sum(r[5] for r in rows)

    def step(self, coeff, lr, beta1, beta2, eps, weight_decay, step, bias_correction, adam_w_mode):
        if self.empty or self.table.shape[0] == 0:
            return
        b1, b2 = _f32(beta1), _f32(beta2)
        if bias_correction:
            bc1, bc2 = _f32(1.0 - b1 ** step), _f32(1.0 - b2 ** step)
        else:
            bc1 = bc2 = 1.0
        adam_bf16_master_step(
            *self.bases, self.table, coeff, self.aligned,
            _f32(lr), b1, b2, _f32(1.0 - b1), _f32(1.0 - b2), bc1, bc2, _f32(eps),
            _f32(weight_decay), bool(adam_w_mode),
        )
