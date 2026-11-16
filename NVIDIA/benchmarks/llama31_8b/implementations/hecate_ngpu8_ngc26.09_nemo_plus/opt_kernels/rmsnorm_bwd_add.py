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


@triton.jit
def _rmsnorm_bwd_add_kernel(
    dy_ptr,
    x_ptr,
    rstd_ptr,
    w_ptr,
    rg_ptr,
    dw_acc_ptr,
    dw_out_ptr,
    counter_ptr,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        w = w + 1.0
    dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    n_blocks = tl.cdiv(M, BLOCK_M)
    for blk in tl.range(pid, n_blocks, nprog, num_stages=NUM_STAGES):
        rows = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        offs = rows.to(tl.int64)[:, None] * N + cols[None, :]
        mask = rmask[:, None] & cmask[None, :]
        dz = tl.load(dy_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        rg = tl.load(rg_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        rs = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
        y = x * rs[:, None]
        dyw = dz * w[None, :]
        mdyy = tl.sum(dyw * y, axis=1) * (1.0 / N)
        dx = rs[:, None] * (dyw - mdyy[:, None] * y)
        dw += tl.sum(dz * y, axis=0)
        out = dx.to(tl.bfloat16).to(tl.float32) + rg
        tl.store(rg_ptr + offs, out.to(tl.bfloat16), mask=mask)
    tl.atomic_add(dw_acc_ptr + cols, dw, mask=cmask, sem="relaxed")
    tl.debug_barrier()
    ticket = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == nprog - 1:
        tl.debug_barrier()
        acc = tl.atomic_xchg(dw_acc_ptr + cols, tl.zeros([BLOCK_N], dtype=tl.float32), mask=cmask,
                             sem="relaxed")
        tl.store(dw_out_ptr + cols, acc.to(dw_out_ptr.dtype.element_ty), mask=cmask)
        tl.atomic_xchg(counter_ptr, 0, sem="relaxed")


_CFG = {4096: (1, 4, 2, 3)}
_DEFAULT_CFG = (1, 8, 2, 1)

_ws = {}
_sm_count = {}


def _workspace(device, n):
    key = (device.index, n)
    ws = _ws.get(key)
    if ws is None:
        acc = torch.zeros(n, dtype=torch.float32, device=device)
        cnt = torch.zeros(1, dtype=torch.int32, device=device)
        ws = _ws[key] = (acc, cnt)
    return ws


def _num_sms(device):
    n = _sm_count.get(device.index)
    if n is None:
        n = _sm_count[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return n


def _launch(dy, x, rstd, gamma, rg, dgamma, zero_centered_gamma, sm_margin, cfg=None):
    M, N = x.shape
    block_m, num_warps, ctas_per_sm, num_stages = cfg or _CFG.get(N, _DEFAULT_CFG)
    acc, cnt = _workspace(x.device, N)
    sms = max(1, _num_sms(x.device) - max(0, int(sm_margin)))
    grid = (max(1, min(triton.cdiv(M, block_m), sms * ctas_per_sm)),)
    _rmsnorm_bwd_add_kernel[grid](
        dy, x, rstd, gamma, rg, acc, dgamma, cnt, M,
        N=N, BLOCK_N=triton.next_power_of_2(N), BLOCK_M=block_m,
        ZERO_CENTERED=bool(zero_centered_gamma), NUM_STAGES=num_stages, num_warps=num_warps,
    )


@torch.library.custom_op("optimized::rmsnorm_bwd_add_", mutates_args=("rg",), device_types="cuda")
def rmsnorm_bwd_add_(
    dy: torch.Tensor,
    x: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    rg: torch.Tensor,
    zero_centered_gamma: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:
    N = x.shape[-1]
    assert dy.is_contiguous() and x.is_contiguous() and rg.is_contiguous() and rstd.is_contiguous()
    assert x.dtype == torch.bfloat16 and dy.dtype == torch.bfloat16 and rg.dtype == torch.bfloat16
    assert rstd.dtype == torch.float32 and gamma.numel() == N
    assert dy.numel() == x.numel() == rg.numel() and rstd.numel() * N == x.numel()
    dgamma = torch.empty(N, dtype=gamma.dtype, device=x.device)
    _launch(dy.view(-1, N), x.view(-1, N), rstd, gamma.contiguous(), rg.view(-1, N), dgamma,
            zero_centered_gamma, sm_margin)
    return dgamma


@rmsnorm_bwd_add_.register_fake
def _(dy, x, rstd, gamma, rg, zero_centered_gamma=False, sm_margin=0):
    return gamma.new_empty(x.shape[-1])
