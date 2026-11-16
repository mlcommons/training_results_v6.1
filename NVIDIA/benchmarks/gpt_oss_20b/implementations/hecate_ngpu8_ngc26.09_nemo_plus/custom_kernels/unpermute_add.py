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

"""Triton kernels for the MoE unpermute (HybridEP dense layout) with a fused row add, for gpt-oss-20b.
HybridEP's combine at EP=1 is the local unpermute (unpermute_kernel<__nv_bfloat16>, csrc/hybrid_ep/extension/permute.cu):
per dense token t it sums the rows permuted[dense_to_expert_map[t, e]] with map >= 0 (fp32 accumulation in map order,
one bf16 rounding); dense_chunk_layout[-1] is the number of real dense tokens.  Megatron adds the residual (forward) and
autograd the router's input gradient (dispatch backward) in separate bf16 passes; here that row starts the accumulator:
  gpt_oss::unpermute_add(permuted [P, H] bf16, map [>= T, E] int32 (-1 = not routed), layout [n] int32, add [T, H] bf16
      or None, T) -> [T, H] bf16 = add[t] + sum_e permuted[map[t, e]]   (rows t >= dense count: add[t] or 0)
  gpt_oss::unpermute_scale_add(permuted, map, layout, probs [>= T, E] fp32 in dense order, add, T): the same sum with
      the routing probability applied in the accumulator (the fc2 grouped GEMM then emits the unscaled z = acc + b2).
  gpt_oss::probs_grad_gather(grad_permuted_probs [P] fp32, map, layout, T) -> [T, E] fp32 = grad[map[t, e]] or 0
      (HybridEP's with_probs unpermute of the probability gradient, split out once the row unpermute moves to the tap).
Numerics: with add=None the sum order equals HybridEP's, so the output is bitwise identical; with add there is one
rounding instead of two.  No @triton.autotune (the workload runs under a full-iteration CUDA graph): the tile is fixed
at import from RESIDUAL_BIAS_FUSION_BLOCK_T / RESIDUAL_BIAS_FUSION_BLOCK_H / RESIDUAL_BIAS_FUSION_NUM_WARPS.
"""
import os
from typing import Optional

import torch
import triton
import triton.language as tl

# Default tile = the fastest of the measured variants, isolated at T=24576 H=2880 E=32 P=102400: +add 94-95 us, plain
# 83-84 us; KMAX=4 is 170 us; one warp per token (1/1024/1) helps the plain variant (64.0 us) but not the +add one (94.7 us).
# Workload shape per rank: T = s*b = 8192*3 = 24576, H = 2880, E = 32 local experts, top-k 4, P = ceil(1.2 * 4 * T / 256)
# * 256 = 118016 permuted rows (MOE_EXPERT_RANK_CAPACITY_FACTOR=1.2, 256-row alignment).
DEFAULT_BLOCK_T = int(os.environ.get("RESIDUAL_BIAS_FUSION_BLOCK_T", "16"))
DEFAULT_BLOCK_H = int(os.environ.get("RESIDUAL_BIAS_FUSION_BLOCK_H", "256"))
DEFAULT_NUM_WARPS = int(os.environ.get("RESIDUAL_BIAS_FUSION_NUM_WARPS", "4"))
KMAX = 8                 # active experts per token handled (this model: top-k 4)


@triton.jit
def _unpermute_add_kernel(perm_ptr, map_ptr, layout_ptr, add_ptr, out_ptr, probs_ptr, n_chunks, T, H,
                          E: tl.constexpr, KMAX: tl.constexpr, HAS_ADD: tl.constexpr, HAS_PROBS: tl.constexpr,
                          BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr):
    # One program = BLOCK_T consecutive tokens x the whole hidden row (in BLOCK_H chunks).  The map tile [BLOCK_T, E] is
    # compacted: rank = running count of active experts per token, so the k-th gather reads exactly the k-th active
    # source row of every token (HybridEP's warp-level compaction, in tiles).  The per-k compaction is loop-invariant and
    # the compiler hoists it out of the H loop (hoisting it by hand was bitwise-identical and 15% slower, 152.4 vs 132.9 us).
    pid = tl.program_id(0)
    n_dense = tl.load(layout_ptr + n_chunks - 1)             # number of real dense tokens (int32 scalar)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, E)
    rmask = rows < T
    rows64 = rows.to(tl.int64)
    mmask = rmask[:, None] & (cols[None, :] < E)
    mrow = tl.load(map_ptr + rows64[:, None] * E + cols[None, :], mask=mmask, other=-1)
    active = (mrow >= 0) & (rows[:, None] < n_dense) & mmask
    rank = tl.cumsum(active.to(tl.int32), axis=1) - 1
    if HAS_PROBS:
        prow = tl.load(probs_ptr + rows64[:, None] * E + cols[None, :], mask=active, other=0.0).to(tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        hm = offs < H
        om = rmask[:, None] & hm[None, :]
        if HAS_ADD:
            acc = tl.load(add_ptr + rows64[:, None] * H + offs[None, :], mask=om, other=0.0).to(tl.float32)
        else:
            acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(KMAX):
            sel = active & (rank == k)
            src_k = tl.sum(tl.where(sel, mrow, 0), axis=1)                 # [BLOCK_T] source row of the k-th active expert
            has_k = tl.sum(sel.to(tl.int32), axis=1) > 0
            v = tl.load(perm_ptr + src_k.to(tl.int64)[:, None] * H + offs[None, :], mask=has_k[:, None] & hm[None, :], other=0.0)
            if HAS_PROBS:
                p_k = tl.sum(tl.where(sel, prow, 0.0), axis=1)              # [BLOCK_T] routing probability of that expert
                acc += v.to(tl.float32) * p_k[:, None]
            else:
                acc += v.to(tl.float32)
        tl.store(out_ptr + rows64[:, None] * H + offs[None, :], acc.to(out_ptr.dtype.element_ty), mask=om)


@triton.jit
def _probs_grad_gather_kernel(gp_ptr, map_ptr, layout_ptr, out_ptr, n_chunks, T,
                              E: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    n_dense = tl.load(layout_ptr + n_chunks - 1)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, E)
    rm2 = (rows[:, None] < T) & (cols[None, :] < E)
    idx = rows[:, None].to(tl.int64) * E + cols[None, :]
    src = tl.load(map_ptr + idx, mask=rm2, other=-1)
    ok = (src >= 0) & (rows[:, None] < n_dense) & rm2
    v = tl.load(gp_ptr + tl.maximum(src, 0).to(tl.int64), mask=ok, other=0.0)
    tl.store(out_ptr + idx, v, mask=rm2)


def _check_map(dense_to_expert_map: torch.Tensor, dense_chunk_layout: torch.Tensor, num_tokens: int):
    assert dense_to_expert_map.dim() == 2 and dense_to_expert_map.dtype == torch.int32, (
        dense_to_expert_map.shape, dense_to_expert_map.dtype)
    assert dense_to_expert_map.is_contiguous()
    assert dense_to_expert_map.shape[0] >= num_tokens, (dense_to_expert_map.shape, num_tokens)
    assert dense_chunk_layout.dtype == torch.int32 and dense_chunk_layout.numel() >= 1 and dense_chunk_layout.is_contiguous()
    return dense_to_expert_map.shape[1]


@torch.library.custom_op("gpt_oss::unpermute_add", mutates_args=())
def unpermute_add(permuted: torch.Tensor, dense_to_expert_map: torch.Tensor, dense_chunk_layout: torch.Tensor,
                  add: Optional[torch.Tensor], num_tokens: int) -> torch.Tensor:
    assert permuted.dim() == 2 and permuted.dtype == torch.bfloat16 and permuted.is_contiguous(), (permuted.shape, permuted.dtype)
    H = permuted.shape[1]
    assert H % 8 == 0, H
    E = _check_map(dense_to_expert_map, dense_chunk_layout, num_tokens)
    if add is not None:
        assert add.shape == (num_tokens, H) and add.dtype == torch.bfloat16 and add.is_contiguous(), (add.shape, add.dtype)
    out = torch.empty(num_tokens, H, dtype=torch.bfloat16, device=permuted.device)
    if num_tokens == 0:
        return out
    assert E & (E - 1) == 0, E                     # tl.arange over the experts needs a power of two
    grid = (triton.cdiv(num_tokens, DEFAULT_BLOCK_T),)
    _unpermute_add_kernel[grid](
        permuted, dense_to_expert_map, dense_chunk_layout, add if add is not None else permuted, out, permuted,
        dense_chunk_layout.numel(), num_tokens, H,
        E=E, KMAX=KMAX, HAS_ADD=add is not None, HAS_PROBS=False, BLOCK_T=DEFAULT_BLOCK_T, BLOCK_H=DEFAULT_BLOCK_H,
        num_warps=DEFAULT_NUM_WARPS)
    return out


@unpermute_add.register_fake
def _(permuted, dense_to_expert_map, dense_chunk_layout, add, num_tokens):
    return torch.empty(num_tokens, permuted.shape[1], dtype=permuted.dtype, device=permuted.device)


@torch.library.custom_op("gpt_oss::unpermute_scale_add", mutates_args=())
def unpermute_scale_add(permuted: torch.Tensor, dense_to_expert_map: torch.Tensor, dense_chunk_layout: torch.Tensor,
                        probs: torch.Tensor, add: Optional[torch.Tensor], num_tokens: int) -> torch.Tensor:
    """out[t] = add[t] + sum_e probs[t, e] * permuted[map[t, e]]  (fp32 accumulation, one bf16 rounding)."""
    assert permuted.dim() == 2 and permuted.dtype == torch.bfloat16 and permuted.is_contiguous(), (permuted.shape, permuted.dtype)
    H = permuted.shape[1]
    assert H % 8 == 0, H
    E = _check_map(dense_to_expert_map, dense_chunk_layout, num_tokens)
    assert probs.dim() == 2 and probs.shape[0] >= num_tokens and probs.shape[1] == E, (probs.shape, num_tokens, E)
    assert probs.dtype == torch.float32 and probs.is_contiguous(), (probs.dtype, probs.stride())
    if add is not None:
        assert add.shape == (num_tokens, H) and add.dtype == torch.bfloat16 and add.is_contiguous(), (add.shape, add.dtype)
    return _launch_scale_add(permuted, dense_to_expert_map, dense_chunk_layout, probs, add, num_tokens)


def _launch_scale_add(permuted, dense_to_expert_map, dense_chunk_layout, probs, add, num_tokens):
    H = permuted.shape[1]
    E = dense_to_expert_map.shape[1]
    out = torch.empty(num_tokens, H, dtype=torch.bfloat16, device=permuted.device)
    if num_tokens == 0:
        return out
    assert E & (E - 1) == 0, E
    grid = (triton.cdiv(num_tokens, DEFAULT_BLOCK_T),)
    _unpermute_add_kernel[grid](
        permuted, dense_to_expert_map, dense_chunk_layout, add if add is not None else permuted, out, probs,
        dense_chunk_layout.numel(), num_tokens, H,
        E=E, KMAX=KMAX, HAS_ADD=add is not None, HAS_PROBS=True, BLOCK_T=DEFAULT_BLOCK_T,
        BLOCK_H=DEFAULT_BLOCK_H, num_warps=DEFAULT_NUM_WARPS)
    return out


@unpermute_scale_add.register_fake
def _(permuted, dense_to_expert_map, dense_chunk_layout, probs, add, num_tokens):
    return torch.empty(num_tokens, permuted.shape[1], dtype=permuted.dtype, device=permuted.device)


@torch.library.custom_op("gpt_oss::probs_grad_gather", mutates_args=())
def probs_grad_gather(grad_permuted_probs: torch.Tensor, dense_to_expert_map: torch.Tensor,
                      dense_chunk_layout: torch.Tensor, num_tokens: int) -> torch.Tensor:
    assert grad_permuted_probs.dtype == torch.float32 and grad_permuted_probs.is_contiguous()
    E = _check_map(dense_to_expert_map, dense_chunk_layout, num_tokens)
    assert E & (E - 1) == 0, E                     # tl.arange needs a power of two
    out = torch.empty(num_tokens, E, dtype=torch.float32, device=grad_permuted_probs.device)
    if num_tokens == 0:
        return out
    BLOCK_T = max(1, 1024 // E)
    _probs_grad_gather_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
        grad_permuted_probs, dense_to_expert_map, dense_chunk_layout, out, dense_chunk_layout.numel(), num_tokens,
        E=E, BLOCK_T=BLOCK_T, num_warps=4)
    return out


@probs_grad_gather.register_fake
def _(grad_permuted_probs, dense_to_expert_map, dense_chunk_layout, num_tokens):
    return torch.empty(num_tokens, dense_to_expert_map.shape[1], dtype=torch.float32, device=grad_permuted_probs.device)


# =============================================================================================
# Torch references and the kernel_fn / reference_fn / get_inputs verification entry points (synthetic HybridEP-style routing)
# =============================================================================================
def unpermute_add_reference(permuted, dense_to_expert_map, dense_chunk_layout, add, num_tokens):
    """fp32 reference with the same accumulation order (add first, then experts in map order)."""
    H = permuted.shape[1]
    n_dense = int(dense_chunk_layout[-1].item())
    m = dense_to_expert_map[:num_tokens].long()
    acc = torch.zeros(num_tokens, H, dtype=torch.float32, device=permuted.device)
    if add is not None:
        acc += add.float()
    rows = torch.arange(num_tokens, device=permuted.device)
    for e in range(m.shape[1]):
        src = m[:, e]
        ok = (src >= 0) & (rows < n_dense)
        acc += torch.where(ok[:, None], permuted[src.clamp(min=0)].float(), 0.0)
    return acc.to(torch.bfloat16)


def unpermute_scale_add_reference(permuted, dense_to_expert_map, dense_chunk_layout, probs, add, num_tokens):
    """fp32 reference of gpt_oss::unpermute_scale_add (add first, then probs[t, e] * row_e in map order)."""
    H = permuted.shape[1]
    n_dense = int(dense_chunk_layout[-1].item())
    m = dense_to_expert_map[:num_tokens].long()
    acc = torch.zeros(num_tokens, H, dtype=torch.float32, device=permuted.device)
    if add is not None:
        acc += add.float()
    rows = torch.arange(num_tokens, device=permuted.device)
    for e in range(m.shape[1]):
        src = m[:, e]
        ok = (src >= 0) & (rows < n_dense)
        p = torch.where(ok, probs[:num_tokens, e].float(), torch.zeros((), device=permuted.device))
        acc += torch.where(ok[:, None], permuted[src.clamp(min=0)].float() * p[:, None], 0.0)
    return acc.to(torch.bfloat16)


def probs_grad_gather_reference(grad_permuted_probs, dense_to_expert_map, dense_chunk_layout, num_tokens):
    n_dense = int(dense_chunk_layout[-1].item())
    m = dense_to_expert_map[:num_tokens].long()
    rows = torch.arange(num_tokens, device=m.device)[:, None]
    ok = (m >= 0) & (rows < n_dense)
    return torch.where(ok, grad_permuted_probs[m.clamp(min=0)], torch.zeros((), device=m.device, dtype=torch.float32))


def make_routing(num_tokens, E, topk, pad_multiple, device="cuda", seed=0, drop_frac=0.0):
    """Synthetic HybridEP dense layout: per-expert chunks aligned to pad_multiple; map[t, e] = row or -1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    experts = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(num_tokens)])      # [T, topk]
    if drop_frac > 0:
        keep = torch.rand(num_tokens, topk, generator=g) >= drop_frac
        experts = torch.where(keep, experts, torch.full_like(experts, -1))
    counts = torch.zeros(E, dtype=torch.long)
    for e in range(E):
        counts[e] = (experts == e).sum()
    padded = (counts + pad_multiple - 1) // pad_multiple * pad_multiple
    starts = torch.cumsum(padded, 0) - padded
    m = torch.full((num_tokens, E), -1, dtype=torch.int32)
    fill = starts.clone()
    for t in range(num_tokens):
        for e in experts[t].tolist():
            if e >= 0:
                m[t, e] = int(fill[e]); fill[e] += 1
    P = int(padded.sum())
    layout = torch.tensor([num_tokens], dtype=torch.int32)
    return m.to(device), layout.to(device), P


def kernel_fn(permuted, dense_to_expert_map, dense_chunk_layout, add, num_tokens):
    return torch.ops.gpt_oss.unpermute_add(permuted, dense_to_expert_map, dense_chunk_layout, add, num_tokens)


def reference_fn(permuted, dense_to_expert_map, dense_chunk_layout, add, num_tokens):
    return unpermute_add_reference(permuted, dense_to_expert_map, dense_chunk_layout, add, num_tokens)


def get_inputs(num_tokens=2048, H=2880, E=32, topk=4, pad_multiple=256):
    m, layout, P = make_routing(num_tokens, E, topk, pad_multiple, drop_frac=0.05)
    permuted = torch.randn(P, H, device="cuda").to(torch.bfloat16)
    add = torch.randn(num_tokens, H, device="cuda").to(torch.bfloat16)
    return [permuted, m, layout, add, num_tokens]


if __name__ == "__main__":
    inp = get_inputs()
    out = kernel_fn(*inp)
    ref = reference_fn(*inp)
    d = (out.float() - ref.float()).abs().max().item()
    print("unpermute_add max|diff| vs reference:", d, "bitwise:", torch.equal(out, ref))
    gp = torch.randn(inp[0].shape[0], device="cuda")
    o2 = torch.ops.gpt_oss.probs_grad_gather(gp, inp[1], inp[2], inp[4])
    r2 = probs_grad_gather_reference(gp, inp[1], inp[2], inp[4])
    print("probs_grad_gather bitwise:", torch.equal(o2, r2))
    probs = torch.rand(inp[4], inp[1].shape[1], device="cuda", dtype=torch.float32)
    o3 = torch.ops.gpt_oss.unpermute_scale_add(inp[0], inp[1], inp[2], probs, inp[3], inp[4])
    r3 = unpermute_scale_add_reference(inp[0], inp[1], inp[2], probs, inp[3], inp[4])
    d3 = (o3.float() - r3.float()).abs().max().item()
    print("unpermute_scale_add max|diff| vs reference:", d3, "bitwise:", torch.equal(o3, r3))
