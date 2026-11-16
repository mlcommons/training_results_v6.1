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

"""Triton kernels: the EP=1 MoE routing metadata in two launches, replacing HybridEP's planner chain.

At EP=1 the dispatch is a local permutation, yet each MoE layer ran HybridEP's planner (one-rank all-gather, memsets,
hybrid_ep::scan, pad_tokens_per_expert), Megatron's over_budget mask ops, the inverse-map scatter, _padded_counts and
TE's splits_to_offsets scan: 17 graph nodes per layer.  gpt_oss::moe_routing_meta emits all of it from the bool map:
tokens_per_expert (int32, drop-adjusted), padded_counts, dense_to_expert_map, dense_chunk_layout, overflow_flag,
row_to_token [P] (token of row p, -1 on padding), permuted_probs [P] and TE's six splits_to_offsets_multi tensors;
has_ob ORs the overflow flag into Megatron's over_budget byte.  Semantics follow HybridEP's scan with token drop: expert
e's rows start at sum_{j<e} round_up(count_j, pad) of the raw counts, token t takes start_e + (tokens < t routed to e),
rows >= P are dropped and overflow = start_{E-1} + count_{E-1} > P.  row_to_token is in token space; the inverse-map op
on HybridEP's dense map yields dense indices, equal iff every token is routed (n_dense == T, required by the hook's
first-call check).  _meta_hist_kernel counts per BT-token block; _meta_main_kernel re-reduces the partials for its
prefix and writes its tokens' maps, the padding rows of its slice of [0, P) and (program 0) the counts, offsets and
flags.  No autotune (CUDA-graph captured); kernel_fn / reference_fn / get_inputs are the standalone check entry points.
"""
import os
from typing import List, Tuple

import torch
import triton
import triton.language as tl

BT = int(os.environ.get("MOE_ROUTING_META_BT", "128"))          # tokens per program (both kernels)
RBC = 128                                                     # padding-row chunk per loop iteration
NUM_WARPS = int(os.environ.get("MOE_ROUTING_META_NUM_WARPS", "4"))


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# =============================================================================================
# kernels
# =============================================================================================
@triton.jit
def _meta_hist_kernel(map_ptr, part_ptr, T_valid,
                      E: tl.constexpr, EP: tl.constexpr, BT: tl.constexpr):
    """partials[b, e] = tokens of block b routed to expert e; partials[b, EP] = tokens of block b routed anywhere."""
    b = tl.program_id(0)
    t = b * BT + tl.arange(0, BT)
    e = tl.arange(0, EP)
    tmask = t < T_valid
    emask = e < E
    mp = tl.load(map_ptr + t.to(tl.int64)[:, None] * E + e[None, :], mask=tmask[:, None] & emask[None, :], other=0)
    mi = (mp != 0).to(tl.int32)
    cnt = tl.sum(mi, axis=0)                                      # [EP]
    needed = (tl.sum(mi, axis=1) > 0).to(tl.int32)                # [BT]
    tl.store(part_ptr + b * (EP + 1) + e, cnt)
    tl.store(part_ptr + b * (EP + 1) + EP, tl.sum(needed, axis=0))


@triton.jit
def _meta_main_kernel(map_ptr, probs_ptr, part_ptr,
                      tpe_ptr, padded_ptr, d2e_ptr, layout_ptr, ovf_ptr, ob_ptr,
                      r2t_ptr, pprob_ptr, sp_ptr, base_ptr, o1_ptr, o2_ptr, o3_ptr, o4_ptr,
                      T_valid, T_rows, P, NB, n_chunks, probs_rows, s1, s2, s3, s4,
                      E: tl.constexpr, EP: tl.constexpr, BT: tl.constexpr, PAD: tl.constexpr, CHUNK: tl.constexpr,
                      HAS_PROBS: tl.constexpr, HAS_OB: tl.constexpr, RB: tl.constexpr, RBC: tl.constexpr):
    b = tl.program_id(0)
    e = tl.arange(0, EP)
    emask = e < E

    # ---- 1. totals over all programs and the exclusive prefix over the programs before this one
    tot = tl.zeros([EP], dtype=tl.int32)
    prev = tl.zeros([EP], dtype=tl.int32)
    tot_nv = tl.zeros([32], dtype=tl.int32)
    prev_nv = tl.zeros([32], dtype=tl.int32)
    for i0 in range(0, NB, 32):
        bi = i0 + tl.arange(0, 32)
        bm = bi < NB
        tile = tl.load(part_ptr + bi[:, None] * (EP + 1) + e[None, :], mask=bm[:, None] & emask[None, :], other=0)
        tot += tl.sum(tile, axis=0)
        prev += tl.sum(tl.where((bi < b)[:, None], tile, 0), axis=0)
        nn = tl.load(part_ptr + bi * (EP + 1) + EP, mask=bm, other=0)
        tot_nv += nn
        prev_nv += tl.where(bi < b, nn, 0)
    tot_n = tl.sum(tot_nv, axis=0)
    prev_n = tl.sum(prev_nv, axis=0)

    # ---- 2. HybridEP's expert layout: starts from the RAW padded counts, token drop against P
    padded_raw = ((tot + (PAD - 1)) // PAD) * PAD
    starts = tl.cumsum(padded_raw, axis=0) - padded_raw          # exclusive prefix [EP]
    end_raw = starts + tot
    tpe = tl.where(end_raw <= P, tot, tl.where(starts < P, P - starts, 0))
    tpe = tl.where(emask, tpe, 0)
    padded = ((tpe + (PAD - 1)) // PAD) * PAD
    n_used = tl.sum(padded, axis=0)
    ovf = tl.sum(tl.where(e == E - 1, end_raw, 0), axis=0) > P

    if b == 0:
        tl.store(tpe_ptr + e, tpe, mask=emask)
        padded64 = padded.to(tl.int64)
        tl.store(padded_ptr + e, padded64, mask=emask)
        incl = tl.cumsum(padded64, axis=0)
        tl.store(sp_ptr + e, incl.to(tl.int32), mask=emask)
        z1 = tl.arange(0, 1)
        zero64 = tl.zeros([1], dtype=tl.int64)
        tl.store(base_ptr + z1, zero64)
        tl.store(base_ptr + 1 + e, incl, mask=emask)
        tl.store(o1_ptr + z1, zero64)
        tl.store(o1_ptr + 1 + e, incl * s1, mask=emask)
        tl.store(o2_ptr + z1, zero64)
        tl.store(o2_ptr + 1 + e, incl * s2, mask=emask)
        tl.store(o3_ptr + z1, zero64)
        tl.store(o3_ptr + 1 + e, incl * s3, mask=emask)
        tl.store(o4_ptr + z1, zero64)
        tl.store(o4_ptr + 1 + e, incl * s4, mask=emask)
        tl.store(layout_ptr + (n_chunks - 1) + z1, tot_n + tl.zeros([1], dtype=tl.int32))
        ovf_i = ovf.to(tl.int32) + tl.zeros([1], dtype=tl.int32)
        tl.store(ovf_ptr + z1, ovf_i)
        if HAS_OB:
            ob = tl.load(ob_ptr + z1)
            tl.store(ob_ptr + z1, ((ob != 0) | (ovf_i != 0)).to(tl.uint8))

    # ---- 3. this program's tokens: dense index, per-expert rank, map rows, chunk layout, row_to_token / probs
    t = b * BT + tl.arange(0, BT)
    t64 = t.to(tl.int64)
    tmask = t < T_valid
    m2 = tmask[:, None] & emask[None, :]
    mp = tl.load(map_ptr + t64[:, None] * E + e[None, :], mask=m2, other=0)
    mi = (mp != 0).to(tl.int32)
    needed_i = (tl.sum(mi, axis=1) > 0).to(tl.int32)             # [BT]
    d = prev_n + tl.cumsum(needed_i, axis=0) - needed_i           # dense index of each token [BT]
    rank = tl.cumsum(mi, axis=0) - mi                             # tokens before t routed to e, within the block
    row = (starts + prev)[None, :] + rank                         # [BT, EP]
    routed = mi != 0
    ok = routed & (row < P)
    entry = tl.where(ok, row, -1)
    tl.store(d2e_ptr + d.to(tl.int64)[:, None] * E + e[None, :], entry, mask=(needed_i > 0)[:, None] & emask[None, :])
    c = t // CHUNK
    is_first = ((t % CHUNK) == 0) & (c > 0) & (c < n_chunks) & (t < T_rows)
    tl.store(layout_ptr + (c - 1), d, mask=is_first)
    row64 = row.to(tl.int64)
    tl.store(r2t_ptr + row64, tl.broadcast_to(t[:, None], (BT, EP)), mask=ok)
    if HAS_PROBS:
        pr = tl.load(probs_ptr + t64[:, None] * E + e[None, :], mask=ok & (t[:, None] < probs_rows), other=0.0)
        tl.store(pprob_ptr + row64, pr, mask=ok)

    # ---- 4. padding rows of this program's slice of [0, P): -1 / 0 (alignment padding inside a group, and the tail)
    ends_real = starts + tpe
    ends_pad = starts + padded
    for r0 in range(0, RB, RBC):
        p = b * RB + r0 + tl.arange(0, RBC)
        pm = p < P
        inpad = (p[:, None] >= ends_real[None, :]) & (p[:, None] < ends_pad[None, :]) & emask[None, :]
        is_pad = (tl.sum(inpad.to(tl.int32), axis=1) > 0) | (p >= n_used)
        wm = pm & is_pad
        tl.store(r2t_ptr + p, tl.full([RBC], -1, tl.int32), mask=wm)
        if HAS_PROBS:
            tl.store(pprob_ptr + p, tl.zeros([RBC], dtype=tl.float32), mask=wm)


# =============================================================================================
# torch op
# =============================================================================================
_META_OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _shapes(T_slots: int, E: int, P: int, chunk: int):
    NB = triton.cdiv(max(T_slots, 1), BT)
    n_chunks = (T_slots - 1) // chunk + 1
    RB = ((triton.cdiv(P, NB) + RBC - 1) // RBC) * RBC
    return NB, n_chunks, RB


def _alloc(T_slots: int, E: int, P: int, chunk: int, device):
    NB, n_chunks, RB = _shapes(T_slots, E, P, chunk)
    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    return (torch.empty((E,), **i32), torch.empty((E,), **i64), torch.empty((T_slots, E), **i32),
            torch.empty((n_chunks,), **i32), torch.empty((1,), **i32), torch.empty((P,), **i32),
            torch.empty((P,), dtype=torch.float32, device=device), torch.empty((E,), **i32),
            torch.empty((E + 1,), **i64), torch.empty((E + 1,), **i64), torch.empty((E + 1,), **i64),
            torch.empty((E + 1,), **i64), torch.empty((E + 1,), **i64))


@torch.library.custom_op("gpt_oss::moe_routing_meta", mutates_args=("over_budget",))
def moe_routing_meta(routing_map: torch.Tensor, probs: torch.Tensor, over_budget: torch.Tensor, T_slots: int, P: int,
                     pad_multiple: int, chunk: int, has_probs: bool, has_ob: bool, strides: List[int]) -> _META_OUT:
    assert routing_map.dim() == 2 and routing_map.dtype == torch.bool and routing_map.is_contiguous(), \
        (routing_map.shape, routing_map.dtype)
    T, E = routing_map.shape
    assert T_slots >= T and T_slots % 16 == 0, (T, T_slots)
    assert E >= 1 and pad_multiple > 0 and P >= 0 and P % pad_multiple == 0 and chunk > 0, (E, pad_multiple, P, chunk)
    assert len(strides) == 4, strides
    if has_probs:
        assert probs.dim() == 2 and probs.shape[1] == E and probs.dtype == torch.float32 and probs.is_contiguous(), \
            (probs.shape, probs.dtype)
    if has_ob:
        assert over_budget.dtype == torch.uint8 and over_budget.numel() == 1 and over_budget.is_cuda, over_budget
    dev = routing_map.device
    EP = _pow2(E)
    NB, n_chunks, RB = _shapes(T_slots, E, P, chunk)
    outs = _alloc(T_slots, E, P, chunk, dev)
    tpe, padded, d2e, layout, ovf, r2t, pprob, sp, base, o1, o2, o3, o4 = outs
    part = torch.empty((NB, EP + 1), dtype=torch.int32, device=dev)
    map_u8 = routing_map.view(torch.uint8)
    _meta_hist_kernel[(NB,)](map_u8, part, T, E=E, EP=EP, BT=BT, num_warps=NUM_WARPS)
    _meta_main_kernel[(NB,)](
        map_u8, probs if has_probs else map_u8, part,
        tpe, padded, d2e, layout, ovf, over_budget if has_ob else ovf,
        r2t, pprob, sp, base, o1, o2, o3, o4,
        T, T_slots, P, NB, n_chunks, probs.shape[0] if has_probs else 0,
        int(strides[0]), int(strides[1]), int(strides[2]), int(strides[3]),
        E=E, EP=EP, BT=BT, PAD=pad_multiple, CHUNK=chunk, HAS_PROBS=has_probs, HAS_OB=has_ob, RB=RB, RBC=RBC,
        num_warps=NUM_WARPS)
    return outs


@moe_routing_meta.register_fake
def _(routing_map, probs, over_budget, T_slots, P, pad_multiple, chunk, has_probs, has_ob, strides):
    T, E = routing_map.shape
    return _alloc(T_slots, E, P, chunk, routing_map.device)


# =============================================================================================
# torch reference (HybridEP's scan semantics) and the fixed-name contract
# =============================================================================================
def routing_meta_reference(routing_map, probs, T_slots, P, pad_multiple, chunk, strides, over_budget=None):
    """Same outputs as gpt_oss::moe_routing_meta, in plain torch (dense_to_expert_map rows >= n_dense are -1)."""
    T, E = routing_map.shape
    dev = routing_map.device
    m = torch.zeros((T_slots, E), dtype=torch.bool, device=dev)
    m[:T] = routing_map
    mi = m.to(torch.int64)
    tot = mi.sum(dim=0)                                                        # raw counts [E]
    padded_raw = (tot + pad_multiple - 1) // pad_multiple * pad_multiple
    starts = torch.cumsum(padded_raw, 0) - padded_raw
    end_raw = starts + tot
    tpe = torch.where(end_raw <= P, tot, torch.where(starts < P, P - starts, torch.zeros_like(tot)))
    padded = (tpe + pad_multiple - 1) // pad_multiple * pad_multiple
    ovf = (end_raw[E - 1] > P).to(torch.int32).reshape(1)
    needed = mi.sum(dim=1) > 0
    d = torch.cumsum(needed.to(torch.int64), 0) - needed.to(torch.int64)      # dense index of each token
    n_dense = int(needed.sum().item())
    rank = torch.cumsum(mi, 0) - mi
    row = starts[None, :] + rank
    ok = m & (row < P)
    entry = torch.where(ok, row, torch.full_like(row, -1)).to(torch.int32)
    d2e = torch.full((T_slots, E), -1, dtype=torch.int32, device=dev)
    d2e[d[needed]] = entry[needed]
    n_chunks = (T_slots - 1) // chunk + 1
    layout = torch.empty((n_chunks,), dtype=torch.int32, device=dev)
    firsts = torch.arange(1, n_chunks, device=dev) * chunk
    layout[:-1] = d[firsts].to(torch.int32)
    layout[-1] = n_dense
    r2t = torch.full((P,), -1, dtype=torch.int32, device=dev)
    pprob = torch.zeros((P,), dtype=torch.float32, device=dev)
    tok = torch.arange(T_slots, device=dev)[:, None].expand(T_slots, E)
    r2t[row[ok]] = tok[ok].to(torch.int32)
    if probs is not None:
        pf = torch.zeros((T_slots, E), dtype=torch.float32, device=dev)
        pf[: probs.shape[0]] = probs.float()
        pprob[row[ok]] = pf[ok]
    incl = torch.cumsum(padded, 0)
    sp = incl.to(torch.int32)
    base = torch.cat([torch.zeros((1,), dtype=torch.int64, device=dev), incl])
    offs = [base * int(s) for s in strides]
    if over_budget is not None:
        over_budget |= (ovf != 0)
    return (tpe.to(torch.int32), padded.to(torch.int64), d2e, layout, ovf, r2t, pprob, sp, base,
            offs[0], offs[1], offs[2], offs[3])


def make_routing_map(num_tokens, E, topk, device="cuda", seed=0, drop_frac=0.0, skew=0.0):
    """Synthetic bool routing map: top-k experts per token; drop_frac of the slots cleared (some tokens may end up
    routed nowhere), skew > 0 concentrates the picks on the low experts (to exercise the token-drop limit)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.ones(E)
    if skew > 0:
        w = torch.exp(-skew * torch.arange(E).float())
    picks = torch.multinomial(w.expand(num_tokens, E), topk, replacement=False, generator=g)        # [T, topk]
    m = torch.zeros((num_tokens, E), dtype=torch.bool)
    m.scatter_(1, picks, True)
    if drop_frac > 0:
        keep = torch.rand(num_tokens, E, generator=g) >= drop_frac
        m &= keep
    probs = torch.rand(num_tokens, E, generator=g).float()
    probs = torch.where(m, probs, torch.zeros_like(probs))
    return m.to(device), probs.to(device)


def kernel_fn(routing_map, probs, P, pad_multiple, chunk, strides):
    T, E = routing_map.shape
    T_slots = (max(T, 16) + 15) // 16 * 16
    dummy = torch.zeros((1,), dtype=torch.uint8, device=routing_map.device)
    outs = torch.ops.gpt_oss.moe_routing_meta(routing_map, probs, dummy, T_slots, P, pad_multiple, chunk, True, False,
                                              list(strides))
    tpe, padded, d2e, layout, ovf, r2t, pprob, sp, base, o1, o2, o3, o4 = outs
    rows = torch.arange(T_slots, device=routing_map.device)[:, None]
    d2e = torch.where(rows < layout[-1], d2e, torch.full_like(d2e, -1))      # rows >= n_dense are undefined in both
    return tpe, padded, d2e, layout, ovf, r2t, pprob, sp, base, o1, o2, o3, o4


def reference_fn(routing_map, probs, P, pad_multiple, chunk, strides):
    T, E = routing_map.shape
    T_slots = (max(T, 16) + 15) // 16 * 16
    return routing_meta_reference(routing_map, probs, T_slots, P, pad_multiple, chunk, strides)


def get_inputs(num_tokens=24576, E=32, topk=4, pad_multiple=256, chunk=64, capacity=1.2, strides=(2880, 5760, 2880, 2880)):
    m, probs = make_routing_map(num_tokens, E, topk, drop_frac=0.02)
    P = int(num_tokens * topk * capacity)
    P += -P % pad_multiple
    return [m, probs, P, pad_multiple, chunk, list(strides)]


if __name__ == "__main__":
    for name, kw in (("workload", dict()), ("skewed-drop", dict(skew=0.35)), ("small-odd", dict(num_tokens=1000, E=8, topk=2))):
        nt = kw.pop("num_tokens", 24576)
        E = kw.pop("E", 32)
        topk = kw.pop("topk", 4)
        m, probs = make_routing_map(nt, E, topk, drop_frac=0.02, **kw)
        P = int(nt * topk * 1.2)
        P += -P % 256
        inp = [m, probs, P, 256, 64, [2880, 5760, 2880, 2880]]
        got = kernel_fn(*inp)
        ref = reference_fn(*inp)
        names = ["tpe", "padded", "d2e", "layout", "ovf", "r2t", "pprob", "sp", "base", "o1", "o2", "o3", "o4"]
        res = {n: bool(torch.equal(a, b)) for n, a, b in zip(names, got, ref)}
        print(name, "all bitwise:", all(res.values()), res, "ovf=", int(ref[4].item()), "tpe=", ref[0].tolist()[:8], "...")
