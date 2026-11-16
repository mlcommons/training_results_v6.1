# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""``torch.ops.flux`` boundary for the optimizer-path weight amax.

Two ops:

==================================  =========================================
op                                  what it does
==================================  =========================================
``flux.shard_copy_partial_amax``    ``prepare_ag_overlap``'s master(f32) →
                                    BF16 shard copy with a per-quantized-
                                    weight partial-max epilogue. One read,
                                    one write — the copy production already
                                    pays (``flat/comm_overlap.py:1077``) —
                                    plus atomics into ``partial``.
``flux.weight_amax_reduce``         ``[shard_size, n_slots] → [n_slots]``
                                    max over the all-gathered partials,
                                    ``grid=(1,)`` — one SM by construction.
==================================  =========================================

The partial max is ``max |bf16_rn(x)|`` — the max of the **converted** BF16
values, because the forward quantizes the BF16 ``param_buffer`` views and
bitwise equality with ``tex.compute_amax`` demands it. Weights carry no RHT
(2D quantization), so this plain max is the NVFP4 ``S_enc`` amax for both
halves.

Segment tables map each :data:`BLOCK_ELTS`-element kernel block of the flat
shard to the quantized weights it covers (:func:`build_segment_tables`).
Blocks are element-granular because ``param_align == 1`` packs biases and
norm vectors between weights and bucket seams fall mid-block.

Indexing inside the kernel is int64: the shard is the whole parameter set
(11.9 G elements) over ``shard_size``, past ``INT32_MAX`` at ``shard_size=4``.

The kernels are the CUTLASS-DSL module ``weight_amax_copy_ctm.py``
(the only implementation). The hand-written CUDA
pair it replaced (``weight_amax_copy_kernel.cu``) was deleted once the port had
shipped: the contract's references are ``aten copy_`` and torch's
``max|bf16_rn(x)|``, not the CUDA kernel, so the swap test had no work left.
:func:`live_kernels` states which kernel symbols the ops launch, for by-name
launch checks.

CUDA graphs: both ops allocate nothing and never touch the host; the caller
owns and zeroes every buffer, so capture and replay are safe.
"""

from __future__ import annotations

import torch
import torch.library

BLOCK_ELTS = 8192
"""Elements per kernel CTA. Must match ``weight_amax_copy_ctm.BLOCK_ELTS``."""

MAX_SEGMENTS = 4
"""Most quantized segments one CTA may cover. Must match ``weight_amax_copy_ctm.MAX_SEGMENTS``."""

_ctm_mod = None


def _ctm():
    """Import (once) the DSL module ``flat.custom_ops.weight_amax_copy_ctm``."""
    global _ctm_mod
    if _ctm_mod is None:
        from flat.custom_ops import weight_amax_copy_ctm as _m
        _ctm_mod = _m
    return _ctm_mod


def live_kernels() -> tuple[str, str]:
    """``(copy symbol, reduce symbol)`` substrings the two ops launch.

    The ops' own statement of what they do, for by-name launch checks.
    Grid rule: ``ceil(n / BLOCK_ELTS)`` CTAs of 256
    threads for the copy, ``(1,)`` of 512 for the reduce.
    """
    m = _ctm()
    return m.COPY_KERNEL, m.REDUCE_KERNEL


def build_segment_tables(
    spans: list[tuple[int, int, int]],
    total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map each kernel block of the flat shard to its quantized segments.

    Args:
        spans: ``(slot, lo, hi)`` per quantized-weight slice of this rank's
            shard, in flat shard-buffer element coordinates —
            non-overlapping; ``hi`` exclusive. A weight split across ranks
            contributes one span per rank; a weight split across a bucket
            seam cannot happen (a param lives in one bucket).
        total: Elements in the flat shard buffer (``master_buf.numel()``).
        device: CUDA device for the tables.

    Returns:
        ``(seg_offset, seg_slot, seg_lo, seg_hi)`` int32 tensors on
        ``device``: CSR-style — block ``b``'s segments are entries
        ``seg_offset[b] : seg_offset[b+1]``, with ``seg_lo``/``seg_hi`` in
        block-local coordinates ``[0, BLOCK_ELTS]``.

    Raises:
        ValueError: If any block would carry more than :data:`MAX_SEGMENTS`
            segments (the kernel's fixed register budget), or spans are out
            of range or overlapping.
    """
    n_blocks = (total + BLOCK_ELTS - 1) // BLOCK_ELTS
    blk, slot_l, lo_l, hi_l = [], [], [], []
    prev_hi = 0
    for slot, lo, hi in sorted(spans, key=lambda s: s[1]):
        if not (0 <= lo < hi <= total):
            raise ValueError(f"span ({slot}, {lo}, {hi}) outside [0, {total})")
        if lo < prev_hi:
            raise ValueError(f"span ({slot}, {lo}, {hi}) overlaps its "
                             f"predecessor (ends at {prev_hi})")
        prev_hi = hi
        b0, b1 = lo // BLOCK_ELTS, (hi - 1) // BLOCK_ELTS
        b = torch.arange(b0, b1 + 1, dtype=torch.int64)
        base = b * BLOCK_ELTS
        blk.append(b)
        slot_l.append(torch.full_like(b, slot))
        lo_l.append((lo - base).clamp_(min=0))
        hi_l.append((hi - base).clamp_(max=BLOCK_ELTS))
    if blk:
        blk = torch.cat(blk)
        seg_slot = torch.cat(slot_l)
        seg_lo = torch.cat(lo_l)
        seg_hi = torch.cat(hi_l)
        order = torch.argsort(blk, stable=True)
        blk, seg_slot = blk[order], seg_slot[order]
        seg_lo, seg_hi = seg_lo[order], seg_hi[order]
        counts = torch.bincount(blk, minlength=n_blocks)
        worst = int(counts.max())
        if worst > MAX_SEGMENTS:
            raise ValueError(
                f"a block carries {worst} quantized segments, over the "
                f"kernel's MAX_SEGMENTS={MAX_SEGMENTS}; the >=9.4 M-element "
                f"weight population cannot produce this — check the spans."
            )
    else:
        seg_slot = seg_lo = seg_hi = torch.empty(0, dtype=torch.int64)
        counts = torch.zeros(n_blocks, dtype=torch.int64)
    seg_offset = torch.zeros(n_blocks + 1, dtype=torch.int64)
    torch.cumsum(counts, 0, out=seg_offset[1:])
    return (
        seg_offset.to(torch.int32).to(device),
        seg_slot.to(torch.int32).to(device),
        seg_lo.to(torch.int32).to(device),
        seg_hi.to(torch.int32).to(device),
    )


def shard_copy_partial_amax(
    src: torch.Tensor,
    dst: torch.Tensor,
    partial: torch.Tensor,
    seg_offset: torch.Tensor,
    seg_slot: torch.Tensor,
    seg_lo: torch.Tensor,
    seg_hi: torch.Tensor,
) -> None:
    """Fused master → BF16 shard copy + per-weight partial amax.

    Args:
        src: f32 flat master shard.
        dst: BF16 destination, same numel — receives ``bf16_rn(src)``,
            bitwise what ``dst.copy_(src)`` writes.
        partial: ``[n_slots]`` f32, **zeroed by the caller** — the epilogue is
            an integer atomic max on the f32 bit pattern, sound only against
            zeros and non-negative values.
        seg_offset: From :func:`build_segment_tables`.
        seg_slot: From :func:`build_segment_tables`.
        seg_lo: From :func:`build_segment_tables`.
        seg_hi: From :func:`build_segment_tables`.
    """
    try:
        _ctm().shard_copy_partial_amax_ctm(
            src, dst, partial, seg_offset, seg_slot, seg_lo, seg_hi,
        )
    except ValueError as e:
        # the op's contract: a guard is an error, never a fallback (the
        # RuntimeError a TORCH_CHECK would raise)
        raise RuntimeError(f"shard_copy_partial_amax: {e}") from e


def weight_amax_reduce(
    gathered: torch.Tensor, out: torch.Tensor, shard_size: int,
) -> None:
    """Reduce the all-gathered partials: ``[shard_size, n_slots] → [n_slots]``.

    Args:
        gathered: f32, ``shard_size * n_slots`` elements, rank-major — the
            layout ``all_gather_into_tensor_out`` produces.
        out: ``[n_slots]`` f32; fully overwritten, no zeroing needed.
        shard_size: Ranks in the shard group (``hier_topology.py:103``).
    """
    try:
        _ctm().weight_amax_reduce_ctm(gathered, out, shard_size)
    except ValueError as e:
        raise RuntimeError(f"weight_amax_reduce: {e}") from e


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "shard_copy_partial_amax(Tensor src, Tensor(a!) dst, Tensor(b!) partial,"
    " Tensor seg_offset, Tensor seg_slot, Tensor seg_lo, Tensor seg_hi) -> ()"
)
_lib.define(
    "weight_amax_reduce(Tensor gathered, Tensor(a!) out, int shard_size) -> ()"
)


@torch.library.impl(_lib, "shard_copy_partial_amax", "CUDA")
def _shard_copy_partial_amax_impl(
    src: torch.Tensor,
    dst: torch.Tensor,
    partial: torch.Tensor,
    seg_offset: torch.Tensor,
    seg_slot: torch.Tensor,
    seg_lo: torch.Tensor,
    seg_hi: torch.Tensor,
) -> None:
    shard_copy_partial_amax(src, dst, partial, seg_offset, seg_slot, seg_lo,
                            seg_hi)


@torch.library.impl(_lib, "weight_amax_reduce", "CUDA")
def _weight_amax_reduce_impl(
    gathered: torch.Tensor, out: torch.Tensor, shard_size: int,
) -> None:
    weight_amax_reduce(gathered, out, shard_size)
