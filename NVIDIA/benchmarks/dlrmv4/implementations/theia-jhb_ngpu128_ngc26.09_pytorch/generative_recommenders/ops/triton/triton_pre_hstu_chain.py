
# pyre-unsafe

"""One Triton launch for the pre-HSTU forward chain of the compacted path.

``triton_pre_hstu_chain`` writes x0 = dropout(pos_ts_embed(alpha * (contextual
++ sum of three compacted components))) once, in place of four launches:
``_compacted_sum_jagged_concat_3_fwd``, ``concat_2D_jagged`` on the
timestamps (the kernel also emits that column, the output postprocessor reads
it), ``_add_timestamp_position_embeddings_kernel`` and ``torch.native_dropout``.
Every rounding point of that chain is kept in the same dtype and order, so x0
is bitwise the original's, and the dropout mask is bitwise
``torch.native_dropout``'s (see ``_native_dropout_philox_params``). The
backward reuses the existing kernels: the tiled position / time table scatter
of ``triton_position`` and the segmented / one-hot component reductions of
``triton_compacted_sum_jagged_concat_3``. The dropout backward belongs to the
first HSTU layer's Function, which receives the mask and hands back the
gradient of the pre-dropout activation.
"""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from generative_recommenders.ops.triton.triton_compacted_sum_jagged_concat_3 import (
    _REDUCTION_CHUNK_SIZE,
    _build_counting_segment_reduction_plans,
    _extract_dense_grad,
    _SegmentReductionPlan,
    _segmented_reduce_sequence_grad,
    _small_domain_route_reduce_sequence_grad,
)
from generative_recommenders.ops.triton.triton_position import _add_embeddings_bwd_tiled
from generative_recommenders.ops.utils import maybe_register_custom_op

_DROPOUT_BLOCK = 256
_DROPOUT_UNROLL = 4
_DROPOUT_VEC = 8

_BLOCK_N = 8
_BLOCK_D = 64
_NUM_WARPS = 2
_D_MIN, _D_MAX = 256, 1024


@triton.jit(do_not_specialize=["L", "philox_seed", "philox_offset4", "threads_total"])
def _pre_hstu_chain_fwd_kernel(
    Dense,
    Comp0,
    Comp1,
    Comp2,
    Rev0,
    Rev1,
    Rev2,
    SeqOffsets,
    TS,
    NumTargets,
    PosEmb,
    TsEmb,
    Out,
    Mask,
    PosInds,
    TsInds,
    TsOut,
    L,
    B,
    stride_db,
    stride_dl,
    stride_dd,
    stride_pn,
    stride_tn,
    num_time_buckets,
    time_bucket_increments,
    time_bucket_scale,
    time_delta,
    max_pos_ind,
    philox_seed,
    philox_offset4,
    threads_total,
    p_keep,
    dropout_scale,
    D: tl.constexpr,
    C: tl.constexpr,
    SEARCH_ITERS: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program: BLOCK_N consecutive rows of the [L, D] token grid."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = rows < L
    rows64 = rows.to(tl.int64)

    lo = tl.zeros([BLOCK_N], dtype=tl.int32)
    hi = tl.zeros([BLOCK_N], dtype=tl.int32) + (B - 1)
    for _ in tl.static_range(SEARCH_ITERS):
        mid = (lo + hi + 1) // 2
        start_mid = tl.load(SeqOffsets + mid).to(tl.int64) + mid.to(tl.int64) * C
        go_right = start_mid <= rows64
        lo = tl.where(go_right, mid, lo)
        hi = tl.where(go_right, hi, mid - 1)
    b = lo
    seq_start = tl.load(SeqOffsets + b).to(tl.int64)
    seq_end = tl.load(SeqOffsets + b + 1).to(tl.int64)
    seq_len = seq_end - seq_start
    out_start = seq_start + b.to(tl.int64) * C
    n = (rows64 - out_start).to(tl.int32)
    total_len = (seq_len + C).to(tl.int32)
    is_dense = n < C
    seq_pos = n - C
    dense_mask = row_valid & is_dense
    seq_mask = row_valid & (seq_pos >= 0)
    occ = seq_start + seq_pos.to(tl.int64)

    occ32 = occ.to(tl.int32)
    rev0 = tl.load(Rev0 + occ32, mask=seq_mask, other=0).to(tl.int32)
    rev1 = tl.load(Rev1 + occ32, mask=seq_mask, other=0).to(tl.int32)
    rev2 = tl.load(Rev2 + occ32, mask=seq_mask, other=0).to(tl.int32)
    ts_raw = tl.load(TS + occ32, mask=seq_mask, other=0)
    query_time = tl.load(
        TS + (seq_end - 1).to(tl.int32), mask=row_valid & (seq_len > 0), other=0
    )
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + b, mask=row_valid, other=0).to(tl.int32)

    if HAS_MULTIPLE_TARGETS:
        if INTERLEAVE_TARGETS:
            high_ind = total_len - num_targets * 2
        else:
            high_ind = total_len - num_targets
    else:
        high_ind = total_len
    pos_inds = tl.where(n < high_ind, n, high_ind)
    pos_inds = high_ind - pos_inds + C
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(n < C, n, pos_inds)
    tl.store(PosInds + rows, pos_inds, mask=row_valid)

    tl.store(TsOut + rows, ts_raw, mask=row_valid)
    ts = query_time - ts_raw + time_delta
    ts = tl.where(ts > 1e-6, ts, 1e-6) / time_bucket_increments
    if TIME_BUCKET_FN == "log":
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    tl.store(TsInds + rows, ts, mask=row_valid)

    for j in tl.static_range(D // BLOCK_D):
        d = j * BLOCK_D + tl.arange(0, BLOCK_D)
        dense_values = tl.load(
            Dense + b[:, None] * stride_db + n[:, None] * stride_dl + d[None, :] * stride_dd,
            mask=dense_mask[:, None],
            other=0.0,
        )
        sequence_values = (
            tl.load(Comp0 + rev0[:, None] * D + d[None, :], mask=seq_mask[:, None], other=0.0)
            + tl.load(Comp1 + rev1[:, None] * D + d[None, :], mask=seq_mask[:, None], other=0.0)
            + tl.load(Comp2 + rev2[:, None] * D + d[None, :], mask=seq_mask[:, None], other=0.0)
        )
        seq_emb = dense_values + sequence_values
        seq_emb = (seq_emb.to(tl.float32) * alpha).to(seq_emb.dtype)
        pos_emb = tl.load(
            PosEmb + pos_inds[:, None] * stride_pn + d[None, :], mask=row_valid[:, None], other=0.0
        )
        ts_emb = tl.load(
            TsEmb + ts[:, None] * stride_tn + d[None, :], mask=row_valid[:, None], other=0.0
        )
        y = seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype)
        out_ptrs = Out + rows[:, None] * D + d[None, :]
        if HAS_DROPOUT:
            m2 = tl.arange(0, BLOCK_D // 8)
            g = rows[:, None] * (D // 8) + (j * (BLOCK_D // 8) + m2[None, :])
            idx = g % threads_total
            k = g // threads_total
            cx0 = philox_offset4 + (k * 2).to(tl.int64)
            cx1 = cx0 + 1
            ctr_x = tl.join((cx0 & 0xFFFFFFFF).to(tl.int32), (cx1 & 0xFFFFFFFF).to(tl.int32))
            ctr_y = tl.join((cx0 >> 32).to(tl.int32), (cx1 >> 32).to(tl.int32))
            ctr_z = tl.join(idx, idx)
            ctr_w = tl.zeros_like(ctr_z)
            r0, r1, r2, r3 = tl.philox(philox_seed, ctr_x, ctr_y, ctr_z, ctr_w)
            u0 = r0.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
            u1 = r1.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
            u2 = r2.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
            u3 = r3.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
            keep = tl.join(
                tl.join((u0 < p_keep).to(tl.int8), (u2 < p_keep).to(tl.int8)),
                tl.join((u1 < p_keep).to(tl.int8), (u3 < p_keep).to(tl.int8)),
            )
            keep = tl.reshape(keep, [BLOCK_N, BLOCK_D])
            x0 = (y.to(tl.float32) * keep.to(tl.float32) * dropout_scale).to(y.dtype)
            tl.store(Mask + rows[:, None] * D + d[None, :], keep.to(tl.uint8), mask=row_valid[:, None])
            tl.store(out_ptrs, x0, mask=row_valid[:, None])
        else:
            tl.store(out_ptrs, y, mask=row_valid[:, None])


def _native_dropout_philox_params(nelem: int, device: torch.device) -> Tuple[int, int, int]:
    """(seed, offset, threads_total) as ``torch.native_dropout`` would draw
    them for a contiguous 16-bit tensor of ``nelem`` elements on ``device``,
    advancing the device's default generator by the same counter_offset, so
    the RNG stream after the fused launch is where native_dropout would have
    left it."""
    props = torch.cuda.get_device_properties(device)
    blocks_per_sm = props.max_threads_per_multi_processor // _DROPOUT_BLOCK
    grid_x = min(
        props.multi_processor_count * blocks_per_sm,
        (nelem + _DROPOUT_BLOCK - 1) // _DROPOUT_BLOCK,
    )
    counter_offset = (
        (nelem - 1) // (_DROPOUT_BLOCK * grid_x * _DROPOUT_UNROLL) + 1
    ) * _DROPOUT_UNROLL
    index = device.index if device.index is not None else torch.cuda.current_device()
    gen = torch.cuda.default_generators[index]
    seed = gen.initial_seed()
    if seed >= 2**63:
        seed -= 2**64
    offset = gen.get_offset()
    if offset % 4 != 0:
        raise RuntimeError("the CUDA generator's Philox offset is not a multiple of 4")
    gen.set_offset(offset + counter_offset)
    return seed, offset, grid_x * _DROPOUT_BLOCK


def _check_rows(name: str, t: torch.Tensor, d: int) -> None:
    if t.dim() != 2 or t.shape[1] != d or t.stride(1) != 1 or t.stride(0) != d:
        raise ValueError(f"{name} must be a contiguous [rows, {d}] tensor")
    if t.shape[0] * d >= 2**31:
        raise ValueError(f"{name} is too large for int32 addressing")


@maybe_register_custom_op("generative_recommenders::pre_hstu_chain_fwd", mutates_args=())
def _pre_hstu_chain_fwd(
    dense: torch.Tensor,
    component_0: torch.Tensor,
    component_1: torch.Tensor,
    component_2: torch.Tensor,
    reverse_0: torch.Tensor,
    reverse_1: torch.Tensor,
    reverse_2: torch.Tensor,
    seq_offsets: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    total_rows: int,
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: float,
    dropout_p: float,
    philox_seed: int,
    philox_offset: int,
    threads_total: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x0 [L, D], mask bool [L, D] ([0] when dropout_p == 0), pos_inds int32
    [L], ts_inds int32 [L], concatenated timestamps [L]). Raises on operands
    the kernel does not take instead of falling back."""
    d = component_0.shape[1]
    for name, t in (
        ("component_0", component_0),
        ("component_1", component_1),
        ("component_2", component_2),
    ):
        _check_rows(name, t, d)
        if t.dtype != component_0.dtype:
            raise ValueError("all components must share one dtype")
    m = reverse_0.numel()
    if reverse_1.numel() != m or reverse_2.numel() != m or timestamps.numel() != m:
        raise ValueError("component routes and timestamps must align with the sequence")
    for t in (reverse_0, reverse_1, reverse_2, timestamps):
        if not t.is_contiguous():
            raise ValueError("routes and timestamps must be contiguous")
    bsz = seq_offsets.numel() - 1
    if dense.dim() != 3 or dense.size(0) != bsz or dense.size(2) != d or dense.stride(2) != 1:
        raise ValueError("dense must have shape [B, C, D] with unit column stride")
    c = dense.size(1)
    if total_rows != m + bsz * c:
        raise ValueError("total_rows must equal M + B * C")
    for name, t in (("pos_embeddings", pos_embeddings), ("ts_embeddings", ts_embeddings)):
        if t.dim() != 2 or t.shape[1] != d or t.stride(1) != 1:
            raise ValueError(f"{name} must be [rows, D] with unit column stride")
        if t.stride(0) * t.shape[0] >= 2**31:
            raise ValueError(f"{name} is too large for int32 addressing")
    if d & (d - 1) or d < _D_MIN or d > _D_MAX:
        raise ValueError(f"D must be a power of two in [{_D_MIN}, {_D_MAX}]")
    if total_rows * d >= 2**31:
        raise ValueError("L * D must be below 2^31 (int32 addressing)")
    if dense.numel() and (dense.stride(0) * (bsz - 1) + dense.stride(1) * (c - 1) + d) >= 2**31:
        raise ValueError("dense is too large for int32 addressing")
    has_dropout = dropout_p > 0.0
    if has_dropout and component_0.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("the dropout mask reproduction expects 16-bit components")
    out = torch.empty((total_rows, d), dtype=component_0.dtype, device=component_0.device)
    mask = torch.empty(
        (total_rows, d) if has_dropout else (0,), dtype=torch.bool, device=out.device
    )
    pos_inds = torch.empty((total_rows,), dtype=torch.int32, device=out.device)
    ts_inds = torch.empty((total_rows,), dtype=torch.int32, device=out.device)
    ts_out = torch.empty((total_rows,), dtype=timestamps.dtype, device=out.device)
    if total_rows == 0:
        return out, mask, pos_inds, ts_inds, ts_out
    p_keep_f32 = torch.tensor(1.0 - dropout_p, dtype=torch.float32)
    p_keep = float(p_keep_f32)
    dropout_scale = float(torch.tensor(1.0, dtype=torch.float32) / p_keep_f32) if has_dropout else 1.0
    _pre_hstu_chain_fwd_kernel[(triton.cdiv(total_rows, _BLOCK_N),)](
        dense,
        component_0,
        component_1,
        component_2,
        reverse_0,
        reverse_1,
        reverse_2,
        seq_offsets,
        timestamps,
        num_targets if num_targets is not None else seq_offsets,
        pos_embeddings,
        ts_embeddings,
        out,
        mask.view(torch.uint8) if has_dropout else out,
        pos_inds,
        ts_inds,
        ts_out,
        total_rows,
        bsz,
        dense.stride(0),
        dense.stride(1),
        dense.stride(2),
        pos_embeddings.stride(0),
        ts_embeddings.stride(0),
        ts_embeddings.shape[0] - 1,
        60.0,
        1.0,
        0,
        pos_embeddings.shape[0],
        philox_seed,
        philox_offset // 4,
        threads_total,
        p_keep,
        dropout_scale,
        D=d,
        C=c,
        SEARCH_ITERS=max(1, (bsz - 1).bit_length()),
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        INTERLEAVE_TARGETS=interleave_targets,
        TIME_BUCKET_FN=time_bucket_fn,
        HAS_DROPOUT=has_dropout,
        alpha=alpha,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=_BLOCK_D,
        num_warps=_NUM_WARPS,
    )
    return out, mask, pos_inds, ts_inds, ts_out


@_pre_hstu_chain_fwd.register_fake
def _pre_hstu_chain_fwd_fake(
    dense,
    component_0,
    component_1,
    component_2,
    reverse_0,
    reverse_1,
    reverse_2,
    seq_offsets,
    timestamps,
    num_targets,
    pos_embeddings,
    ts_embeddings,
    total_rows,
    interleave_targets,
    time_bucket_fn,
    alpha,
    dropout_p,
    philox_seed,
    philox_offset,
    threads_total,
):
    d = component_0.shape[1]
    dev = component_0.device
    return (
        torch.empty((total_rows, d), dtype=component_0.dtype, device=dev),
        torch.empty((total_rows, d) if dropout_p > 0.0 else (0,), dtype=torch.bool, device=dev),
        torch.empty((total_rows,), dtype=torch.int32, device=dev),
        torch.empty((total_rows,), dtype=torch.int32, device=dev),
        torch.empty((total_rows,), dtype=timestamps.dtype, device=dev),
    )


class _PreHSTUChainFunction(torch.autograd.Function):
    """Forward: one launch of _pre_hstu_chain_fwd. Backward: the kernels of
    the two Functions it replaces (_AddTimestampPositionEmbeddingsFunction
    and _CompactedSumJaggedConcat3 in their fused-chain forms). The incoming
    gradient is the first layer's dx, which already carries the dropout mask
    and scale, so no dropout backward runs here."""

    @staticmethod
    def forward(
        ctx,
        dense: torch.Tensor,
        component_0: torch.Tensor,
        component_1: torch.Tensor,
        component_2: torch.Tensor,
        pos_embeddings: torch.Tensor,
        ts_embeddings: torch.Tensor,
        reverse_0: torch.Tensor,
        reverse_1: torch.Tensor,
        reverse_2: torch.Tensor,
        seq_offsets: torch.Tensor,
        timestamps: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        interleave_targets: bool,
        time_bucket_fn: str,
        alpha: float,
        dropout_p: float,
    ):
        component_0 = component_0.contiguous()
        component_1 = component_1.contiguous()
        component_2 = component_2.contiguous()
        pos_embeddings = pos_embeddings.contiguous()
        ts_embeddings = ts_embeddings.contiguous()
        timestamps = timestamps.contiguous()
        bsz = seq_offsets.numel() - 1
        c = dense.size(1)
        m = reverse_0.numel()
        d = component_0.shape[1]
        total_rows = m + bsz * c
        if dropout_p > 0.0:
            seed, offset, threads_total = _native_dropout_philox_params(
                total_rows * d, component_0.device
            )
        else:
            seed, offset, threads_total = 0, 0, 1
        out, mask, pos_inds, ts_inds, ts_out = _pre_hstu_chain_fwd(
            dense,
            component_0,
            component_1,
            component_2,
            reverse_0,
            reverse_1,
            reverse_2,
            seq_offsets,
            timestamps,
            num_targets,
            pos_embeddings,
            ts_embeddings,
            total_rows,
            interleave_targets,
            time_bucket_fn,
            float(alpha),
            float(dropout_p),
            seed,
            offset,
            threads_total,
        )
        values = torch.arange(0, total_rows, dtype=torch.int32, device=out.device)
        try:
            sorted_ts_key_inds, sorted_ts_value_inds = torch.ops.hammer.sort_kv_pairs(
                ts_inds, values
            )
            sorted_pos_key_inds, sorted_pos_value_inds = torch.ops.hammer.sort_kv_pairs(
                pos_inds, values
            )
        except Exception:
            sorted_ts_key_inds, sorted_ts_value_inds = torch.sort(ts_inds)
            sorted_pos_key_inds, sorted_pos_value_inds = torch.sort(pos_inds)
        sequence_lengths = seq_offsets[1:] - seq_offsets[:-1]
        occurrence_batches = torch.repeat_interleave(
            torch.arange(bsz, device=seq_offsets.device), sequence_lengths, output_size=m
        )
        occurrence_grad_rows = (
            torch.arange(m, dtype=torch.int32, device=seq_offsets.device)
            + (occurrence_batches.to(torch.int32) + 1) * c
        )
        plan_0, plan_1 = _build_counting_segment_reduction_plans(
            reverse_0, component_0.size(0), reverse_1, component_1.size(0), occurrence_grad_rows
        )
        ctx.save_for_backward(
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
            seq_offsets,
            reverse_2,
            occurrence_grad_rows,
            *plan_0,
            *plan_1,
        )
        ctx.dense_shape = dense.shape
        ctx.component_rows = (component_0.size(0), component_1.size(0), component_2.size(0))
        ctx.pos_emb_size = pos_embeddings.shape[0]
        ctx.ts_emb_size = ts_embeddings.shape[0]
        ctx.pos_dtype = pos_embeddings.dtype
        ctx.ts_dtype = ts_embeddings.dtype
        ctx.alpha = float(alpha)
        ctx.c = c
        ctx.d = d
        ctx.mark_non_differentiable(mask, ts_out)
        return out, mask, ts_out

    @staticmethod
    def backward(ctx, d_out: torch.Tensor, _d_mask: torch.Tensor, _d_ts_out: torch.Tensor):
        saved = ctx.saved_tensors
        (
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
            seq_offsets,
            reverse_2,
            occurrence_grad_rows,
        ) = saved[:7]
        plan_0 = _SegmentReductionPlan(*saved[7:11])
        plan_1 = _SegmentReductionPlan(*saved[11:15])
        d_out = d_out.contiguous()
        rows_0, rows_1, rows_2 = ctx.component_rows
        d = ctx.d
        bsz = seq_offsets.numel() - 1
        d_pos = torch.zeros((ctx.pos_emb_size, d), device=d_out.device, dtype=torch.float32)
        d_ts = torch.zeros((ctx.ts_emb_size, d), device=d_out.device, dtype=torch.float32)
        d_seq = torch.empty_like(d_out) if ctx.alpha != 1.0 else d_out
        _add_embeddings_bwd_tiled(
            d_out,
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            d_pos,
            d_seq if ctx.alpha != 1.0 else None,
            ctx.alpha,
        )
        _add_embeddings_bwd_tiled(d_out, sorted_ts_key_inds, sorted_ts_value_inds, d_ts, None, 1.0)
        block_d = triton.next_power_of_2(d)
        block_n = 16
        grad_dense = d_seq.new_empty(ctx.dense_shape)
        _extract_dense_grad[(triton.cdiv(ctx.c, block_n), bsz)](
            d_seq,
            seq_offsets,
            grad_dense,
            D=d,
            DENSE_LEN=ctx.c,
            DENSE_FIRST=True,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8 if block_d >= 512 else 4,
        )
        grad_component_0 = d_seq.new_zeros((rows_0, d))
        grad_component_1 = d_seq.new_zeros((rows_1, d))
        for plan, row_count, grad_component in (
            (plan_0, rows_0, grad_component_0),
            (plan_1, rows_1, grad_component_1),
        ):
            _segmented_reduce_sequence_grad(
                grad_output=d_seq,
                plan=plan,
                grad_component=grad_component,
                row_count=row_count,
                chunk_size=_REDUCTION_CHUNK_SIZE,
                block_m=16,
                block_d=block_d,
                pipeline_stages=3,
                num_warps=4,
            )
        if rows_2 <= 16:
            grad_component_2 = _small_domain_route_reduce_sequence_grad(
                d_seq, occurrence_grad_rows, reverse_2, rows_2
            )
        else:
            occurrences = torch.index_select(d_seq, 0, occurrence_grad_rows.to(torch.int64))
            onehot = torch.nn.functional.one_hot(reverse_2.to(torch.int64), num_classes=rows_2)
            grad_component_2 = torch.mm(onehot.to(occurrences.dtype).transpose(0, 1), occurrences)
        return (
            grad_dense,
            grad_component_0,
            grad_component_1,
            grad_component_2,
            d_pos.to(ctx.pos_dtype),
            d_ts.to(ctx.ts_dtype),
        ) + (None,) * 10


@torch.fx.wrap
def triton_pre_hstu_chain(
    *,
    dense: torch.Tensor,
    components: List[torch.Tensor],
    reverse_indices: List[torch.Tensor],
    seq_offsets: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    alpha: float,
    dropout_p: float,
    interleave_targets: bool = False,
    time_bucket_fn: str = "sqrt",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """x0 = dropout(pos_ts_embed(alpha * (dense ++ sum of the compacted
    components))) in one launch. Returns (x0 [L, D], the bool dropout mask
    [L, D] or None when dropout_p == 0, the concatenated timestamps [L]).
    ``dropout_p > 0`` draws from the device's default generator exactly as
    ``torch.native_dropout`` would; the caller hands the mask to the first
    HSTU layer, whose backward applies it."""
    if len(components) != 3 or len(reverse_indices) != 3:
        raise ValueError("the fused pre-HSTU chain covers exactly three compacted components")
    out, mask, ts_out = _PreHSTUChainFunction.apply(
        dense,
        components[0],
        components[1],
        components[2],
        pos_embeddings,
        ts_embeddings,
        reverse_indices[0],
        reverse_indices[1],
        reverse_indices[2],
        seq_offsets,
        timestamps,
        num_targets,
        interleave_targets,
        time_bucket_fn,
        alpha,
        dropout_p,
    )
    return out, (mask if dropout_p > 0.0 else None), ts_out
