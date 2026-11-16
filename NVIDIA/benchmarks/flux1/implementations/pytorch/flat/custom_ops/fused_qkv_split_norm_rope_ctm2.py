# SPDX-License-Identifier: Apache-2.0
"""torch.ops.flux.qkv_split_norm_rope_ctm: QKV split + QK RMSNorm + RoPE forward, CUTLASS DSL.

256 threads per CTA, [16 rows x 128] head tiles, the next unit prefetched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

ROWS = 16                 # rows per tile (a row group)
DH = 128                  # head dim, fixed by the model
CHUNK = 8                 # bf16 per thread per row (16 B)
CHUNKS = DH // CHUNK      # 16 threads per row
THREADS = ROWS * CHUNKS   # 256
EPS = 1e-6
_RN = cute.math.RoundingMode.NEAREST_EVEN
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class Mirror:
    """Which product Triton's lowering fused, per operand.

    fuse16_*: first butterfly level of tl.sum(x*x) is fma(x, x, t_partner) (True) or add.
    rope_cos_*: the RoPE combine fuses the cos product (True) or the sin product (False).
    """

    fuse16_q: bool = True
    fuse16_k: bool = True
    rope_cos_q: bool = True
    rope_cos_k: bool = True


#: Triton 3.8.0's lowering, read from its PTX; not a tuning knob.
MIRROR = Mirror(fuse16_q=True, fuse16_k=True, rope_cos_q=True, rope_cos_k=True)


@dataclass(frozen=True)
class Config:
    """Launch shape.

    hp: heads per unit, 1 or 2 (16-byte loads in flight per thread = 2 * 3 * hp).
    min_blocks: launch(min_blocks_per_mp=), the register cap and the residency the grid assumes.
    """

    hp: int = 2
    min_blocks: int = 2


#: Two heads per unit, the next unit prefetched, two CTAs of 256 threads per SM.
CONFIG = Config()

#: CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, percent.
SMEM_CARVEOUT_PCT = 50


# Device-side helpers: traced inline, plain Python control flow only (no range_constexpr).


def _mul(a, b):
    return cute.math.mul(a, b, rounding=_RN)


def _add(a, b):
    return cute.math.add(a, b, rounding=_RN)


def _fma(a, b, c):
    return cute.math.fma(a, b, c, rounding=_RN)


def _row_sum_sq(x: list, lane, fuse16: bool):
    """tl.sum(x * x, axis=0) over a 128-column row held by 16 lanes, in Triton's order.

    Lane c holds Triton lanes 8(c % 4)..8(c % 4) + 7 of warp c // 4: level 16 is c ^ 2, level
    8 is c ^ 1, levels 4/2/1 are lane-local, the cross-warp combine is c ^ 8 then c ^ 4.
    """
    t = [_mul(x[e], x[e]) for e in range(CHUNK)]
    a = []
    for e in range(CHUNK):
        tr = cute.arch.shuffle_sync_bfly(t[e], 2)
        if fuse16:
            a.append(_fma(x[e], x[e], tr))
        else:
            a.append(_add(t[e], tr))
    b = [_add(a[e], cute.arch.shuffle_sync_bfly(a[e], 1)) for e in range(CHUNK)]
    w0 = _add(b[0], b[4])
    w1 = _add(b[1], b[5])
    w2 = _add(b[2], b[6])
    w3 = _add(b[3], b[7])
    x0 = _add(w0, w2)
    x1 = _add(w1, w3)
    y = _add(x0, x1)
    s = _add(y, cute.arch.shuffle_sync_bfly(y, 8))
    tot = _add(s, cute.arch.shuffle_sync_bfly(s, 4))
    return cute.arch.shuffle_sync(tot, lane & 16)


def _rstd(ss):
    """tl.math.rsqrt(ss / Dh + EPS) as Triton lowers it."""
    q = cute.math.div(ss, cutlass.Float32(float(DH)), full=True)
    return cute.math.rsqrt(_add(q, cutlass.Float32(EPS)), approx=True, ftz=True)


def _rope(n: list, cs: list, sn: list, fuse_cos: bool) -> list:
    """n*cos + sign*n[d^1]*sin with Triton's fused/rounded product roles."""
    out = []
    for e in range(CHUNK):
        p = e ^ 1
        # sign * n_p: exact (a negation) for even e, the value itself for odd e
        sp = -n[p] if e % 2 == 0 else n[p]
        if fuse_cos:
            out.append(_fma(n[e], cs[e], _mul(sp, sn[e])))
        else:
            out.append(_fma(sp, sn[e], _mul(n[e], cs[e])))
    return out


def _to_f32(vec) -> list:
    return [vec[e].to(cutlass.Float32) for e in range(CHUNK)]


def _to_bf16(vals: list):
    return cutlass.Vector.from_elements(
        tuple(v.to(cutlass.BFloat16) for v in vals), cutlass.BFloat16)


def _load_unit(u, r, c8, UPG: int, HP: int, D: int, stride_qkv_l, qkv_p):
    """The 3 * HP 16-byte loads of unit u (head-fastest), ld.global.nc.

    Returns six vectors so the loop-carried set has a fixed shape; at HP == 1 the second
    triple aliases the first.
    """
    g = u // UPG
    hp0 = (u - g * UPG) * HP
    rg = g * ROWS + r
    base = rg * stride_qkv_l + c8 + hp0 * DH
    q0 = (qkv_p + base).load(16, count=CHUNK, is_invariant=True)
    k0 = (qkv_p + (base + D)).load(16, count=CHUNK, is_invariant=True)
    v0 = (qkv_p + (base + 2 * D)).load(16, count=CHUNK, is_invariant=True)
    if HP == 2:
        q1 = (qkv_p + (base + DH)).load(16, count=CHUNK, is_invariant=True)
        k1 = (qkv_p + (base + D + DH)).load(16, count=CHUNK, is_invariant=True)
        v1 = (qkv_p + (base + 2 * D + DH)).load(16, count=CHUNK, is_invariant=True)
    else:
        q1, k1, v1 = q0, k0, v0
    return q0, k0, v0, q1, k1, v1


@cute.kernel
def qkv_split_norm_rope_ctm3_kernel(
    qkv: cute.Tensor, q_out: cute.Tensor, k_out: cute.Tensor, v_out: cute.Tensor,
    q_rstd: cute.Tensor, k_rstd: cute.Tensor,
    qn_w: cute.Tensor, kn_w: cute.Tensor, freqs: cute.Tensor,
    L: cutlass.Int32, stride_qkv_l: cutlass.Int32, freq_stride_s: cutlass.Int32,
    pos_offset: cutlass.Int32, out_stride_b: cutlass.Int32, n_units: cutlass.Int32,
    H: cutlass.Constexpr[int], HP: cutlass.Constexpr[int],
    FUSE16_Q: cutlass.Constexpr[bool], FUSE16_K: cutlass.Constexpr[bool],
    ROPE_COS_Q: cutlass.Constexpr[bool], ROPE_COS_K: cutlass.Constexpr[bool],
) -> None:
    D: cutlass.Constexpr[int] = H * DH
    UPG: cutlass.Constexpr[int] = H // HP  # units per row group
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    lane = tidx & 31
    r = tidx >> 4
    c = tidx & 15
    c8 = c * CHUNK

    qkv_p = cutlass.Pointer(qkv.iterator, dtype=cutlass.BFloat16)
    q_p = cutlass.Pointer(q_out.iterator, dtype=cutlass.BFloat16)
    k_p = cutlass.Pointer(k_out.iterator, dtype=cutlass.BFloat16)
    v_p = cutlass.Pointer(v_out.iterator, dtype=cutlass.BFloat16)
    qr_p = cutlass.Pointer(q_rstd.iterator, dtype=cutlass.Float32)
    kr_p = cutlass.Pointer(k_rstd.iterator, dtype=cutlass.Float32)
    f_p = cutlass.Pointer(freqs.iterator, dtype=cutlass.Float32)

    # norm weights: fp32 [2][128] in smem, re-read per unit through a volatile load so
    # LLVM cannot hoist them into 16 live registers; threads t and t+128 write the same slot
    s_w = cute.arch.alloc_smem(cutlass.Float32, 2 * DH, 16)
    swp = cutlass.Pointer(s_w, dtype=cutlass.Float32)
    iw = tidx & (DH - 1)
    (swp + iw).store(qn_w.iterator[iw].to(cutlass.Float32))
    (swp + (DH + iw)).store(kn_w.iterator[iw].to(cutlass.Float32))
    cute.arch.sync_threads()

    # persistent, contiguous, head-fastest unit range of this CTA
    n64 = n_units.to(cutlass.Int64)
    g64 = gdim.to(cutlass.Int64)
    t0 = ((n64 * bidx.to(cutlass.Int64)) // g64).to(cutlass.Int32)
    t1 = ((n64 * (bidx + 1).to(cutlass.Int64)) // g64).to(cutlass.Int32)
    g0 = t0 // UPG
    g1 = (t1 - 1) // UPG + 1

    # the first unit's loads, issued before the loop; every later unit is loaded
    # one iteration ahead of its math (the last iteration re-reads itself, clamped)
    q0, k0, v0, q1, k1, v1 = _load_unit(t0, r, c8, UPG, HP, D, stride_qkv_l, qkv_p)

    for g in cutlass.range(g0, g1, 1):
        rg = g * ROWS + r
        bb = rg // L
        pos = rg - bb * L
        # cos/sin of this row's angles, once per row group
        fv = (f_p + ((pos + pos_offset) * freq_stride_s + c8)).load(
            16, count=CHUNK, is_invariant=True)
        cs = []
        sn = []
        for e in cutlass.range_constexpr(CHUNK):
            cs.append(cute.math.cos(fv[e]))
            sn.append(cute.math.sin(fv[e]))
        out_row = bb * out_stride_b + pos * D + c8
        u_lo = cutlass.max(t0, g * UPG)
        u_hi = cutlass.min(t1, (g + 1) * UPG)
        for u in cutlass.range(u_lo, u_hi, 1):
            # prefetch: the next unit's loads go out before any math
            nq0, nk0, nv0, nq1, nk1, nv1 = _load_unit(
                cutlass.min(u + 1, t1 - 1), r, c8, UPG, HP, D, stride_qkv_l, qkv_p)
            hp0 = (u - g * UPG) * HP
            qv = [q0, q1]
            kv = [k0, k1]
            vv = [v0, v1]
            for j in cutlass.range_constexpr(HP):
                h = hp0 + j
                qf = _to_f32(qv[j])
                kf = _to_f32(kv[j])
                q_r = _rstd(_row_sum_sq(qf, lane, FUSE16_Q))
                k_r = _rstd(_row_sum_sq(kf, lane, FUSE16_K))
                wqv = (swp + c8).load(32, count=CHUNK, is_volatile=True)
                wkv = (swp + (DH + c8)).load(32, count=CHUNK, is_volatile=True)
                qn = [_mul(_mul(qf[e], q_r), wqv[e]) for e in range(CHUNK)]
                kn = [_mul(_mul(kf[e], k_r), wkv[e]) for e in range(CHUNK)]
                qo = _to_bf16(_rope(qn, cs, sn, ROPE_COS_Q))
                ko = _to_bf16(_rope(kn, cs, sn, ROPE_COS_K))
                o = out_row + h * DH
                (q_p + o).store(qo, alignment=16)
                (k_p + o).store(ko, alignment=16)
                (v_p + o).store(vv[j], alignment=16)
                if c == 0:
                    (qr_p + (rg * H + h)).store(q_r)
                    (kr_p + (rg * H + h)).store(k_r)
            q0 = nq0
            k0 = nk0
            v0 = nv0
            q1 = nq1
            k1 = nk1
            v1 = nv1


@cute.jit
def qkv_split_norm_rope_ctm3_host(
    qkv: cute.Tensor, q_out: cute.Tensor, k_out: cute.Tensor, v_out: cute.Tensor,
    q_rstd: cute.Tensor, k_rstd: cute.Tensor,
    qn_w: cute.Tensor, kn_w: cute.Tensor, freqs: cute.Tensor,
    L: cutlass.Int32, stride_qkv_l: cutlass.Int32, freq_stride_s: cutlass.Int32,
    pos_offset: cutlass.Int32, out_stride_b: cutlass.Int32, n_units: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    H: cutlass.Constexpr[int], HP: cutlass.Constexpr[int], MB: cutlass.Constexpr[int],
    FUSE16_Q: cutlass.Constexpr[bool], FUSE16_K: cutlass.Constexpr[bool],
    ROPE_COS_Q: cutlass.Constexpr[bool], ROPE_COS_K: cutlass.Constexpr[bool],
) -> None:
    qkv_split_norm_rope_ctm3_kernel(
        qkv, q_out, k_out, v_out, q_rstd, k_rstd, qn_w, kn_w, freqs,
        L, stride_qkv_l, freq_stride_s, pos_offset, out_stride_b, n_units,
        H, HP, FUSE16_Q, FUSE16_K, ROPE_COS_Q, ROPE_COS_K,
    ).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream,
             min_blocks_per_mp=MB, preferred_smem_carveout=SMEM_CARVEOUT_PCT)


# Host side

_compiled: dict[tuple, Callable] = {}
_sms: dict[int, int] = {}


def _compile(device_index: int, H: int, cfg: Config, mirror: Mirror) -> Callable:
    """Compile once per (device, H, hp, min_blocks, mirror); every size stays symbolic."""
    key = (device_index, H, cfg.hp, cfg.min_blocks, mirror)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            def bf16():
                return make_fake_compact_tensor(
                    cutlass.BFloat16, (cute.sym_int64(divisibility=8),), assumed_align=16)

            def f32():
                return make_fake_compact_tensor(
                    cutlass.Float32, (cute.sym_int64(divisibility=4),), assumed_align=16)

            _compiled[key] = cute.compile(
                qkv_split_norm_rope_ctm3_host,
                bf16(), bf16(), bf16(), bf16(), f32(), f32(), bf16(), bf16(), f32(),
                *([cutlass.Int32(0)] * 7), make_fake_stream(),
                H, cfg.hp, cfg.min_blocks,
                mirror.fuse16_q, mirror.fuse16_k, mirror.rope_cos_q, mirror.rope_cos_k,
                options="--enable-tvm-ffi")
    return _compiled[key]


def _flat(t: torch.Tensor, n: int) -> torch.Tensor:
    """A 1-D view of n elements starting at t's first element."""
    return t.as_strided((n,), (1,), t.storage_offset())


def _check_out(o: torch.Tensor, name: str, B: int, L: int, H: int, sb: int) -> None:
    if o.dtype != torch.bfloat16 or tuple(o.shape) != (B, L, H, DH):
        raise ValueError(f"{name}: expected bf16 [{B}, {L}, {H}, {DH}], got {o.dtype} {tuple(o.shape)}")
    if o.stride(3) != 1 or o.stride(2) != DH or o.stride(1) != H * DH or o.stride(0) != sb:
        raise ValueError(f"{name}: strides must be ({sb}, {H * DH}, {DH}, 1), got {o.stride()}")
    if o.data_ptr() % 16:
        raise ValueError(f"{name}: data pointer must be 16-byte aligned")


def grid_size(device_index: int, B: int, L: int, H: int, config: Config = CONFIG) -> int:
    """The persistent grid, min(units, SMs * min_blocks)."""
    if device_index not in _sms:
        _sms[device_index] = torch.cuda.get_device_properties(device_index).multi_processor_count
    n_units = (B * L // ROWS) * (H // config.hp)
    return max(1, min(n_units, _sms[device_index] * config.min_blocks))


def _launch(
    qkv: torch.Tensor, H: int, qn_weight: torch.Tensor, kn_weight: torch.Tensor,
    freqs: torch.Tensor, pos_offset: int,
    out_q: torch.Tensor, out_k: torch.Tensor, out_v: torch.Tensor,
    q_rstd: torch.Tensor, k_rstd: torch.Tensor, *,
    config: Config = CONFIG, mirror: Mirror = MIRROR, max_ctas: int | None = None,
) -> int:
    """Validate, compile (cached) and launch. Returns the grid size."""
    if config.hp not in (1, 2):
        raise ValueError(f"qkv_split_norm_rope_ctm: HP must be 1 or 2, got {config.hp}")
    if not 1 <= config.min_blocks <= 8:
        raise ValueError(f"qkv_split_norm_rope_ctm: min_blocks must be in 1..8, got {config.min_blocks}")
    if not (qkv.is_cuda and qkv.dtype == torch.bfloat16 and qkv.dim() == 3):
        raise ValueError("qkv_split_norm_rope_ctm: qkv must be a CUDA bf16 [B, L, 3*D] tensor")
    B, L, three_D = qkv.shape
    D = H * DH
    if three_D != 3 * D:
        raise ValueError(f"qkv_split_norm_rope_ctm: head dim must be {DH} (got 3*D={three_D}, H={H})")
    if qkv.stride(2) != 1 or qkv.stride(1) < three_D or qkv.stride(1) % CHUNK:
        raise ValueError("qkv_split_norm_rope_ctm: qkv rows must be contiguous, 16-byte aligned")
    if qkv.stride(0) != L * qkv.stride(1):
        raise ValueError("qkv_split_norm_rope_ctm: qkv batch stride must be L * row stride")
    if qkv.data_ptr() % 16:
        raise ValueError("qkv_split_norm_rope_ctm: qkv must be 16-byte aligned")
    if (B * L) % ROWS:
        raise ValueError(f"qkv_split_norm_rope_ctm requires (B*L) % {ROWS} == 0, got B*L={B * L}")
    if H % config.hp:
        raise ValueError(f"qkv_split_norm_rope_ctm: H={H} must be a multiple of HP={config.hp}")
    for w, name in ((qn_weight, "qn_weight"), (kn_weight, "kn_weight")):
        if w.dtype != torch.bfloat16 or w.numel() != DH or not w.is_contiguous() or w.data_ptr() % 16:
            raise ValueError(f"qkv_split_norm_rope_ctm: {name} must be a contiguous bf16 [{DH}]")
    if (freqs.dtype != torch.float32 or freqs.stride(-1) != 1 or freqs.size(-1) != DH
            or freqs.stride(0) % 4 or freqs.size(0) < pos_offset + L or freqs.data_ptr() % 16):
        raise ValueError("qkv_split_norm_rope_ctm: freqs must be fp32 [S >= pos_offset+L, ..., 128]")
    sb = out_q.stride(0)
    for o, name in ((out_q, "out_q"), (out_k, "out_k"), (out_v, "out_v")):
        _check_out(o, name, B, L, H, sb)
    if sb < L * D or sb % CHUNK:
        raise ValueError(f"qkv_split_norm_rope_ctm: out batch stride {sb} must be >= L*D and a multiple of 8")
    for t, name in ((q_rstd, "q_rstd"), (k_rstd, "k_rstd")):
        if (t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != B * L * H
                or t.data_ptr() % 16):
            raise ValueError(f"qkv_split_norm_rope_ctm: {name} must be contiguous, 16-byte aligned fp32 [B*L, H]")
    n_qkv = B * L * qkv.stride(1)
    n_out = (B - 1) * sb + L * D
    n_freq = (freqs.size(0) - 1) * freqs.stride(0) + DH
    if max(n_qkv, n_out, n_freq, B * L * H) > _INT32_MAX:
        raise ValueError("qkv_split_norm_rope_ctm: tensors exceed the Int32 index range")

    dev = qkv.device.index
    n_units = (B * L // ROWS) * (H // config.hp)
    grid = max(1, min(n_units, max_ctas)) if max_ctas else grid_size(dev, B, L, H, config)
    fn = _compile(dev, H, config, mirror)
    fn(_flat(qkv, n_qkv), _flat(out_q, n_out), _flat(out_k, n_out), _flat(out_v, n_out),
       q_rstd.view(-1), k_rstd.view(-1), qn_weight, kn_weight, _flat(freqs, n_freq),
       L, qkv.stride(1), freqs.stride(0), pos_offset, sb, n_units, grid,
       cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream))
    return grid


def qkv_split_norm_rope_ctm(
    qkv: torch.Tensor, H: int, qn_weight: torch.Tensor, kn_weight: torch.Tensor,
    freqs: torch.Tensor, pos_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV split + QK RMSNorm + RoPE forward (CUTLASS DSL).

    Same signature and semantics as flux.qkv_split_norm_rope: returns RoPE'd Q and K, raw V
    and the two rstd tensors; pre-norm Q/K are not written (the backward re-reads qkv).
    """
    B, L, _ = qkv.shape
    q = torch.empty(B, L, H, DH, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(B, L, H, DH, dtype=qkv.dtype, device=qkv.device)
    v = torch.empty(B, L, H, DH, dtype=qkv.dtype, device=qkv.device)
    q_rstd = torch.empty(B * L, H, dtype=torch.float32, device=qkv.device)
    k_rstd = torch.empty(B * L, H, dtype=torch.float32, device=qkv.device)
    _launch(qkv, H, qn_weight, kn_weight, freqs, pos_offset, q, k, v, q_rstd, k_rstd)
    return q, k, v, q_rstd, k_rstd


def qkv_split_norm_rope_inplace_ctm(
    qkv: torch.Tensor, H: int, qn_weight: torch.Tensor, kn_weight: torch.Tensor,
    freqs: torch.Tensor, pos_offset: int,
    out_q: torch.Tensor, out_k: torch.Tensor, out_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place variant: write Q/K/V into caller-provided [B, L, H, 128] views.

    The views may be sequence slices of a larger [B, L_total, H, 128] buffer.
    Returns (q_rstd, k_rstd), both [B*L, H] fp32.
    """
    B, L, _ = qkv.shape
    q_rstd = torch.empty(B * L, H, dtype=torch.float32, device=qkv.device)
    k_rstd = torch.empty(B * L, H, dtype=torch.float32, device=qkv.device)
    _launch(qkv, H, qn_weight, kn_weight, freqs, pos_offset, out_q, out_k, out_v, q_rstd, k_rstd)
    return q_rstd, k_rstd


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "qkv_split_norm_rope_ctm(Tensor qkv, int H, Tensor qn_weight,"
    " Tensor kn_weight, Tensor freqs, int pos_offset)"
    " -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
_lib.define(
    "qkv_split_norm_rope_inplace_ctm(Tensor qkv, int H, Tensor qn_weight,"
    " Tensor kn_weight, Tensor freqs, int pos_offset,"
    " Tensor out_q, Tensor out_k, Tensor out_v) -> (Tensor, Tensor)"
)


@torch.library.impl(_lib, "qkv_split_norm_rope_ctm", "CUDA")
def _qkv_split_norm_rope_ctm_impl(qkv, H, qn_weight, kn_weight, freqs, pos_offset):
    return qkv_split_norm_rope_ctm(qkv, H, qn_weight, kn_weight, freqs, pos_offset)


@torch.library.impl(_lib, "qkv_split_norm_rope_inplace_ctm", "CUDA")
def _qkv_split_norm_rope_inplace_ctm_impl(qkv, H, qn_weight, kn_weight, freqs, pos_offset,
                                          out_q, out_k, out_v):
    return qkv_split_norm_rope_inplace_ctm(qkv, H, qn_weight, kn_weight, freqs, pos_offset,
                                           out_q, out_k, out_v)
