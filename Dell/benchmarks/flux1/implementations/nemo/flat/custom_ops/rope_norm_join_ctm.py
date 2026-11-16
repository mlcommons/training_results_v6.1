# SPDX-License-Identifier: Apache-2.0
"""torch.ops.flux.rope_norm_join_ctm: fused RoPE backward + QK RMSNorm backward + QKV join.

256 threads per CTA, a [16 rows x 128] head tile per iteration, bitwise the CUDA kernel.
"""

from __future__ import annotations

import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl.cutlass import CuTeDSL

ROWS = 16                  # rows per tile
DH = 128                   # head dim (fixed by the model)
CHUNK = 8                  # bf16 per thread per row (16 B)
CHUNKS = DH // CHUNK       # 16 threads per row
THREADS = ROWS * CHUNKS    # 256
MIN_CTAS_PER_SM = 2        # __launch_bounds__(256, 2)
#: CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, percent.
SMEM_CARVEOUT_PCT = 50

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32
I64 = cutlass.Int64

# div.full.f32 (__fdividef), Triton's lowering of the / Dh division; the DSL's / is div.rn.f32.
# Raw llvm.inline_asm: inline_ptx gives the float constant an 'n' constraint libNVVM rejects.
_DIV_FULL_ASM = "div.full.f32 $0, $1, $2;"
_DIV_FULL_CONSTRAINTS = "=f,f,f"

#: cos/sin lowering: "math" = cute.math.cos (math dialect); "libdevice" = a direct call
#: to libdevice's __nv_cosf / __nv_sinf declared in the DSL's gpu.module.
COS_MODES = ("math", "libdevice")
_libdevice_declared: dict[int, set] = {}


def _div_full(a, b):
    """__fdividef(a, b) = div.full.f32, both operands in registers."""
    res = llvm.inline_asm(
        T.f32(), [a.ir_value(), b.ir_value()], _DIV_FULL_ASM, _DIV_FULL_CONSTRAINTS,
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return F32(res)


def _libdevice_f32(name: str, x):
    """Call libdevice name(f32) -> f32 from a kernel body.

    Declares llvm.func @name(f32) -> f32 once per gpu.module and emits llvm.call.
    """
    gm = CuTeDSL._get_dsl().gpu_module
    blk = gm.bodyRegion.blocks[0]
    done = _libdevice_declared.setdefault(id(gm), set())
    if name not in done:
        present = False
        for op in blk.operations:
            o = op.operation
            if o.name == "llvm.func" and ir.StringAttr(o.attributes["sym_name"]).value == name:
                present = True
                break
        if not present:
            try:
                decl = ir.Operation.parse(f"llvm.func @{name}(f32) -> f32")
                blk.append(decl.operation)
            except Exception:  # noqa: BLE001 -- older bindings: build the op directly
                llvm.LLVMFuncOp(
                    name, ir.TypeAttr.get(ir.Type.parse("!llvm.func<f32 (f32)>")),
                    linkage=ir.Attribute.parse("#llvm.linkage<external>"),
                    ip=ir.InsertionPoint(blk),
                )
        done.add(name)
    callee = ir.FlatSymbolRefAttr.get(name)
    try:
        res = llvm.call(T.f32(), [x.ir_value()], [], [], callee=callee)
    except TypeError:  # bindings without operand bundles
        res = llvm.call(T.f32(), [x.ir_value()], callee=callee)
    return F32(res)


def _cos(x, mode: str):
    if mode == "libdevice":
        return _libdevice_f32("__nv_cosf", x)
    return cute.math.cos(x)   # math dialect -> libdevice __nv_cosf


def _sin(x, mode: str):
    if mode == "libdevice":
        return _libdevice_f32("__nv_sinf", x)
    return cute.math.sin(x)


def _load_tile(i, H, L, r, c, D, in_stride_b, stride_qkv_l,
               gqp, gkp, gvp, qkvp, rqp, rkp):
    """The seven loads of tile i (head-fastest), __ldg cache path."""
    g = i // H
    h = i - g * H
    rg = g * ROWS + r
    bb = rg // L
    l = rg - bb * L
    col = h * DH + c * CHUNK
    in_off = bb.to(I64) * in_stride_b + (l * D + col).to(I64)
    vq = (gqp + in_off).load(16, count=CHUNK, is_invariant=True)
    vk = (gkp + in_off).load(16, count=CHUNK, is_invariant=True)
    vv = (gvp + in_off).load(16, count=CHUNK, is_invariant=True)
    pre = rg.to(I64) * stride_qkv_l + col.to(I64)
    pq = (qkvp + pre).load(16, count=CHUNK, is_invariant=True)
    pk = (qkvp + (pre + D.to(I64))).load(16, count=CHUNK, is_invariant=True)
    ro = rg * H + h
    rq = (rqp + ro).load(4, is_invariant=True)
    rk = (rkp + ro).load(4, is_invariant=True)
    return vq, vk, vv, pq, pk, rq, rk


def _triton_row_sum(gyw, pre, lane):
    """tl.sum(gyw * pre, axis=0) exactly as Triton 3.8.0 lowered it.

    Products rounded, first butterfly level re-fused as fma(gyw, pre, t_partner), plain
    adds at xor 8/4/2/1, then the four 32-column partials combine as (P0 + P2) + (P1 + P3).
    """
    t = gyw * pre
    u = []
    for e in range(CHUNK):
        tr = cute.arch.shuffle_sync_bfly(t[e], 2)
        u.append(cute.math.fma(gyw[e], pre[e], tr))
    v = []
    for e in range(CHUNK):
        ur = cute.arch.shuffle_sync_bfly(u[e], 1)
        v.append(u[e] + ur)
    w0 = v[0] + v[4]
    w1 = v[1] + v[5]
    w2 = v[2] + v[6]
    w3 = v[3] + v[7]
    x0 = w0 + w2
    x1 = w1 + w3
    y = x0 + x1
    yr = cute.arch.shuffle_sync_bfly(y, 8)
    s = y + yr
    sr = cute.arch.shuffle_sync_bfly(s, 4)
    tot = s + sr
    return cute.arch.shuffle_sync(tot, lane & 16)


@cute.kernel
def rope_norm_join_ctm_kernel(
    gq: cute.Tensor, gk: cute.Tensor, gv: cute.Tensor, qkv: cute.Tensor,
    q_rstd: cute.Tensor, k_rstd: cute.Tensor,
    qn_w: cute.Tensor, kn_w: cute.Tensor, freqs: cute.Tensor,
    out: cute.Tensor, grad_qn_w: cute.Tensor, grad_kn_w: cute.Tensor,
    L: I32, H: I32, in_stride_b: I64, stride_qkv_l: I64, freq_stride_s: I64,
    pos_offset: I32, n_tiles: I32, COS: cutlass.Constexpr[str],
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    lane = tid & 31
    r = tid >> 4
    c = tid & 15
    D = H * DH
    Dh = H * (DH // 2)          # D in Int32 words
    D3h = 3 * Dh

    s_w = cute.arch.alloc_smem(F32, 2 * DH, 32)           # [2][128]
    s_acc = cute.arch.alloc_smem(F32, 2 * ROWS * DH, 16)  # [2][16][128], epilogue only

    # norm weights -> smem, one entry per thread (2 * DH == THREADS)
    if tid < DH:
        (s_w + tid).store(qn_w.iterator[tid].to(F32))
    else:
        (s_w + tid).store(kn_w.iterator[tid - DH].to(F32))
    cute.arch.sync_threads()

    gqp = cutlass.Pointer(gq.iterator, dtype=BF16)
    gkp = cutlass.Pointer(gk.iterator, dtype=BF16)
    gvp = cutlass.Pointer(gv.iterator, dtype=BF16)
    qkvp = cutlass.Pointer(qkv.iterator, dtype=BF16)
    outp = cutlass.Pointer(out.iterator, dtype=I32)
    rqp = cutlass.Pointer(q_rstd.iterator, dtype=F32)
    rkp = cutlass.Pointer(k_rstd.iterator, dtype=F32)
    fp = cutlass.Pointer(freqs.iterator, dtype=F32)
    swp = cutlass.Pointer(s_w, dtype=F32)

    # persistent, contiguous tile range of this CTA (n_tiles * gdim fits Int64)
    nt = n_tiles.to(I64)
    b64 = bidx.to(I64)
    g64 = gdim.to(I64)
    t0 = ((nt * b64) // g64).to(I32)
    t1 = ((nt * (b64 + 1)) // g64).to(I32)

    wq_off = c * CHUNK
    wk_off = DH + c * CHUNK

    zero = F32(0.0)
    acc_q = cutlass.Vector.from_elements((zero,) * CHUNK, F32)
    acc_k = cutlass.Vector.from_elements((zero,) * CHUNK, F32)

    cq, ck, cv, cpq, cpk, crq, crk = _load_tile(
        t0, H, L, r, c, D, in_stride_b, stride_qkv_l, gqp, gkp, gvp, qkvp, rqp, rkp)

    g_lo = t0 // H
    g_hi = (t1 - 1) // H
    for g in cutlass.range(g_lo, g_hi + 1):
        # cos/sin for this row group (once per H tiles)
        rg_g = g * ROWS + r
        l_g = rg_g - (rg_g // L) * L
        foff = (l_g + pos_offset).to(I64) * freq_stride_s + (c * CHUNK).to(I64)
        fv = (fp + foff).load(16, count=CHUNK, is_invariant=True)
        cos_l = []
        sin_l = []
        for e in cutlass.range_constexpr(CHUNK):
            cos_l.append(_cos(fv[e], COS))
            sin_l.append(_sin(fv[e], COS))
        cs_c = cutlass.Vector.from_elements(tuple(cos_l), F32)
        cs_s = cutlass.Vector.from_elements(tuple(sin_l), F32)

        h_lo = cutlass.max(t0 - g * H, 0)
        h_hi = cutlass.min(t1 - g * H, H)
        for h in cutlass.range(h_lo, h_hi):
            i = g * H + h
            # prefetch the next tile (clamped: the last iteration re-reads itself)
            nq, nk, nv, npq, npk, nrq, nrk = _load_tile(
                cutlass.min(i + 1, t1 - 1), H, L, r, c, D, in_stride_b,
                stride_qkv_l, gqp, gkp, gvp, qkvp, rqp, rkp)

            # RoPE backward: sin product rounded, cos product fused (Triton 3.8.0)
            qf = cq.to(F32)
            kf = ck.to(F32)
            pqf = cpq.to(F32)
            pkf = cpk.to(F32)
            gq_l = []
            gk_l = []
            for i4 in cutlass.range_constexpr(4):
                e0 = 2 * i4
                e1 = e0 + 1
                q0 = qf[e0]
                q1 = qf[e1]
                gq_l.append(cute.math.fma(cs_c[e0], q0, cs_s[e0] * q1))
                gq_l.append(cute.math.fma(cs_c[e1], q1, cs_s[e1] * (-q0)))
                k0 = kf[e0]
                k1 = kf[e1]
                gk_l.append(cute.math.fma(cs_c[e0], k0, cs_s[e0] * k1))
                gk_l.append(cute.math.fma(cs_c[e1], k1, cs_s[e1] * (-k0)))
            gradq = cutlass.Vector.from_elements(tuple(gq_l), F32)
            gradk = cutlass.Vector.from_elements(tuple(gk_l), F32)

            # RMSNorm backward; volatile pins the smem weight reads inside the tile loop
            # (hoisted, the 16 weights stay live across tiles at the 128-register bound)
            wq = (swp + wq_off).load(32, count=CHUNK, is_volatile=True)
            wk = (swp + wk_off).load(32, count=CHUNK, is_volatile=True)
            gyw_q = gradq * wq
            gyw_k = gradk * wk
            sum_q = _triton_row_sum(gyw_q, pqf, lane)
            sum_k = _triton_row_sum(gyw_k, pkf, lane)
            c_q = _div_full((sum_q * crq) * crq, F32(128.0))
            c_k = _div_full((sum_k * crk) * crk, F32(128.0))
            oq = []
            ok = []
            aq = []
            ak = []
            for e in cutlass.range_constexpr(CHUNK):
                oq.append((cute.math.fma(-c_q, pqf[e], gyw_q[e]) * crq).to(BF16))
                ok.append((cute.math.fma(-c_k, pkf[e], gyw_k[e]) * crk).to(BF16))
                # norm-weight partials (fp32)
                aq.append(cute.math.fma(gradq[e] * pqf[e], crq, acc_q[e]))
                ak.append(cute.math.fma(gradk[e] * pkf[e], crk, acc_k[e]))
            acc_q = cutlass.Vector.from_elements(tuple(aq), F32)
            acc_k = cutlass.Vector.from_elements(tuple(ak), F32)

            # store the joined gradient (Int32 pointer, word offsets)
            rg = g * ROWS + r
            o_off = rg.to(I64) * D3h.to(I64) + (h * (DH // 2) + c * (CHUNK // 2)).to(I64)
            (outp + o_off).store(
                cutlass.Vector.from_elements(tuple(oq), BF16).bitcast(I32), alignment=16)
            (outp + (o_off + Dh.to(I64))).store(
                cutlass.Vector.from_elements(tuple(ok), BF16).bitcast(I32), alignment=16)
            (outp + (o_off + (2 * Dh).to(I64))).store(cv.bitcast(I32), alignment=16)

            cq = nq
            ck = nk
            cv = nv
            cpq = npq
            cpk = npk
            crq = nrq
            crk = nrk

    # CTA epilogue: reduce the norm-weight partials over the 16 rows
    for e in cutlass.range_constexpr(CHUNK):
        (s_acc + (r * DH + c * CHUNK + e)).store(acc_q[e])
        (s_acc + (ROWS * DH + r * DH + c * CHUNK + e)).store(acc_k[e])
    cute.arch.sync_threads()
    tau = tid >> 7
    d = tid & (DH - 1)
    s = F32(0.0)
    for rr in cutlass.range_constexpr(ROWS):
        s = s + (s_acc + (tau * (ROWS * DH) + rr * DH + d)).load()
    # atomic_add (atomicrmw fadd, result unused -> RED); cute.arch.red with default
    # sem/scope is rejected by libNVVM
    if tau == 0:
        cute.arch.atomic_add(grad_qn_w.iterator + d, s)
    else:
        cute.arch.atomic_add(grad_kn_w.iterator + d, s)


@cute.jit
def rope_norm_join_ctm_host(
    gq: cute.Tensor, gk: cute.Tensor, gv: cute.Tensor, qkv: cute.Tensor,
    q_rstd: cute.Tensor, k_rstd: cute.Tensor,
    qn_w: cute.Tensor, kn_w: cute.Tensor, freqs: cute.Tensor,
    out: cute.Tensor, grad_qn_w: cute.Tensor, grad_kn_w: cute.Tensor,
    L: I32, H: I32, in_stride_b: I64, stride_qkv_l: I64, freq_stride_s: I64,
    pos_offset: I32, n_tiles: I32, grid_x: I32, stream: cuda.CUstream,
    COS: cutlass.Constexpr[str],
) -> None:
    rope_norm_join_ctm_kernel(
        gq, gk, gv, qkv, q_rstd, k_rstd, qn_w, kn_w, freqs, out, grad_qn_w,
        grad_kn_w, L, H, in_stride_b, stride_qkv_l, freq_stride_s, pos_offset,
        n_tiles, COS,
    ).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream,
             min_blocks_per_mp=MIN_CTAS_PER_SM,
             preferred_smem_carveout=SMEM_CARVEOUT_PCT)


_compiled: dict[tuple[int, str], Callable] = {}


def cos_mode() -> str:
    """The cos/sin lowering in use: ROPE_CTM_COS (math unless set)."""
    mode = os.environ.get("ROPE_CTM_COS", "math")
    if mode not in COS_MODES:
        raise ValueError(f"ROPE_CTM_COS must be one of {COS_MODES}, got {mode!r}")
    return mode


def _compile(device_index: int, mode: str | None = None) -> Callable:
    """Compile once per (device, cos mode); every length and stride stays symbolic."""
    mode = mode or cos_mode()
    key = (device_index, mode)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            def bf(align: int = 16):
                return make_fake_compact_tensor(BF16, (cute.sym_int64(),), assumed_align=align)

            def f32(align: int):
                return make_fake_compact_tensor(F32, (cute.sym_int64(),), assumed_align=align)

            fakes = [bf(), bf(), bf(), bf(),        # gq gk gv qkv
                     f32(4), f32(4),                # q_rstd k_rstd
                     bf(2), bf(2),                  # qn_w kn_w (scalar reads)
                     f32(16),                       # freqs (float4 reads)
                     bf(),                          # out
                     f32(4), f32(4)]                # grad_qn_w grad_kn_w
            ints = [I32(0), I32(0), I64(0), I64(0), I64(0), I32(0), I32(0), I32(0)]
            _compiled[key] = cute.compile(
                rope_norm_join_ctm_host, *fakes, *ints, make_fake_stream(), mode,
                options="--enable-tvm-ffi")
    return _compiled[key]


def max_ctas(device_index: int = 0) -> int:
    """Persistent grid upper bound: SMs x MIN_CTAS_PER_SM."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count * MIN_CTAS_PER_SM


def _flat(t: torch.Tensor) -> torch.Tensor:
    """1-D stride-1 view over the span this tensor addresses (strided slices too)."""
    if t.is_contiguous():
        return t.view(-1)
    span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if s > 0)
    return t.as_strided((span,), (1,))


def _check(grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd,
           qn_weight, kn_weight, grad_qn_weight, grad_kn_weight, freqs, pos_offset):
    B, L, D = grad_q_rope.shape
    if D != H * DH:
        raise ValueError("rope_norm_join_ctm: head dim must be 128")
    if L % ROWS:
        raise ValueError(f"rope_norm_join_ctm requires L % 16 == 0, got L={L}")
    for t in (grad_q_rope, grad_k_rope, grad_v):
        if t.dtype != torch.bfloat16 or t.shape != grad_q_rope.shape:
            raise ValueError("rope_norm_join_ctm: grad views must be bf16 [B, L, D]")
        if t.stride(2) != 1 or t.stride(1) != D or t.stride(0) != grad_q_rope.stride(0):
            raise ValueError("rope_norm_join_ctm: grad views must be row-contiguous "
                             "[B, L, D] slices of one buffer")
        if t.data_ptr() % 16:
            raise ValueError("rope_norm_join_ctm: grad views must be 16-byte aligned")
    if grad_q_rope.stride(0) % CHUNK:
        raise ValueError("rope_norm_join_ctm: batch stride must be a multiple of 8")
    if (qkv.dtype != torch.bfloat16 or not qkv.is_contiguous()
            or tuple(qkv.shape) != (B, L, 3 * D)):
        raise ValueError("rope_norm_join_ctm: qkv must be contiguous bf16 [B, L, 3D]")
    for t in (q_rstd, k_rstd):
        if t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != B * L * H:
            raise ValueError("rope_norm_join_ctm: rstd must be contiguous fp32 [B*L, H]")
    for t in (qn_weight, kn_weight):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.numel() != DH:
            raise ValueError("rope_norm_join_ctm: norm weights must be contiguous bf16 [128]")
    for t in (grad_qn_weight, grad_kn_weight):
        if t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != DH:
            raise ValueError("rope_norm_join_ctm: norm-weight grads must be contiguous fp32 [128]")
    if (freqs.dtype != torch.float32 or freqs.stride(-1) != 1 or freqs.shape[-1] != DH
            or freqs.stride(0) % 4 or freqs.shape[0] < pos_offset + L
            or freqs.data_ptr() % 16):
        raise ValueError("rope_norm_join_ctm: freqs must be fp32 [S >= pos_offset + L, ..., 128], "
                         "row stride a multiple of 4, 16-byte aligned")
    return B, L, D


def rope_norm_join_ctm(
    grad_q_rope: torch.Tensor,
    grad_k_rope: torch.Tensor,
    grad_v: torch.Tensor,
    qkv: torch.Tensor,
    H: int,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    qn_weight: torch.Tensor,
    kn_weight: torch.Tensor,
    grad_qn_weight: torch.Tensor,
    grad_kn_weight: torch.Tensor,
    freqs: torch.Tensor,
    pos_offset: int,
) -> torch.Tensor:
    """Fused RoPE backward + QK norm backward + QKV join (CUTLASS DSL).

    Same contract as flux.rope_norm_join: grad_* bf16 [B, L, D] views (slices of one buffer
    allowed), grad_*_weight fp32 atomic accumulators, L % 16 == 0. Returns bf16 [B, L, 3*D].
    """
    B, L, D = _check(grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd,
                     qn_weight, kn_weight, grad_qn_weight, grad_kn_weight, freqs,
                     pos_offset)
    dev = grad_q_rope.device.index
    grad_qkv = torch.empty(B, L, 3 * D, dtype=torch.bfloat16, device=grad_q_rope.device)
    n_tiles = B * L // ROWS * H
    cap = int(os.environ.get("ROPE_CTM_MAX_CTAS", "0")) or max_ctas(dev)
    grid = min(n_tiles, cap)
    _compile(dev)(
        _flat(grad_q_rope), _flat(grad_k_rope), _flat(grad_v), qkv.view(-1),
        q_rstd.view(-1), k_rstd.view(-1), qn_weight.view(-1), kn_weight.view(-1),
        _flat(freqs), grad_qkv.view(-1), grad_qn_weight.view(-1), grad_kn_weight.view(-1),
        L, H, grad_q_rope.stride(0), qkv.stride(1), freqs.stride(0), pos_offset,
        n_tiles, grid, cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream),
    )
    return grad_qkv


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "rope_norm_join_ctm(Tensor grad_q_rope, Tensor grad_k_rope,"
    " Tensor grad_v, Tensor qkv, int H,"
    " Tensor q_rstd, Tensor k_rstd,"
    " Tensor qn_weight, Tensor kn_weight,"
    " Tensor grad_qn_weight, Tensor grad_kn_weight,"
    " Tensor freqs, int pos_offset) -> Tensor"
)


@torch.library.impl(_lib, "rope_norm_join_ctm", "CUDA")
def _rope_norm_join_ctm_impl(
    grad_q_rope: torch.Tensor, grad_k_rope: torch.Tensor, grad_v: torch.Tensor,
    qkv: torch.Tensor, H: int, q_rstd: torch.Tensor, k_rstd: torch.Tensor,
    qn_weight: torch.Tensor, kn_weight: torch.Tensor,
    grad_qn_weight: torch.Tensor, grad_kn_weight: torch.Tensor,
    freqs: torch.Tensor, pos_offset: int,
) -> torch.Tensor:
    return rope_norm_join_ctm(
        grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd,
        qn_weight, kn_weight, grad_qn_weight, grad_kn_weight, freqs, pos_offset,
    )
