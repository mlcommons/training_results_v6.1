"""The X12 fused QKV RoPE forward straight to FP8 (Llama 3.1 8B campaign, 2026-10-07): two passes instead of the RoPE
writing the bf16 q|k|v buffer and the FP8 current-scaling cast reading it back.

TE's FP8 attention (Float8CurrentScaling, NVTE_DPA_FP8CS_O_in_F16=1) quantizes the packed [q | k | v] buffer with one
per-tensor scale and saves only the FP8 q, k, v for the backward (backends.py: fp8_tensors = (q_fp8, k_fp8, v_fp8,
None)), so once quantized the bf16 buffer is dead.  Here:
  * `_qkv_rope_amax_only_kernel` is `fused_qkv_rope_amax._qkv_rope_fwd_amax_kernel` (same grid, loads, arithmetic and
    bf16 rounding; the same partial maximum per (s, b) position) without its q / k / v stores;
  * the 1-CTA scale kernel turns the partials into the scale and scale_inv (TE's compute_scale_from_amax);
  * `_qkv_rope_fp8_kernel` recomputes the same bf16 values from the same inputs and writes fp8(bf16 value * scale)
    (round to nearest even, saturating: TE's cast) at each element's position in the packed buffer -- the bytes TE's
    quantize of that buffer would write.  REV walks the positions back to front (the first pass leaves the tail of its
    input in L2).
Traffic per layer: 201 MB read twice and 100 MB written, against 201 MB read, 201 MB written, 201 MB read and 100 MB
written for the RoPE + cast.
"""
import triton
import triton.language as tl

from fused_qkv_rope import _neg


@triton.jit
def _e4m3_sat(x, ASM: tl.constexpr):
    """fp32 -> FP8 E4M3 bytes, round to nearest even, saturating (TE's __nv_cvt_float2_to_fp8x2(.., __NV_SATFINITE,
    __NV_E4M3)): Triton's own conversion, or cvt.rn.satfinite.e4m3x2.f32 directly (ASM)"""
    if ASM:
        return tl.inline_asm_elementwise(asm="cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;", constraints="=h,r,r",
                                         args=[x], dtype=tl.uint8, is_pure=True, pack=2)
    else:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _qkv_rope_amax_only_kernel(qkv_ptr, tab_ptr, part_ptr, b,
                               D: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                               TOTAL: tl.constexpr, SAME: tl.constexpr):
    pid = tl.program_id(0)                     # one (s, b) position of H*TOTAL elements
    s_id = pid // b
    half: tl.constexpr = D // 2
    dd = tl.arange(0, half)[None, :]
    tb = s_id * (4 * D)
    c0 = tl.load(tab_ptr + tb + dd)
    c1 = tl.load(tab_ptr + tb + half + dd)
    t0 = tl.load(tab_ptr + tb + D + dd)
    t1 = tl.load(tab_ptr + tb + D + half + dd)
    if SAME:
        kc0 = c0
        kc1 = c1
        kt0 = t0
        kt1 = t1
    else:
        kc0 = tl.load(tab_ptr + tb + 2 * D + dd)
        kc1 = tl.load(tab_ptr + tb + 2 * D + half + dd)
        kt0 = tl.load(tab_ptr + tb + 3 * D + dd)
        kt1 = tl.load(tab_ptr + tb + 3 * D + half + dd)
    base = pid * (H * TOTAL)
    r = tl.arange(0, H * NQ)[:, None]
    qrow = base + (r // NQ) * TOTAL + (r % NQ) * D
    x0 = tl.load(qkv_ptr + qrow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + qrow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, c0, _neg(x1) * t0)
    y1 = tl.fma(x1, c1, x0 * t1)
    b0 = y0.to(tl.bfloat16)
    b1 = y1.to(tl.bfloat16)
    m = tl.maximum(tl.max(tl.abs(b0.to(tl.float32))), tl.max(tl.abs(b1.to(tl.float32))))
    g = tl.arange(0, H)[:, None]
    krow = base + g * TOTAL + NQ * D
    x0 = tl.load(qkv_ptr + krow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + krow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, kc0, _neg(x1) * kt0)
    y1 = tl.fma(x1, kc1, x0 * kt1)
    b0 = y0.to(tl.bfloat16)
    b1 = y1.to(tl.bfloat16)
    m = tl.maximum(m, tl.maximum(tl.max(tl.abs(b0.to(tl.float32))), tl.max(tl.abs(b1.to(tl.float32)))))
    gv = tl.arange(0, H)[:, None]
    dv = tl.arange(0, D)[None, :]
    vv = tl.load(qkv_ptr + base + gv * TOTAL + (NQ + 1) * D + dv)
    m = tl.maximum(m, tl.max(tl.abs(vv.to(tl.float32))))
    tl.store(part_ptr + pid, m)


@triton.jit
def _qkv_rope_fp8_kernel(qkv_ptr, tab_ptr, ws_ptr, out_ptr, b, n_prog, nqe, nke,
                         D: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                         TOTAL: tl.constexpr, SAME: tl.constexpr, REV: tl.constexpr, ASM: tl.constexpr):
    if REV:
        pid = n_prog - 1 - tl.program_id(0)
    else:
        pid = tl.program_id(0)
    s_id = pid // b
    half: tl.constexpr = D // 2
    dd = tl.arange(0, half)[None, :]
    tb = s_id * (4 * D)
    c0 = tl.load(tab_ptr + tb + dd)
    c1 = tl.load(tab_ptr + tb + half + dd)
    t0 = tl.load(tab_ptr + tb + D + dd)
    t1 = tl.load(tab_ptr + tb + D + half + dd)
    if SAME:
        kc0 = c0
        kc1 = c1
        kt0 = t0
        kt1 = t1
    else:
        kc0 = tl.load(tab_ptr + tb + 2 * D + dd)
        kc1 = tl.load(tab_ptr + tb + 2 * D + half + dd)
        kt0 = tl.load(tab_ptr + tb + 3 * D + dd)
        kt1 = tl.load(tab_ptr + tb + 3 * D + half + dd)
    scale = tl.load(ws_ptr)
    base = pid * (H * TOTAL)
    # ---- query heads: the packed buffer's q section
    r = tl.arange(0, H * NQ)[:, None]
    qrow = base + (r // NQ) * TOTAL + (r % NQ) * D
    x0 = tl.load(qkv_ptr + qrow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + qrow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, c0, _neg(x1) * t0)
    y1 = tl.fma(x1, c1, x0 * t1)
    qout = pid * (H * NQ * D) + r * D
    tl.store(out_ptr + qout + dd, _e4m3_sat(y0.to(tl.bfloat16).to(tl.float32) * scale, ASM))
    tl.store(out_ptr + qout + half + dd, _e4m3_sat(y1.to(tl.bfloat16).to(tl.float32) * scale, ASM))
    # ---- key heads: the k section
    g = tl.arange(0, H)[:, None]
    krow = base + g * TOTAL + NQ * D
    x0 = tl.load(qkv_ptr + krow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + krow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, kc0, _neg(x1) * kt0)
    y1 = tl.fma(x1, kc1, x0 * kt1)
    kout = nqe + pid * (H * D) + g * D
    tl.store(out_ptr + kout + dd, _e4m3_sat(y0.to(tl.bfloat16).to(tl.float32) * scale, ASM))
    tl.store(out_ptr + kout + half + dd, _e4m3_sat(y1.to(tl.bfloat16).to(tl.float32) * scale, ASM))
    # ---- value heads: the v section, unrotated
    gv = tl.arange(0, H)[:, None]
    dv = tl.arange(0, D)[None, :]
    vv = tl.load(qkv_ptr + base + gv * TOTAL + (NQ + 1) * D + dv)
    tl.store(out_ptr + nqe + nke + pid * (H * D) + gv * D + dv, _e4m3_sat(vv.to(tl.float32) * scale, ASM))


@triton.jit
def _qkv_rope_amax_only_kernel2(qkv_ptr, tab_ptr, part_ptr, b,
                                D: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                                TOTAL: tl.constexpr, SAME: tl.constexpr):
    """_qkv_rope_amax_only_kernel with three block reductions instead of five: the element-wise max of each section's
    two halves first (max is order-free, NaN ignored as tl.max / tl.maximum ignore it)"""
    pid = tl.program_id(0)
    s_id = pid // b
    half: tl.constexpr = D // 2
    dd = tl.arange(0, half)[None, :]
    tb = s_id * (4 * D)
    c0 = tl.load(tab_ptr + tb + dd)
    c1 = tl.load(tab_ptr + tb + half + dd)
    t0 = tl.load(tab_ptr + tb + D + dd)
    t1 = tl.load(tab_ptr + tb + D + half + dd)
    if SAME:
        kc0 = c0
        kc1 = c1
        kt0 = t0
        kt1 = t1
    else:
        kc0 = tl.load(tab_ptr + tb + 2 * D + dd)
        kc1 = tl.load(tab_ptr + tb + 2 * D + half + dd)
        kt0 = tl.load(tab_ptr + tb + 3 * D + dd)
        kt1 = tl.load(tab_ptr + tb + 3 * D + half + dd)
    base = pid * (H * TOTAL)
    r = tl.arange(0, H * NQ)[:, None]
    qrow = base + (r // NQ) * TOTAL + (r % NQ) * D
    x0 = tl.load(qkv_ptr + qrow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + qrow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, c0, _neg(x1) * t0)
    y1 = tl.fma(x1, c1, x0 * t1)
    aq = tl.maximum(tl.abs(y0.to(tl.bfloat16).to(tl.float32)), tl.abs(y1.to(tl.bfloat16).to(tl.float32)))
    g = tl.arange(0, H)[:, None]
    krow = base + g * TOTAL + NQ * D
    x0 = tl.load(qkv_ptr + krow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + krow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, kc0, _neg(x1) * kt0)
    y1 = tl.fma(x1, kc1, x0 * kt1)
    ak = tl.maximum(tl.abs(y0.to(tl.bfloat16).to(tl.float32)), tl.abs(y1.to(tl.bfloat16).to(tl.float32)))
    gv = tl.arange(0, H)[:, None]
    dv = tl.arange(0, D)[None, :]
    vv = tl.load(qkv_ptr + base + gv * TOTAL + (NQ + 1) * D + dv)
    m = tl.maximum(tl.maximum(tl.max(aq), tl.max(ak)), tl.max(tl.abs(vv.to(tl.float32))))
    tl.store(part_ptr + pid, m)
