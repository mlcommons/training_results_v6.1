"""The X12 fused QKV RoPE forward with the FP8 current-scaling amax of its output (Llama 3.1 8B campaign, 2026-10-02).

`_qkv_rope_fwd_amax_kernel` is `fused_qkv_rope._qkv_rope_fwd_kernel` (same grid, arithmetic and stores) plus one
partial maximum per program: max |y| over the bf16 values it stores (the rotated q and k heads and the copied v heads)
of its (s, b) position, written to part[pid].  TE quantizes the packed [q | k | v] buffer with one current-scaling
scale whose amax is max |x| over the whole buffer (fmax, NaN ignored); the max of these partials is that amax, so the
amax pass over the 100 M-element buffer (PROF-L0: 52 us per layer) is not needed.  Plain stores, no atomics: the
16384 partials are reduced by one CTA in the FP8 cast's prologue (ak_l8b_fp8cs.from_partials).
"""
import triton
import triton.language as tl

from fused_qkv_rope import _neg


@triton.jit
def _qkv_rope_fwd_amax_kernel(qkv_ptr, tab_ptr, q_ptr, k_ptr, v_ptr, part_ptr, b,
                              D: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                              TOTAL: tl.constexpr, SAME: tl.constexpr):
    pid = tl.program_id(0)                     # one (s, b) position of H*TOTAL elements
    s_id = pid // b
    half: tl.constexpr = D // 2
    dd = tl.arange(0, half)[None, :]           # offset inside a half
    tb = s_id * (4 * D)                        # this position's row of the angle table
    c0 = tl.load(tab_ptr + tb + dd)                     # cos of the low half
    c1 = tl.load(tab_ptr + tb + half + dd)              # cos of the high half
    t0 = tl.load(tab_ptr + tb + D + dd)                 # sin of the low half
    t1 = tl.load(tab_ptr + tb + D + half + dd)          # sin of the high half
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
    # ---- query heads
    r = tl.arange(0, H * NQ)[:, None]
    qrow = base + (r // NQ) * TOTAL + (r % NQ) * D
    x0 = tl.load(qkv_ptr + qrow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + qrow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, c0, _neg(x1) * t0)
    y1 = tl.fma(x1, c1, x0 * t1)
    qout = pid * (H * NQ * D) + r * D
    b0 = y0.to(tl.bfloat16)
    b1 = y1.to(tl.bfloat16)
    tl.store(q_ptr + qout + dd, b0)
    tl.store(q_ptr + qout + half + dd, b1)
    m = tl.maximum(tl.max(tl.abs(b0.to(tl.float32))), tl.max(tl.abs(b1.to(tl.float32))))
    # ---- key heads
    g = tl.arange(0, H)[:, None]
    krow = base + g * TOTAL + NQ * D
    x0 = tl.load(qkv_ptr + krow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + krow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, kc0, _neg(x1) * kt0)
    y1 = tl.fma(x1, kc1, x0 * kt1)
    kout = pid * (H * D) + g * D
    b0 = y0.to(tl.bfloat16)
    b1 = y1.to(tl.bfloat16)
    tl.store(k_ptr + kout + dd, b0)
    tl.store(k_ptr + kout + half + dd, b1)
    m = tl.maximum(m, tl.maximum(tl.max(tl.abs(b0.to(tl.float32))), tl.max(tl.abs(b1.to(tl.float32)))))
    # ---- value heads: copied unchanged
    gv = tl.arange(0, H)[:, None]
    dv = tl.arange(0, D)[None, :]
    vv = tl.load(qkv_ptr + base + gv * TOTAL + (NQ + 1) * D + dv)
    tl.store(v_ptr + pid * (H * D) + gv * D + dv, vv)
    m = tl.maximum(m, tl.max(tl.abs(vv.to(tl.float32))))
    tl.store(part_ptr + pid, m)
