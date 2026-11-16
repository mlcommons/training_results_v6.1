"""gen_variants.py: writes variants.cuh -- a copy of TE's row_col_rht_gemm_device and its launcher (taken from the generated
ak_l8b_rht.cu, i.e. TE's text) renamed *_v with one extra template parameter kVar_ and these switchable changes:
  bit 0 (1): the 16-element block amax as a tree of non-volatile max.NaN.f32 (TE: a serial chain of asm volatile
             max.NaN.f32 from 0; max with NaN propagation over |x| is order-independent, so the value is the same);
  bit 1 (2): the block multiply x * scale as packed f32x2 multiplies (the same IEEE product per element);
  bit 2 (4): the stochastic-rounding subsequence computed from the 128-token block index as TE's k_tile_size 1024 would
             give it, so any k_tile_size draws TE's random bits (TE: from the tile index of the launch's own tiling);
  bit 3 (8): Philox draw v of a k-tile computed from the counter + v directly (a branch-free 128-bit add) instead of v
             successive generate4() calls whose counter increments branch; the same counters, so the same bits, and the
             draws of a k-tile become independent;
  bit 4 (16): as bit 3, with a warp-uniform fast path: after init the counter's low word is the call's offset (the same
             in every thread), so while it cannot wrap within the k-tile's draws, draw v's counter is (x + v, y, z, w);
  bit 5 (32): draw v's counter as the 128-bit sum counter + v in one add.cc / addc chain (4 instructions, no branch;
             exactly v successive increments), then TE's rounds;
  bit 6 (64): the column copy's round trip to bf16 (TE: pack pairs with cvt.rn.bf16x2.f32, unpack both halves) as one
             cvt.rn.bf16x2.f32 per element with +0.0 as the low partner, whose register is then the rounded value as
             an fp32 bit pattern (the same rounding, NaN and -0.0 included).
Usage: python3 gen_variants.py ak_l8b_rht.cu variants.cuh"""
import re
import argparse
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("src")
_ap.add_argument("out")
_args = _ap.parse_args()
src, out = _args.src, _args.out
s = open(src).read()
a = s.index("template <class MShape, class NShape, class KShape, class ClusterShape, class ClusterTileShape,")
b = s.index("// this function computes RHT-GEMM for")
c = s.index("}  // namespace detail")
dev = s[a:b]
lau = s[b:c]

# --- device kernel
n0 = dev.count("row_col_rht_gemm_device(")
assert n0 == 1, n0
dev = dev.replace("          bool kUseFastMath_ = true>\n__launch_bounds__(512, 1)\n__global__ static void row_col_rht_gemm_device(",
                  "          bool kUseFastMath_ = true,\n          int kVar_ = 0>\n__launch_bounds__(512, 1)\n__global__ static void row_col_rht_gemm_device_v(")
assert "row_col_rht_gemm_device_v(" in dev
# amax
k = dev.count("amax_reduction(ElementAccumulator(0), compute_frgs[v])")
k2 = dev.count("amax_reduction(ElementAccumulator(0), compute_frgs_up)")
assert k == 1 and k2 == 1, (k, k2)
dev = dev.replace("amax_reduction(ElementAccumulator(0), compute_frgs[v])", "ak_amax<kVar_>(amax_reduction, compute_frgs[v])")
dev = dev.replace("amax_reduction(ElementAccumulator(0), compute_frgs_up)", "ak_amax<kVar_>(amax_reduction, compute_frgs_up)")
# Philox draws
rt = dev.count("""            tTR_rAcc_frag(_0{}) = convert_bf16_to_accum(convert_accum_to_bf16(tTR_rAcc_frag(_0{})));
            tTR_rAcc_frag(_1{}) = convert_bf16_to_accum(convert_accum_to_bf16(tTR_rAcc_frag(_1{})));""")
assert rt == 1, rt
dev = dev.replace("""            tTR_rAcc_frag(_0{}) = convert_bf16_to_accum(convert_accum_to_bf16(tTR_rAcc_frag(_0{})));
            tTR_rAcc_frag(_1{}) = convert_bf16_to_accum(convert_accum_to_bf16(tTR_rAcc_frag(_1{})));""", """            tTR_rAcc_frag(_0{}) = ak_round_bf16<kVar_>(convert_accum_to_bf16, convert_bf16_to_accum, tTR_rAcc_frag(_0{}));
            tTR_rAcc_frag(_1{}) = ak_round_bf16<kVar_>(convert_accum_to_bf16, convert_bf16_to_accum, tTR_rAcc_frag(_1{}));""")
g = dev.count("random_uint4 = rng.generate4();")
assert g == 2, g
dev = dev.replace("random_uint4 = rng.generate4();", "random_uint4 = ak_draw<kVar_>(rng, v);")
# multiplies
m = dev.count("cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(")
assert m == 4, m
dev = dev.replace("cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(", "ak_mul<kVar_>(")
# rng sequences: column path first (k_tile = the block being processed), row path second (k_tile already incremented)
seq = "const size_t rng_sequence = global_thread_idx + k_tile * 512 + scheduler.get_linear_tile_idx() * K_TILE_MAX * 512;"
assert dev.count(seq) == 2
i1 = dev.index(seq)
dev = dev[:i1] + ("const size_t rng_sequence = ak_seq<kVar_>(global_thread_idx, k_tile, 0, scheduler.tile_m(), "
                  "scheduler.tile_n_base(), scheduler.tiles_m(), K_TILE_MAX, scheduler.get_linear_tile_idx());") + \
      dev[i1 + len(seq):]
i2 = dev.index(seq)
dev = dev[:i2] + ("const size_t rng_sequence = ak_seq<kVar_>(global_thread_idx, k_tile, 1, scheduler.tile_m(), "
                  "scheduler.tile_n_base(), scheduler.tiles_m(), K_TILE_MAX, scheduler.get_linear_tile_idx());") + \
      dev[i2 + len(seq):]

# --- launcher
assert lau.count("void row_col_rht_gemm_ntt_w_sfc(") == 1
lau = lau.replace("class TA, class TB, class TD, class TSFD, class TQA, class TSFA, bool kUseFastMath=true>\nvoid row_col_rht_gemm_ntt_w_sfc(",
                  "class TA, class TB, class TD, class TSFD, class TQA, class TSFA, bool kUseFastMath=true, int kVar=0>\nvoid row_col_rht_gemm_ntt_w_sfc_v(")
assert "row_col_rht_gemm_ntt_w_sfc_v(" in lau
lau = lau.replace("  auto* kernel_ptr = &row_col_rht_gemm_device<", "  auto* kernel_ptr = &row_col_rht_gemm_device_v<")
lau = lau.replace("    kEnableRowQuant,\n    kUseFastMath>;", "    kEnableRowQuant,\n    kUseFastMath,\n    kVar>;")
assert "kVar>;" in lau and "row_col_rht_gemm_device_v<" in lau

helpers = r'''
// ---- helpers of the variants (kVar bits: 1 tree amax, 2 packed f32x2 multiply, 4 TE-1024 RNG subsequence) ----
CUTLASS_DEVICE float ak_max_nan(float a, float b) {
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));   // TE's instruction, not volatile: free to schedule
  return r;
}

template <int kVar, class Red, class Arr>
CUTLASS_DEVICE float ak_amax(Red const& red, Arr const& x) {
  if constexpr ((kVar & 1) != 0) {
    static_assert(Arr::kElements == 16, "16-element blocks");
    float m[8];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) m[i] = ak_max_nan(fabsf(float(x[2 * i])), fabsf(float(x[2 * i + 1])));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) m[i] = ak_max_nan(m[2 * i], m[2 * i + 1]);
    m[0] = ak_max_nan(m[0], m[1]);
    m[2] = ak_max_nan(m[2], m[3]);
    return ak_max_nan(ak_max_nan(m[0], m[2]), 0.0f);   // TE folds from +0.0
  } else {
    return red(float(0), x);
  }
}

template <int kVar, class Arr>
CUTLASS_DEVICE Arr ak_mul(Arr const& x, float s) {
  if constexpr ((kVar & 2) != 0) {
    static_assert(Arr::kElements % 2 == 0, "pairs");
    Arr r;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Arr::kElements; i += 2) {
      asm("{\n\t.reg .b64 ta, tb, td;\n\t"
          "mov.b64 ta, {%2, %3};\n\t"
          "mov.b64 tb, {%4, %4};\n\t"
          "mul.rn.f32x2 td, ta, tb;\n\t"
          "mov.b64 {%0, %1}, td;\n\t}"
          : "=f"(r[i]), "=f"(r[i + 1]) : "f"(x[i]), "f"(x[i + 1]), "f"(s));
    }
    return r;
  } else {
    return cutlass::multiplies<Arr>{}(x, s);
  }
}


// bit 8: draw v of a k-tile from the initialised state (counter + v, a 128-bit add with carries) -- philox_state_incr
// applied v times is exactly that add; TE calls generate4() v + 1 times, each incrementing after its rounds.
template <int kVar, class Rng>
CUTLASS_DEVICE uint4 ak_draw16(Rng& rng, int v);
template <int kVar, class Rng>
CUTLASS_DEVICE uint4 ak_draw(Rng& rng, int v) {
  if constexpr ((kVar & 32) != 0) {
    uint32_t x, y, z, w;
    asm("add.cc.u32 %0, %4, %8;\n\taddc.cc.u32 %1, %5, 0;\n\taddc.cc.u32 %2, %6, 0;\n\taddc.u32 %3, %7, 0;"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
        : "r"(rng.ctr.x), "r"(rng.ctr.y), "r"(rng.ctr.z), "r"(rng.ctr.w), "r"(uint32_t(v)));
    return transformer_engine::curanddx::detail::multiple_rounds<Rng::rounds>(make_uint4(x, y, z, w), rng.key);
  } else if constexpr ((kVar & 16) != 0) {
    return ak_draw16<kVar>(rng, v);
  } else if constexpr ((kVar & 8) != 0) {
    uint4 c = rng.ctr;
    const uint32_t x = c.x + uint32_t(v);
    uint32_t carry = x < c.x ? 1u : 0u;
    const uint32_t y = c.y + carry;
    carry = carry & (y == 0u ? 1u : 0u);
    const uint32_t z = c.z + carry;
    carry = carry & (z == 0u ? 1u : 0u);
    const uint32_t w = c.w + carry;
    return transformer_engine::curanddx::detail::multiple_rounds<Rng::rounds>(make_uint4(x, y, z, w), rng.key);
  } else {
    return rng.generate4();
  }
}


// bit 16: as bit 8, with the warp-uniform fast path x + v (no carry possible while x <= 0xFFFFFFF0)
template <int kVar, class Rng>
CUTLASS_DEVICE uint4 ak_draw16(Rng& rng, int v) {
  const uint4 c = rng.ctr;
  if (c.x <= 0xFFFFFFF0u) {
    return transformer_engine::curanddx::detail::multiple_rounds<Rng::rounds>(make_uint4(c.x + uint32_t(v), c.y, c.z, c.w),
                                                                                rng.key);
  }
  const uint32_t x = c.x + uint32_t(v);
  uint32_t carry = x < c.x ? 1u : 0u;
  const uint32_t y = c.y + carry;
  carry = carry & (y == 0u ? 1u : 0u);
  const uint32_t z = c.z + carry;
  carry = carry & (z == 0u ? 1u : 0u);
  const uint32_t w = c.w + carry;
  return transformer_engine::curanddx::detail::multiple_rounds<Rng::rounds>(make_uint4(x, y, z, w), rng.key);
}

// bit 64: round an fp32 fragment to bf16 and back in one instruction per element
template <int kVar, class ToBf16, class ToF32, class Frag>
CUTLASS_DEVICE Frag ak_round_bf16(ToBf16 const& to_bf16, ToF32 const& to_f32, Frag const& f) {
  if constexpr ((kVar & 64) != 0) {
    Frag r;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Frag::kElements; ++i) {
      uint32_t u;
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(f[i]), "f"(0.0f));
      r[i] = __uint_as_float(u);
    }
    return r;
  } else {
    return to_f32(to_bf16(f));
  }
}

// TE: seq = thread + k * 512 + L * K_TILE_MAX * 512, L = m_idx + n_idx * tiles_m, k the k-tile index (the row path's
// already incremented: k + 1).  With bit 4, the 128-token block b = n_idx * K_TILE_MAX + k is mapped to TE's k_tile_size
// 1024 tiling (8 blocks per tile) first, so any k_tile_size gives TE's subsequences.
template <int kVar>
CUTLASS_DEVICE size_t ak_seq(int thread, int k_tile, int row_path, uint32_t m_idx, uint32_t n_base, uint32_t tiles_m,
                             int k_tile_max, uint32_t linear) {
  if constexpr ((kVar & 4) != 0) {
    const uint32_t blk = n_base + uint32_t(k_tile - row_path);     // the 128-token block being processed
    const uint32_t te_k = blk % 8u, te_n = blk / 8u;
    const size_t te_l = size_t(m_idx) + size_t(te_n) * size_t(tiles_m);
    return size_t(thread) + size_t(te_k + uint32_t(row_path)) * 512 + te_l * 8 * 512;
  } else {
    return size_t(thread) + size_t(k_tile) * 512 + size_t(linear) * size_t(k_tile_max) * 512;
  }
}
'''
body = "// variants.cuh: generated by gen_variants.py from TE's kernel text (see its docstring).\nnamespace ak_rht {\nnamespace detail {\n" + \
       helpers + "\n" + dev + "\n" + lau + "\n}  // namespace detail\n}  // namespace ak_rht\n"
open(out, "w").write(body)
print("wrote", out)
