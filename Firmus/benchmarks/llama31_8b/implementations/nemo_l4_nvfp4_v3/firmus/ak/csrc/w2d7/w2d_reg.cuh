// w2d_reg.cuh (Llama 3.1 8B kernel campaign, 2026-10-07): TE's NVFP4 2D weight cast (quantize_transpose_nvfp4_2D_kernel
// for a bf16 weight, round to nearest) on a register-tile schedule.  Every value is produced by TE's own device code on
// the same operands as in TE's kernel:
//   - the 16 x 16 block amax: ptx::abs_max_2x from 0 over bf16 pairs (max |x| of the block, NaN skipped -- the same
//     value in any grouping, since every partial starts from 0), then __hmax(__habs(.x), __habs(.y)) to float and fmaxf
//     across the lanes of the block, as TE reduces it;
//   - S_enc / S_dec (compute_global_encode_scaling_factor_FP4, 1.0 / S_enc), the E4M3 block scale
//     (compute_decoding_scaling_factor) and the block encode scale fminf(1.0f / (S_dec_b * S_dec), FLT_MAX) as TE's
//     kernel writes them, per direction;
//   - ptx::mul_cvt_bf16_to_fp4_4x<false> on the same four elements: four row neighbours for the row-wise data, four
//     column neighbours (gathered with byte permutes) for the column-wise data.
// Only the data movement differs from TE's kernel: each thread owns a 16-row x 8-column piece of the weight (sixteen
// 16-byte loads straight from global memory, no shared-memory pass), two lanes share a 16 x 16 block (one shuffle for
// its amax), a warp covers 64 rows x 64 columns; the row-wise data leaves as one 4-byte store per row (8 lanes: 32
// contiguous bytes), the column-wise data as one 8-byte store per column (4 lanes: 32 contiguous bytes), and the scales
// are staged in shared memory and written as 16- / 8- / 4-byte pieces of the GEMM-swizzled (or compact) layout.
// Template: WR warps along the rows (CTA tile 64 WR x 128, 64 WR threads), SWZ (GEMM-swizzled scales), COLW (write the
// column-wise output), LD (load flavour: 0 ld.global.nc, 1 + L1::no_allocate, 2 + L1::no_allocate.L2::256B).
// Requires rows % (64 WR) == 0 (and % 128 with SWZ), cols % 128 == 0, contiguous buffers.
#pragma once
namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace ak_w2d {
using namespace quantize_transpose_kernel;
using namespace quantization_and_transposition_SF;
using namespace core;
using namespace ptx;
#if FP4_TYPE_SUPPORTED

template <int LD>
__device__ __forceinline__ uint4 w2r_load(const uint4 *p) {
  uint4 v;
  if constexpr (LD == 0) {
    v = __ldg(p);
  } else if constexpr (LD == 1) {
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
  } else {
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
  }
  return v;
}

__device__ __forceinline__ uint32_t w2r_word(const uint4 &v, const int w) {
  return w == 0 ? v.x : (w == 1 ? v.y : (w == 2 ? v.z : v.w));
}

__device__ __forceinline__ uint32_t w2r_bits(const fp4e2m1x4 f) {
  return static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&f));
}

template <int WR, bool SWZ, bool COLW, int LD>
__global__ void __launch_bounds__(64 * WR)
    w2d_reg_kernel(const uint4 *const x, uint32_t *const row_data, nvfp4_scale_t *const scales_ptr,
                   uint2 *const col_data, nvfp4_scale_t *const scales_t_ptr, const float *noop,
                   const float *const amax_rowwise_ptr, const float *const amax_colwise_ptr, const size_t rows,
                   const size_t cols, const size_t scale_stride, const size_t scale_stride_t) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType = bf16;
  using IType2 = typename ptx::FPx2<IType>;
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  constexpr int ROWS_CTA = 64 * WR;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 8, j = lane % 8;
  const int rt = (warp / 2) * 64 + g * 16;  // the thread's first row within the CTA tile
  const int ct = (warp % 2) * 64 + j * 8;   // its first column within the CTA tile
  const size_t r0 = static_cast<size_t>(blockIdx.y) * ROWS_CTA + rt;
  const size_t c0 = static_cast<size_t>(blockIdx.x) * 128 + ct;

  // 1. the 16 x 8 piece: row i in v[i] (columns c0 .. c0 + 7, four bf16 pairs)
  uint4 v[16];
  {
    const uint4 *p = x + (r0 * cols + c0) / 8;
    const size_t ld = cols / 8;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      v[i] = w2r_load<LD>(p + i * ld);
    }
  }

  // TE's global encode / decode scales
  const float S_enc_rowwise = (amax_rowwise_ptr == nullptr)
                                  ? 1.0f
                                  : compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
  const float S_dec_rowwise = 1.0 / S_enc_rowwise;
  const float S_enc_colwise = (amax_colwise_ptr == nullptr)
                                  ? S_enc_rowwise
                                  : compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);
  const float S_dec_colwise = 1.0 / S_enc_colwise;
  constexpr float float_max = detail::TypeExtrema<float>::max;

  // 2. the block amax (the thread's 64 pairs in four chains from 0, then the other half of the block: lane ^ 1)
  IType2 acc[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    acc[w] = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t word = w2r_word(v[i], w);
      ptx::abs_max_2x(acc[w], acc[w], reinterpret_cast<const IType2 &>(word));
    }
  }
  ptx::abs_max_2x(acc[0], acc[0], acc[1]);
  ptx::abs_max_2x(acc[2], acc[2], acc[3]);
  ptx::abs_max_2x(acc[0], acc[0], acc[2]);
  const float thread_amax = static_cast<float>(__hmax(__habs(acc[0].x), __habs(acc[0].y)));
  const float block_amax = fmaxf(thread_amax, __shfl_xor_sync(0xffffffffu, thread_amax, 1));

  // 3. row-wise: TE's block scale and encode scale; per row, columns c0 .. c0 + 3 and c0 + 4 .. c0 + 7
  const nvfp4_scale_t S_dec_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc_rowwise);
  {
    const float block_scale_inverse =
        fminf(1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);
    const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
    uint32_t *const rowp = row_data + (r0 * cols + c0) / 8;
    const size_t ld = cols / 8;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint64_t e03 = (static_cast<uint64_t>(v[i].y) << 32) | v[i].x;
      const uint64_t e47 = (static_cast<uint64_t>(v[i].w) << 32) | v[i].z;
      const uint32_t lo = w2r_bits(ptx::mul_cvt_bf16_to_fp4_4x<false>(e03, block_scale_inverse_2x, 0));
      const uint32_t hi = w2r_bits(ptx::mul_cvt_bf16_to_fp4_4x<false>(e47, block_scale_inverse_2x, 0));
      rowp[i * ld] = lo | (hi << 16);
    }
  }

  // 4. column-wise: per column k, rows 4q .. 4q + 3 (q = 0 .. 3) gathered from the row pairs
  nvfp4_scale_t S_dec_b_fp8_t{};
  if constexpr (COLW) {
    S_dec_b_fp8_t = compute_decoding_scaling_factor(block_amax, S_enc_colwise);
    const float block_scale_inverse =
        fminf(1.0f / (static_cast<float>(S_dec_b_fp8_t) * S_dec_colwise), float_max);
    const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
    uint2 *const colp = col_data + (c0 * rows + r0) / 16;
    const size_t ld = rows / 16;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const uint32_t sel = (k & 1) ? 0x7632u : 0x5410u;
      uint32_t q16[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t r01 = __byte_perm(w2r_word(v[4 * q], k / 2), w2r_word(v[4 * q + 1], k / 2), sel);
        const uint32_t r23 = __byte_perm(w2r_word(v[4 * q + 2], k / 2), w2r_word(v[4 * q + 3], k / 2), sel);
        const uint64_t e = (static_cast<uint64_t>(r23) << 32) | r01;
        q16[q] = w2r_bits(ptx::mul_cvt_bf16_to_fp4_4x<false>(e, block_scale_inverse_2x, 0));
      }
      colp[k * ld] = make_uint2(q16[0] | (q16[1] << 16), q16[2] | (q16[3] << 16));
    }
  }

  // 5. the scales through shared memory: the CTA's row-wise [64 WR rows x 8] and column-wise [128 x 4 WR] pieces
  __shared__ alignas(16) uint8_t sc_row_sh[ROWS_CTA * 8];
  __shared__ alignas(16) uint8_t sc_col_sh[COLW ? ROWS_CTA * 8 : 16];
  {
    const int sc = (warp % 2) * 4 + j / 2;  // row-wise scale column within the CTA (0 .. 7)
    const uint8_t b = *reinterpret_cast<const uint8_t *>(&S_dec_b_fp8);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      const int r = rt + (j & 1) * 8 + m;
      int idx;
      if constexpr (SWZ && WR == 2) {
        idx = (sc / 4) * 512 + (r % 32) * 16 + (r / 32) * 4 + (sc % 4);
      } else if constexpr (SWZ) {
        idx = (sc / 4) * 256 + (r % 32) * 8 + (r / 32) * 4 + (sc % 4);
      } else {
        idx = r * 8 + sc;
      }
      sc_row_sh[idx] = b;
    }
    if constexpr (COLW) {
      const int tc = rt / 16;  // column-wise scale column within the CTA (0 .. 4 WR - 1)
      const uint8_t bt = *reinterpret_cast<const uint8_t *>(&S_dec_b_fp8_t);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const int oc = ct + m;
        int idx;
        if constexpr (SWZ && WR == 2) {
          idx = (tc / 4) * 512 + (oc % 32) * 16 + (oc / 32) * 4 + (tc % 4);
        } else if constexpr (SWZ) {
          idx = (oc % 32) * 16 + (oc / 32) * 4 + tc;
        } else {
          idx = oc * (4 * WR) + tc;
        }
        sc_col_sh[idx] = bt;
      }
    }
  }
  __syncthreads();
  const int t = threadIdx.x;
  if constexpr (SWZ) {
    const size_t cbg_cnt = (cols / SCALE_DIM + 3) / 4;
    const size_t cbg_cnt_t = (rows / SCALE_DIM + 3) / 4;
    if constexpr (WR == 2) {
      // both 1 KB blocks: row-wise at (blockIdx.y cbg_cnt + 2 blockIdx.x) 512, column-wise at
      // (blockIdx.x cbg_cnt_t + 2 blockIdx.y) 512
      if (t < 64) {
        reinterpret_cast<uint4 *>(scales_ptr + (blockIdx.y * cbg_cnt + 2 * blockIdx.x) * 512)[t] =
            reinterpret_cast<const uint4 *>(sc_row_sh)[t];
      } else if constexpr (COLW) {
        reinterpret_cast<uint4 *>(scales_t_ptr + (blockIdx.x * cbg_cnt_t + 2 * blockIdx.y) * 512)[t - 64] =
            reinterpret_cast<const uint4 *>(sc_col_sh)[t - 64];
      }
    } else {
      // row-wise: 64 pieces of 8 bytes (tile u = t / 32, line d3 = t % 32) in the CTA's half of each 16-byte line;
      // column-wise: one 512-byte tile, 32 pieces of 16 bytes
      const int u = t / 32, d3 = t % 32;
      *reinterpret_cast<uint2 *>(scales_ptr + (((blockIdx.y / 2) * cbg_cnt + 2 * blockIdx.x + u) * 32 + d3) * 16 +
                                 (blockIdx.y % 2) * 8) = reinterpret_cast<const uint2 *>(sc_row_sh)[t];
      if constexpr (COLW) {
        if (t < 32) {
          reinterpret_cast<uint4 *>(scales_t_ptr + (blockIdx.x * cbg_cnt_t + blockIdx.y) * 512)[t] =
              reinterpret_cast<const uint4 *>(sc_col_sh)[t];
        }
      }
    }
  } else {
    // compact: row r of the row-wise piece (8 bytes) at (blockIdx.y 64 WR + r) scale_stride + 8 blockIdx.x; column-wise
    // row oc (4 WR bytes) at (128 blockIdx.x + oc) scale_stride_t + 4 WR blockIdx.y
    for (int r = t; r < ROWS_CTA; r += 64 * WR) {
      *reinterpret_cast<uint2 *>(scales_ptr + (static_cast<size_t>(blockIdx.y) * ROWS_CTA + r) * scale_stride +
                                 8 * blockIdx.x) = reinterpret_cast<const uint2 *>(sc_row_sh)[r];
    }
    if constexpr (COLW) {
      for (int oc = t; oc < 128; oc += 64 * WR) {
        nvfp4_scale_t *const dst =
            scales_t_ptr + (static_cast<size_t>(blockIdx.x) * 128 + oc) * scale_stride_t + 4 * WR * blockIdx.y;
        if constexpr (WR == 2) {
          *reinterpret_cast<uint2 *>(dst) = reinterpret_cast<const uint2 *>(sc_col_sh)[oc];
        } else {
          *reinterpret_cast<uint32_t *>(dst) = reinterpret_cast<const uint32_t *>(sc_col_sh)[oc];
        }
      }
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace ak_w2d
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine
