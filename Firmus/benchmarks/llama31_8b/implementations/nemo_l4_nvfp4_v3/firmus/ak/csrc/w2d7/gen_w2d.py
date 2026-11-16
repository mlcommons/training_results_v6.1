"""gen_w2d.py: writes w2d_kernel.cuh -- TE's quantize_transpose_nvfp4_2D_kernel (cast/nvfp4/quantize_transpose_nvfp4.cuh)
copied verbatim into namespace transformer_engine::dispatch::nvfp4::ak_w2d (so every helper and constant it names
resolves to TE's own), renamed quantize_transpose_nvfp4_2D_kernel_v, with three template parameters that change only
the schedule:
  CY_V  rows per CTA (TE: CHUNK_DIM_Y = 128; 64 or 32 give 2 or 1 stages of 32 rows),
  NB_V  input / output staging buffers (TE: BUFFS_NUM = 2),
  PRE_V stages whose loads are issued before the first compute (TE: 1; = the stage count loads all at once, then each
        stage owns its buffers and no store-completion wait is needed).
  SCS_V (with GEMM-swizzled scales and CY_V 128) the scales staged in shared memory and stored as two contiguous 1 KB
        blocks with 16-byte stores: in the swizzled layout a CTA's 128 rows x 8 row-wise scale columns, and its 128
        columns x 8 column-wise scale columns, are each one contiguous 1 KB block (two adjacent 512-byte 128 x 4 tiles);
        TE writes them as scattered single bytes (the same bytes at the same offsets);
  CREG_V the column-wise pass takes its 32 elements per thread from registers: the block-amax pass loads exactly those
        elements (thread t: column t, rows 0..31 of the stage tile) and TE loads them again from shared memory;
  BIGT_V (CY_V 128, with the transpose) the CTA's column-wise output, 128 rows x 64 bytes, collected in shared memory
        (rows padded to 72 bytes: conflict-free 8-byte writes) and stored after the last stage with 16-byte stores,
        full 64-byte row segments; TE stores each stage's 128 x 16-byte slice by TMA (the same bytes at the same
        offsets).  The kernel takes the column-wise data pointer as a last argument.
Every element, block amax, scale and its global position are computed exactly as TE's: the 16 x 16 blocks lie inside
one 32 x 128 stage tile whatever the chunking.  The stochastic-rounding subsequence (threadIdx / blockIdx / gridDim)
does change with CY_V, so only CY_V = 128 is TE's under SR (the weights are cast round-to-nearest).
Usage: python3 gen_w2d.py <TE quantize_transpose_nvfp4.cuh> <out w2d_kernel.cuh>"""
import re
import argparse
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("src")
_ap.add_argument("out")
_args = _ap.parse_args()
src, out = _args.src, _args.out
s = open(src).read()
a = s.index("quantize_transpose_nvfp4_2D_kernel(const __grid_constant__")
start = s.rindex("template <bool COMPUTE_ACTIVATIONS", 0, a)
e = s.index("#endif  // FP4_TYPE_SUPPORTED\n}  // namespace quantize_transpose_kernel", a)
k = s[start:e].rstrip() + "\n"
# signature
old_t = "bool WITH_GEMM_SWIZZLED_SCALES = false>"
assert k.count(old_t) == 1
k = k.replace(old_t, "bool WITH_GEMM_SWIZZLED_SCALES = false, int CY_V = 128, int PRE_V = 1, int NB_V = 2, bool SCS_V = false, bool CREG_V = false, bool BIGT_V = false>")
assert k.count("quantize_transpose_nvfp4_2D_kernel(const __grid_constant__") == 1
k = k.replace("quantize_transpose_nvfp4_2D_kernel(const __grid_constant__", "quantize_transpose_nvfp4_2D_kernel_v(const __grid_constant__")
# body: the schedule constants
sig_end = k.index("{", k.index("const size_t *rng_state)"))
head, body = k[:sig_end + 1], k[sig_end + 1:]
for name, new in (("STAGES", "STAGES_V"), ("BUFFS_NUM", "BUFFS_NUM_V"), ("CHUNK_DIM_Y", "CHUNK_DIM_Y_V"),
                  ("SCALES_PER_CHUNK_Y", "SCALES_PER_CHUNK_Y_V")):
    body, n = re.subn(r"\b%s\b" % name, new, body)
    print(name, n)
guard = "#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)\n"
assert body.startswith("\n" + guard)          # (the closing #endif's comment repeats the text)
body = body.replace(guard, guard + """  // ---- the schedule of this copy (TE's: CY_V 128, NB_V 2, PRE_V 1)
  constexpr size_t CHUNK_DIM_Y_V = CY_V;
  constexpr size_t STAGES_V = CHUNK_DIM_Y_V / TILE_DIM_Y;
  constexpr size_t BUFFS_NUM_V = NB_V;
  constexpr size_t SCALES_PER_CHUNK_Y_V = CHUNK_DIM_Y_V / SCALE_DIM;
  constexpr size_t PRELOAD_V = PRE_V;
  static_assert(STAGES_V >= 1 && STAGES_V * TILE_DIM_Y == CHUNK_DIM_Y_V, "rows per CTA: a multiple of 32");
  static_assert(PRELOAD_V >= 1 && PRELOAD_V <= STAGES_V && PRELOAD_V <= BUFFS_NUM_V, "preloads need their own buffers");
""", 1)
old_load = """  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                    &mbar[0], is_master_thread);
"""
assert body.count(old_load) == 1
body = body.replace(old_load, """#pragma unroll
  for (size_t ps = 0; ps < PRELOAD_V; ++ps) {
    copy_2d_to_shared(&in_sh[ps * BUFF_IN_SIZE], &tensor_map_input, block_offset_X,
                      block_offset_Y + ps * BUFF_DIM_Y, shmem_buff_size, &mbar[ps], is_master_thread);
  }
""")
old_next = "    if (next_stage < STAGES_V) {\n      // Wait for TMA transfer to have finished reading shared memory."
assert body.count(old_next) == 1
body = body.replace(old_next, "    if (next_stage < STAGES_V && next_stage >= PRELOAD_V) {\n      // Wait for TMA transfer to have finished reading shared memory.")
# ---- SCS_V: the scales staged in shared memory, two contiguous 1 KB stores
decl_old = "  __shared__ __align__(16) float block_amax_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1];\n"
assert body.count(decl_old) == 1
body = body.replace(decl_old, decl_old + """  // SCS_V: the CTA's swizzled scale blocks (row-wise, column-wise), 1 KB each
  constexpr bool SCS = SCS_V && WITH_GEMM_SWIZZLED_SCALES && (CHUNK_DIM_Y_V == 128 || CHUNK_DIM_Y_V == 64);
  __shared__ __align__(16) nvfp4_scale_t row_scales_stage_sh[SCS ? 1024 : 16];
  __shared__ __align__(16) nvfp4_scale_t col_scales_stage_sh[SCS ? 1024 : 16];
""")
col_old = "        out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;\n"
assert body.count(col_old) == 1
body = body.replace(col_old, """        if constexpr (SCS) {
          // (row tid_Y_t, scale column stage * 2 + it) of the CTA's 128 x 8 block, at its swizzled offset
          const size_t ct = stage * ITERATIONS_TRANSPOSE + it;     // < 8 (CY 128) or < 4 (CY 64: one tile)
          col_scales_stage_sh[(ct / 4) * 512 + (tid_Y_t % 32) * 16 + (tid_Y_t / 32) * 4 + (ct % 4)] = S_dec_b_fp8;
        } else {
          out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;
        }
""")
row_old = """        if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_dec_b_fp8;
        }
"""
assert body.count(row_old) == 1
body = body.replace(row_old, """        if constexpr (SCS) {
          // (row stage * 32 + it * 16 + tid_Y_rowwise, scale column tid_X_rowwise) of the CTA's 128 x 8 block
          const size_t rr = stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise;
          if constexpr (CHUNK_DIM_Y_V == 128) {
            row_scales_stage_sh[(tid_X_rowwise / 4) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + (tid_X_rowwise % 4)] =
                S_dec_b_fp8;
          } else {
            // CY 64: the CTA holds half of each 128-row tile (d4 in {2 (blockIdx.y % 2), +1}): 8 bytes per 16-byte line
            row_scales_stage_sh[(tid_X_rowwise / 4) * 256 + (rr % 32) * 8 + (rr / 32) * 4 + (tid_X_rowwise % 4)] =
                S_dec_b_fp8;
          }
        } else if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_dec_b_fp8;
        }
""")
end_old = "  // Vectorized store scaling factors through SHMEM\n  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {"
assert body.count(end_old) == 1
body = body.replace(end_old, """  if constexpr (SCS) {
    // both 1 KB blocks: row-wise at ((blockIdx.y * cbg_cnt + 2 blockIdx.x) * 512), cbg_cnt = ceil(cols / 64);
    // column-wise at ((blockIdx.x * cbg_cnt_t + 2 blockIdx.y) * 512), cbg_cnt_t = ceil(rows / 64)
    __syncthreads();
    const size_t i = threadIdx.x % 64;
    if constexpr (CHUNK_DIM_Y_V == 128) {
      if (threadIdx.x < 64) {
        if constexpr (RETURN_ROWWISE) {
          const size_t cbg_cnt = (cols / SCALE_DIM + 3) / 4;
          uint4 *dst = reinterpret_cast<uint4 *>(scales_ptr + (blockIdx.y * cbg_cnt + 2 * blockIdx.x) * 512);
          dst[i] = reinterpret_cast<const uint4 *>(row_scales_stage_sh)[i];
        }
      } else {
        if constexpr (RETURN_TRANSPOSE) {
          const size_t cbg_cnt_t = (rows / SCALE_DIM + 3) / 4;
          uint4 *dst = reinterpret_cast<uint4 *>(scales_t_ptr + (blockIdx.x * cbg_cnt_t + 2 * blockIdx.y) * 512);
          dst[i] = reinterpret_cast<const uint4 *>(col_scales_stage_sh)[i];
        }
      }
    } else {
      // CY 64: row-wise 64 chunks of 8 bytes (tile t = i / 32, line d3 = i % 32) at the CTA's d4 pair; column-wise one
      // 512-byte tile (scale columns blockIdx.y * 4 .. + 3 of the transposed matrix), 32 chunks of 16 bytes
      if (threadIdx.x < 64) {
        if constexpr (RETURN_ROWWISE) {
          const size_t cbg_cnt = (cols / SCALE_DIM + 3) / 4;
          const size_t t = i / 32, d3 = i % 32;
          uint2 *dst = reinterpret_cast<uint2 *>(
              scales_ptr + (((blockIdx.y / 2) * cbg_cnt + 2 * blockIdx.x + t) * 32 + d3) * 16 + (blockIdx.y % 2) * 8);
          *dst = reinterpret_cast<const uint2 *>(row_scales_stage_sh)[i];
        }
      } else if (threadIdx.x < 96) {
        if constexpr (RETURN_TRANSPOSE) {
          const size_t cbg_cnt_t = (rows / SCALE_DIM + 3) / 4;
          uint4 *dst = reinterpret_cast<uint4 *>(scales_t_ptr + (blockIdx.x * cbg_cnt_t + blockIdx.y) * 512);
          dst[i] = reinterpret_cast<const uint4 *>(col_scales_stage_sh)[i];
        }
      }
    }
  } else
  // Vectorized store scaling factors through SHMEM
  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {""")
# ---- CREG_V: the column-wise pass from the amax pass's registers
blk_old = "    float block_amax = 0.0f;\n"
assert body.count(blk_old) == 1
body = body.replace(blk_old, blk_old + "    IType col_cache[CREG_V ? ITERATIONS_BLOCK : 1][CREG_V ? BLOCK_DIM : 1];  // CREG_V\n")
amax_old = """        for (int elem = 0; elem < BLOCK_DIM; elem += 2) {
          const size_t elem_0_row = block_iter * BLOCK_DIM + elem;
          const size_t elem_1_row = elem_0_row + 1;
          const size_t elem_0_col = warp_id * BLOCKS_PER_WARP * BLOCK_DIM + lane_id;
          const size_t elem_1_col = elem_0_col;

          const size_t shmem_offset_0 = buff_offset_in + elem_0_row * BUFF_IN_DIM_X + elem_0_col;
          const size_t shmem_offset_1 = buff_offset_in + elem_1_row * BUFF_IN_DIM_X + elem_1_col;

          IType2 val_2x;
          val_2x.x = in_sh[shmem_offset_0];
          val_2x.y = in_sh[shmem_offset_1];
          ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, val_2x);
        }
"""
assert body.count(amax_old) == 1
amax_new = """        if constexpr (CREG_V) {
          // TE's loop, its two loads kept for the column-wise pass
#pragma unroll
          for (int elem = 0; elem < BLOCK_DIM; elem += 2) {
            const size_t elem_0_row = block_iter * BLOCK_DIM + elem;
            const size_t elem_1_row = elem_0_row + 1;
            const size_t elem_0_col = warp_id * BLOCKS_PER_WARP * BLOCK_DIM + lane_id;
            const size_t elem_1_col = elem_0_col;

            const size_t shmem_offset_0 = buff_offset_in + elem_0_row * BUFF_IN_DIM_X + elem_0_col;
            const size_t shmem_offset_1 = buff_offset_in + elem_1_row * BUFF_IN_DIM_X + elem_1_col;

            IType2 val_2x;
            val_2x.x = in_sh[shmem_offset_0];
            val_2x.y = in_sh[shmem_offset_1];
            col_cache[block_iter][elem] = val_2x.x;
            col_cache[block_iter][elem + 1] = val_2x.y;
            ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, val_2x);
          }
        } else {
""" + amax_old + """        }
"""
body = body.replace(amax_old, amax_new)
col_old = "            in_colwise_IType[i] = in_sh[shmem_offset_colwise];\n"
assert body.count(col_old) == 1
body = body.replace(col_old, """            if constexpr (CREG_V) {
              in_colwise_IType[i] = col_cache[it][i];      // = in_sh[shmem_offset_colwise], loaded by the amax pass
            } else {
              in_colwise_IType[i] = in_sh[shmem_offset_colwise];
            }
""")
# ---- BIGT_V: the column-wise output collected per CTA, stored in 64-byte row segments
sig_old = "const size_t scale_stride_t, const size_t *rng_state) {"
assert head.count(sig_old) == 1
head = head.replace(sig_old, "const size_t scale_stride_t, const size_t *rng_state,\n                                       uint8_t *const col_data_ptr) {")
decl2 = "  __shared__ __align__(16) nvfp4_scale_t col_scales_stage_sh[SCS ? 1024 : 16];\n"
assert body.count(decl2) == 1
body = body.replace(decl2, decl2 + """  // BIGT_V: the CTA's 128 x 64-byte column-wise output, rows padded to 72 bytes
  constexpr bool BIGT = BIGT_V && RETURN_TRANSPOSE && (CHUNK_DIM_Y_V == 128);
  constexpr size_t BIGT_LD = 72;
  __shared__ __align__(16) uint8_t colT_sh[BIGT ? 128 * BIGT_LD : 16];
""")
mem_old = "  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;\n"
assert body.count(mem_old) == 1
body = body.replace(mem_old, "  constexpr size_t out_mem_colwise_data =\n      (BIGT_V && RETURN_TRANSPOSE && (CY_V == 128)) ? 0 : buff_size_aligned_out;  // BIGT_V: unused\n")
w_old = """        uint32_t *out_t_data_sh_as_uint32_t =
            reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);
        out_t_data_sh_as_uint32_t[group] = val[0];            // idx1 = (group + 0) % 2;
        out_t_data_sh_as_uint32_t[(group + 1) & 1] = val[1];  // idx2 = (group + 1) % 2;
"""
assert body.count(w_old) == 1
body = body.replace(w_old, """        if constexpr (BIGT) {
          // the same 8 bytes (word 0 = regs_4x[0], word 1 = regs_4x[1]) at row tid_Y_t, byte stage * 16 + it * 8
          *reinterpret_cast<uint2 *>(&colT_sh[tid_Y_t * BIGT_LD + stage * 16 + it * 8]) =
              make_uint2(regs_4x[0], regs_4x[1]);
        } else {
""" + "  " + w_old.replace("\n        ", "\n          ").rstrip(" ") + """        }
""")
t_old = """      if constexpr (RETURN_TRANSPOSE) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_t,"""
assert body.count(t_old) == 1
body = body.replace(t_old, """      if constexpr (RETURN_TRANSPOSE && !BIGT) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_t,""")
e_old = "  }  // end of stages\n"
assert body.count(e_old) == 1
body = body.replace(e_old, e_old + """
  if constexpr (BIGT) {
    // the CTA's transposed tile: rows block_offset_Y_t + r (r < 128), bytes block_offset_X_t / 2 + [0, 64)
    __syncthreads();
    const size_t ld_t = rows / 2;
    uint8_t *const base = col_data_ptr + block_offset_Y_t * ld_t + block_offset_X_t / 2;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q = threadIdx.x + 128 * j;
      const int r = q / 4, part = q % 4;
      const uint2 lo = *reinterpret_cast<const uint2 *>(&colT_sh[r * BIGT_LD + part * 16]);
      const uint2 hi = *reinterpret_cast<const uint2 *>(&colT_sh[r * BIGT_LD + part * 16 + 8]);
      *reinterpret_cast<uint4 *>(base + r * ld_t + part * 16) = make_uint4(lo.x, lo.y, hi.x, hi.y);
    }
  }
""")
hdr = """// w2d_kernel.cuh: generated by gen_w2d.py from TE's cast/nvfp4/quantize_transpose_nvfp4.cuh (see its docstring).
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
"""
ftr = """#endif  // FP4_TYPE_SUPPORTED
}  // namespace ak_w2d
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine
"""
open(out, "w").write(hdr + head + body + ftr)
print("wrote", out)
