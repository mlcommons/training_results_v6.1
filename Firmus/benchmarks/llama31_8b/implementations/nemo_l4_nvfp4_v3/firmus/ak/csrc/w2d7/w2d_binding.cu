// w2d_binding.cu: the torch entry point of ak_l8b_w2d (one translation unit per variant: inst_w*.cu).
//   w2d_cast(x, row_data, row_si, col_data, col_si, amax_row, amax_col, swz, variant)
// writes TE's NVFP4 2D cast of the bf16 weight x [rows, cols] into the given buffers (col_data empty: row-wise only).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
namespace ak_w2d {
template <int CY, int PRE, int NB, bool SCS, bool CREG, bool BIGT>
void launch_w2d(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                const at::Tensor &, const at::Tensor &, bool, cudaStream_t);
template <int WR, int LD>
void launch_w2r(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                const at::Tensor &, const at::Tensor &, bool, cudaStream_t);
}

void w2d_cast(at::Tensor x, at::Tensor row_data, at::Tensor row_si, at::Tensor col_data, at::Tensor col_si,
              at::Tensor amax_row, at::Tensor amax_col, bool swz, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: bf16 2D");
  TORCH_CHECK(amax_row.scalar_type() == at::kFloat && amax_row.numel() == 1, "amax_row: fp32 [1]");
  TORCH_CHECK(col_data.numel() == 0 || (amax_col.scalar_type() == at::kFloat && amax_col.numel() == 1), "amax_col");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define AK_W2D_GO(CY, PRE, NB, SCS, CREG, BIGT) ak_w2d::launch_w2d<CY, PRE, NB, SCS, CREG, BIGT>(x, row_data, row_si, col_data, col_si, amax_row, amax_col, swz, st)
#define AK_W2R_GO(WR, LD) ak_w2d::launch_w2r<WR, LD>(x, row_data, row_si, col_data, col_si, amax_row, amax_col, swz, st)
  switch (variant) {
#ifdef AK_W2D_HAVE_V0
    case 0: AK_W2D_GO(128, 1, 2, false, false, false); break;     // TE's schedule
#endif
#ifdef AK_W2D_HAVE_V1
    case 1: AK_W2D_GO(64, 1, 2, false, false, false); break;      // 64 rows per CTA
#endif
#ifdef AK_W2D_HAVE_V2
    case 2: AK_W2D_GO(32, 1, 1, false, false, false); break;      // 32 rows per CTA (one stage)
#endif
#ifdef AK_W2D_HAVE_V3
    case 3: AK_W2D_GO(128, 4, 4, false, false, false); break;     // TE's rows, all four loads issued up front
#endif
#ifdef AK_W2D_HAVE_V4
    case 4: AK_W2D_GO(64, 2, 2, false, false, false); break;      // 64 rows, both loads up front
#endif
#ifdef AK_W2D_HAVE_V5
    case 5: AK_W2D_GO(128, 1, 2, true, false, false); break;      // TE's schedule, the swizzled scales staged: 2 x 1 KB stores
#endif
#ifdef AK_W2D_HAVE_V6
    case 6: AK_W2D_GO(128, 1, 2, false, true, false); break;      // TE's schedule, the column-wise pass from registers
#endif
#ifdef AK_W2D_HAVE_V7
    case 7: AK_W2D_GO(128, 1, 2, true, true, false); break;       // both
#endif
#ifdef AK_W2D_HAVE_V8
    case 8: AK_W2D_GO(128, 1, 2, true, true, true); break;        // both + the column-wise output in 64-byte rows
#endif
#ifdef AK_W2D_HAVE_V9
    case 9: AK_W2D_GO(128, 1, 2, false, false, true); break;      // the column-wise output in 64-byte rows only
#endif
#ifdef AK_W2D_HAVE_V10
    case 10: AK_W2D_GO(64, 1, 2, true, true, false); break;      // 64 rows per CTA, staged scales, registers
#endif
#ifdef AK_W2D_HAVE_V20
    case 20: AK_W2R_GO(2, 0); break;      // register tiles: 128 x 128 per CTA (4 warps), ld.global.nc
#endif
#ifdef AK_W2D_HAVE_V21
    case 21: AK_W2R_GO(1, 0); break;      // register tiles: 64 x 128 per CTA (2 warps), ld.global.nc
#endif
#ifdef AK_W2D_HAVE_V22
    case 22: AK_W2R_GO(2, 1); break;      // 128 x 128, + L1::no_allocate
#endif
#ifdef AK_W2D_HAVE_V23
    case 23: AK_W2R_GO(1, 1); break;      // 64 x 128, + L1::no_allocate
#endif
#ifdef AK_W2D_HAVE_V24
    case 24: AK_W2R_GO(2, 2); break;      // 128 x 128, + L1::no_allocate.L2::256B
#endif
#ifdef AK_W2D_HAVE_V25
    case 25: AK_W2R_GO(1, 2); break;      // 64 x 128, + L1::no_allocate.L2::256B
#endif
    default: TORCH_CHECK(false, "variant not built: ", variant);
  }
#undef AK_W2D_GO
#undef AK_W2R_GO
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w2d_cast", &w2d_cast, "TE's NVFP4 2D weight cast (standalone copy, schedule variants)");
}
