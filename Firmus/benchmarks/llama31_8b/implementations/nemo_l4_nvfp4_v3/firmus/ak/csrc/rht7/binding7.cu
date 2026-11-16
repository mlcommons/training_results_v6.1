// binding7.cu: the torch entry points of ak_l8b_rht7 (NVFP4P version 14): the RMSNorm forward whose apply is folded into
// the RHT cast -- rmsnorm_amax (norm_amax.cu: the two amaxes of z, z not written), rmsnorm_stats (TE's rsigma) and
// rht_cast_norm (inst_n70.cu / norm_variant.cuh: TE's fused row + column-RHT cast forming z from x, rsigma and gamma) --
// and rht_cast_vp (inst_p79.cu / inst_p99.cu / philox_variant.cuh: version 12's stochastic-rounding casts with the
// generator's Philox state passed by value, without the one-thread unpack kernel in front)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
std::vector<at::Tensor> rmsnorm_amax(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& rs, int64_t sign_mask,
                                     int64_t version);
at::Tensor rmsnorm_stats(const at::Tensor& x, double eps);
std::vector<at::Tensor> rmsnorm_apply_amax7(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& rs,
                                            int64_t sign_mask, int64_t version);
void sm_hog(int64_t ctas, int64_t ns, at::Tensor sink);
namespace ak_rht {
template <int kVar>
void launch_vn(int64_t m, int64_t n, const at::Tensor& x, const at::Tensor& had, const at::Tensor& col_data,
               const at::Tensor& col_si, const at::Tensor& row_data, const at::Tensor& row_si, const at::Tensor& amax_row,
               const at::Tensor& amax_col, const at::Tensor& gamma, const at::Tensor& rs, int sms, cudaStream_t stream,
               int k_tile);
template <bool kSR, int kVar>
void launch_vp(int64_t m, int64_t n, const at::Tensor& x, const at::Tensor& had, const at::Tensor& col_data,
               const at::Tensor& col_si, const at::Tensor& row_data, const at::Tensor& row_si, const at::Tensor& amax_row,
               const at::Tensor& amax_col, at::PhiloxCudaState philox, int sms, cudaStream_t stream, int k_tile);
}  // namespace ak_rht

// rht_cast_norm(x [m, n] bf16, gamma [n] bf16, rs [m] f32, amax_row, amax_col, row_data, row_si, col_data, col_si,
// rht_matrix, k_tile, variant): tex.nvfp4_quantize_with_amax(z, q, amax_row, amax_col) for the recipe's 1D RHT
// quantizer (row + column, GEMM-swizzled scales, round to nearest), z = bf16(gamma * (rs * x)) formed inside the cast
void rht_cast_norm(at::Tensor x, at::Tensor gamma, at::Tensor rs, at::Tensor amax_row, at::Tensor amax_col,
                   at::Tensor row_data, at::Tensor row_si, at::Tensor col_data, at::Tensor col_si, at::Tensor rht_matrix,
                   int64_t k_tile, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: bf16 2D");
  const int64_t m = x.size(0), n = x.size(1);
  TORCH_CHECK(m % 64 == 0 && n % 128 == 0, "rows % 64, cols % 128");
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() && gamma.numel() == n,
              "gamma: bf16 [cols]");
  TORCH_CHECK(rs.is_cuda() && rs.scalar_type() == at::kFloat && rs.is_contiguous() && rs.numel() == m, "rs: f32 [rows]");
  TORCH_CHECK(rht_matrix.scalar_type() == at::kBFloat16 && rht_matrix.numel() == 256, "rht_matrix: bf16 16x16");
  TORCH_CHECK(amax_row.scalar_type() == at::kFloat && amax_col.scalar_type() == at::kFloat, "amaxes: f32");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  switch (variant) {
    case 70:
      ak_rht::launch_vn<70>(m, n, x, rht_matrix, col_data, col_si, row_data, row_si, amax_row, amax_col, gamma, rs, sms,
                            stream, static_cast<int>(k_tile));
      break;
    case 198:
      ak_rht::launch_vn<198>(m, n, x, rht_matrix, col_data, col_si, row_data, row_si, amax_row, amax_col, gamma, rs, sms,
                             stream, static_cast<int>(k_tile));
      break;
    default: TORCH_CHECK(false, "rht_cast_norm: variant not built: ", variant);
  }
}

// rht_cast_vp(x, amax_row, amax_col, row_data, row_si, col_data, col_si, rht_matrix, sr, k_tile, variant): ak_l8b_rht6's
// rht_cast_v for its stochastic-rounding variants (79, 99), the generator's Philox state handed to the cast by value:
// the same philox_cuda_state(1024) per call, the same unpack, inside the cast's epilogue warps
void rht_cast_vp(at::Tensor x, at::Tensor amax_row, at::Tensor amax_col, at::Tensor row_data, at::Tensor row_si,
                 at::Tensor col_data, at::Tensor col_si, at::Tensor rht_matrix, bool sr, int64_t k_tile, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: bf16 2D");
  TORCH_CHECK(rht_matrix.scalar_type() == at::kBFloat16 && rht_matrix.numel() == 256, "rht_matrix: bf16 16x16");
  TORCH_CHECK(sr, "rht_cast_vp: stochastic rounding only");
  const int64_t m = x.size(0), n = x.size(1);
  TORCH_CHECK(m % 64 == 0 && n % 128 == 0, "rows % 64, cols % 128");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    philox = gen->philox_cuda_state(1024);
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int kt = static_cast<int>(k_tile);
  switch (variant) {
    case 79:
      ak_rht::launch_vp<true, 79>(m, n, x, rht_matrix, col_data, col_si, row_data, row_si, amax_row, amax_col, philox, sms,
                                  stream, kt);
      break;
    case 99:
      ak_rht::launch_vp<true, 99>(m, n, x, rht_matrix, col_data, col_si, row_data, row_si, amax_row, amax_col, philox, sms,
                                  stream, kt);
      break;
    default: TORCH_CHECK(false, "rht_cast_vp: variant not built: ", variant);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_amax", &rmsnorm_amax, "the pre/post-RHT amaxes of z = bf16(gamma * (rs * x)), z not written",
        py::arg("x"), py::arg("gamma"), py::arg("rs"), py::arg("sign_mask"), py::arg("version") = 4);
  m.def("rmsnorm_stats", &rmsnorm_stats, "TE's RMSNorm rsigma (hidden 4096)", py::arg("x"), py::arg("eps"));
  m.def("rmsnorm_apply_amax7", &rmsnorm_apply_amax7, "ak_l8b_nvfp4n's apply v6 (z written + its amaxes), persistent",
        py::arg("x"), py::arg("gamma"), py::arg("rs"), py::arg("sign_mask"), py::arg("version") = 3);
  m.def("rht_cast_norm", &rht_cast_norm, "TE's NVFP4 row + column-RHT cast of z = bf16(gamma * (rs * x)), z formed in the cast",
        py::arg("x"), py::arg("gamma"), py::arg("rs"), py::arg("amax_row"), py::arg("amax_col"), py::arg("row_data"),
        py::arg("row_si"), py::arg("col_data"), py::arg("col_si"), py::arg("rht_matrix"), py::arg("k_tile"),
        py::arg("variant"));
  m.def("sm_hog", &sm_hog, "test only: holds `ctas` SMs for `ns` nanoseconds (a stand-in for a concurrent collective)");
  m.def("rht_cast_vp", &rht_cast_vp, "ak_l8b_rht6's SR cast variants with the Philox state passed by value");
}
