// binding8.cu (Llama 3.1 8B campaign, 2026-10-08, rx26): the torch entry points of ak_l8b_rht8 -- TE's fused NVFP4 row
// + column-RHT cast (rhtdev's variants) with fewer instructions in its epilogue warps, the same bytes:
//   rht_cast_v  (round to nearest; inst8_rn*.cu): ak_l8b_rht6's rht_cast_v for its RN variants with bit 256 added
//               (326 = 70 + 256, 359 = 103 + 256: each block's reciprocal scale from a per-CTA table);
//   rht_cast_vp (stochastic rounding; inst8_sr*.cu): ak_l8b_rht7's rht_cast_vp (the generator's Philox state by value)
//               with bit 256 (335 = 79 + 256, 355 = 99 + 256) and bit 512 (the draws' shared multiplies once, the work
//               loop unswitched per launch: 847 = 79 + 256 + 512, 867 = 99 + 256 + 512).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
namespace ak_rht8 {
template <bool kSR, int kVar>
void launch_v8(int64_t m, int64_t n, const at::Tensor& x, const at::Tensor& had, const at::Tensor& col_data,
               const at::Tensor& col_si, const at::Tensor& row_data, const at::Tensor& row_si, const at::Tensor& amax_row,
               const at::Tensor& amax_col, const size_t* rng, int sms, cudaStream_t stream, int k_tile);
template <bool kSR, int kVar>
void launch_vp8(int64_t m, int64_t n, const at::Tensor& x, const at::Tensor& had, const at::Tensor& col_data,
                const at::Tensor& col_si, const at::Tensor& row_data, const at::Tensor& row_si, const at::Tensor& amax_row,
                const at::Tensor& amax_col, at::PhiloxCudaState philox, int sms, cudaStream_t stream, int k_tile);
}  // namespace ak_rht8

// rht_cast_v(x, amax_row, amax_col, row_data, row_si, col_data, col_si, rht_matrix, sr, k_tile, variant): round to
// nearest only (sr must be false), as ak_l8b_rht6's rht_cast_v without the generator
void rht_cast_v(at::Tensor x, at::Tensor amax_row, at::Tensor amax_col, at::Tensor row_data, at::Tensor row_si,
                at::Tensor col_data, at::Tensor col_si, at::Tensor rht_matrix, bool sr, int64_t k_tile, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x: bf16 2D");
  TORCH_CHECK(rht_matrix.scalar_type() == at::kBFloat16 && rht_matrix.numel() == 256, "rht_matrix: bf16 16x16");
  TORCH_CHECK(!sr, "rht_cast_v: round to nearest only (rht_cast_vp for stochastic rounding)");
  const int64_t m = x.size(0), n = x.size(1);
  TORCH_CHECK(m % 64 == 0 && n % 128 == 0, "rows % 64, cols % 128");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int kt = static_cast<int>(k_tile);
  switch (variant) {
    case 326:
      ak_rht8::launch_v8<false, 326>(m, n, x, rht_matrix, col_data, col_si, row_data, row_si, amax_row, amax_col, nullptr,
                                     sms, stream, kt);
      break;
    case 359:
      ak_rht8::launch_v8<false, 359>(m, n, x, rht_matrix, col_data, col_si, row_data, row_si, amax_row, amax_col, nullptr,
                                     sms, stream, kt);
      break;
    default: TORCH_CHECK(false, "rht_cast_v: variant not built: ", variant);
  }
}

// rht_cast_vp(x, amax_row, amax_col, row_data, row_si, col_data, col_si, rht_matrix, sr, k_tile, variant): stochastic
// rounding, the generator's Philox state by value -- the same philox_cuda_state(1024) per call as ak_l8b_rht6/7
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
#define AK_RHT8_VP(V) ak_rht8::launch_vp8<true, V>(m, n, x, rht_matrix, col_data, col_si, row_data, row_si, amax_row, \
                                                   amax_col, philox, sms, stream, kt)
  switch (variant) {
    case 335: AK_RHT8_VP(335); break;
    case 355: AK_RHT8_VP(355); break;
    case 847: AK_RHT8_VP(847); break;
    case 867: AK_RHT8_VP(867); break;
    default: TORCH_CHECK(false, "rht_cast_vp: variant not built: ", variant);
  }
#undef AK_RHT8_VP
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rht_cast_v", &rht_cast_v, "rx26: TE's NVFP4 row + column-RHT cast, round to nearest (variants 326, 359)");
  m.def("rht_cast_vp", &rht_cast_vp,
        "rx26: TE's NVFP4 row + column-RHT cast, stochastic rounding, Philox state by value (335, 355, 847, 867)");
}
