// inst_p99.cu: the RHT cast taking the Philox state by value (philox_variant.cuh), variant 99, stochastic rounding
#include "ak_rht_common.cuh"
#include "philox_variant.cuh"
namespace ak_rht {
template <bool kSR, int kVar>
void launch_vp(int64_t m, int64_t n, const at::Tensor& x, const at::Tensor& had, const at::Tensor& col_data,
               const at::Tensor& col_si, const at::Tensor& row_data, const at::Tensor& row_si, const at::Tensor& amax_row,
               const at::Tensor& amax_col, at::PhiloxCudaState philox, int sms, cudaStream_t stream, int k_tile) {
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TD = cutlass::float_e2m1_t;
  using TSFD = cutlass::float_ue4m3_t;
  detail::row_col_rht_gemm_ntt_w_sfc_vp<kSR, true, true, true, TA, TB, TD, TSFD, TD, TSFD, false, kVar>(
      static_cast<int>(m), static_cast<int>(n), reinterpret_cast<TA const*>(x.data_ptr()),
      reinterpret_cast<TB const*>(had.data_ptr()), reinterpret_cast<TD*>(col_data.data_ptr()),
      reinterpret_cast<TSFD*>(col_si.data_ptr()), reinterpret_cast<TD*>(row_data.data_ptr()),
      reinterpret_cast<TSFD*>(row_si.data_ptr()), amax_row.data_ptr<float>(), amax_col.data_ptr<float>(), philox,
      static_cast<uint32_t>(sms), stream, k_tile);
}
template void launch_vp<true, 99>(int64_t, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                  const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                  const at::Tensor&, at::PhiloxCudaState, int, cudaStream_t, int);
}  // namespace ak_rht
