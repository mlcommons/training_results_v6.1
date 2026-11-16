// launch_vp8.cuh (rx26): the launcher the SR instantiation files instantiate -- the rx26 cast with the generator's
// Philox state by value (philox_variant8.cuh), as p14dev's inst_p79.cu's launch_vp
#pragma once
#include "ak_rht8_common.cuh"
#include "philox_variant8.cuh"
namespace ak_rht8 {
template <bool kSR, int kVar>
void launch_vp8(int64_t m, int64_t n, const at::Tensor& x, const at::Tensor& had, const at::Tensor& col_data,
                const at::Tensor& col_si, const at::Tensor& row_data, const at::Tensor& row_si, const at::Tensor& amax_row,
                const at::Tensor& amax_col, at::PhiloxCudaState philox, int sms, cudaStream_t stream, int k_tile) {
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TD = cutlass::float_e2m1_t;
  using TSFD = cutlass::float_ue4m3_t;
  detail::row_col_rht_gemm_ntt_w_sfc_vp8<kSR, true, true, true, TA, TB, TD, TSFD, TD, TSFD, false, kVar>(
      static_cast<int>(m), static_cast<int>(n), reinterpret_cast<TA const*>(x.data_ptr()),
      reinterpret_cast<TB const*>(had.data_ptr()), reinterpret_cast<TD*>(col_data.data_ptr()),
      reinterpret_cast<TSFD*>(col_si.data_ptr()), reinterpret_cast<TD*>(row_data.data_ptr()),
      reinterpret_cast<TSFD*>(row_si.data_ptr()), amax_row.data_ptr<float>(), amax_col.data_ptr<float>(), philox,
      static_cast<uint32_t>(sms), stream, k_tile);
}
}  // namespace ak_rht8
