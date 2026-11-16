// inst_n70.cu: the RHT cast with the RMSNorm apply in front (norm_variant.cuh), variant 70 (packed f32x2 products, the
// column copy's bf16 round trip in one conversion; round to nearest only -- the norm outputs are activations) and 198
// (70 with gamma staged per work tile in shared memory as fp32)
#include "ak_rht_common.cuh"
#include "norm_variant.cuh"
namespace ak_rht {
template <int kVar>
void launch_vn(int64_t m, int64_t n, const at::Tensor& x, const at::Tensor& had, const at::Tensor& col_data,
               const at::Tensor& col_si, const at::Tensor& row_data, const at::Tensor& row_si, const at::Tensor& amax_row,
               const at::Tensor& amax_col, const at::Tensor& gamma, const at::Tensor& rs, int sms, cudaStream_t stream,
               int k_tile) {
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TD = cutlass::float_e2m1_t;
  using TSFD = cutlass::float_ue4m3_t;
  detail::row_col_rht_gemm_ntt_w_sfc_vn<false, true, true, true, TA, TB, TD, TSFD, TD, TSFD, false, kVar>(
      static_cast<int>(m), static_cast<int>(n), reinterpret_cast<TA const*>(x.data_ptr()),
      reinterpret_cast<TB const*>(had.data_ptr()), reinterpret_cast<TD*>(col_data.data_ptr()),
      reinterpret_cast<TSFD*>(col_si.data_ptr()), reinterpret_cast<TD*>(row_data.data_ptr()),
      reinterpret_cast<TSFD*>(row_si.data_ptr()), amax_row.data_ptr<float>(), amax_col.data_ptr<float>(), nullptr,
      static_cast<uint32_t>(sms), stream, k_tile, reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
      rs.data_ptr<float>());
}
template void launch_vn<70>(int64_t, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, int, cudaStream_t, int);
template void launch_vn<198>(int64_t, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, int, cudaStream_t, int);
}  // namespace ak_rht
