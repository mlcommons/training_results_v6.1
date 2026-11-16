// inst8_sr335.cu (rx26): launch_vp8<true, 335> -- stochastic rounding, the Philox state by value (launch_vp8.cuh)
#include "launch_vp8.cuh"
namespace ak_rht8 {
template void launch_vp8<true, 335>(int64_t, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::PhiloxCudaState, int, cudaStream_t, int);
}  // namespace ak_rht8
