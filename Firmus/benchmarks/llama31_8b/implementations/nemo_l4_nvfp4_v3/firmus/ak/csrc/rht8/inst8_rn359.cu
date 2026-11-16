// inst8_rn359.cu (rx26): launch_v8<false, 359> -- round to nearest (ak_rht8_common.cuh / variants8.cuh)
#include "ak_rht8_common.cuh"
namespace ak_rht8 {
template void launch_v8<false, 359>(int64_t, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const size_t*, int, cudaStream_t, int);
}  // namespace ak_rht8
