// w2d_common.cuh (Llama 3.1 8B kernel campaign, 2026-10-07): the standalone copy of TE's NVFP4 2D weight cast.
// TE's own header (cast/nvfp4/quantize_transpose_nvfp4.cuh: the kernel's constants and every helper) is included as
// is; w2d_kernel.cuh (gen_w2d.py) adds the kernel copy with its schedule parameters; launch_w2d below is TE's host code
// for the 2D path (quantize_transpose<true>, same file) with create_2D_tensor_map (common.cu) copied, for one variant.
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common/cast/nvfp4/quantize_transpose_nvfp4.cuh"
#include "w2d_kernel.cuh"
#include "w2d_reg.cuh"

namespace ak_w2d {
using namespace transformer_engine;

// common.cu: create_2D_tensor_map (not exported by the library; the driver-symbol lookup it uses is)
inline void tensor_map_2d(CUtensorMap &tensorMap, void *dptr, CUtensorMapDataType tensorDataType, const uint64_t globalY,
                          const uint64_t globalX, const uint32_t shmemY, const uint32_t shmemX,
                          const uint32_t stride_elems, const uint32_t offset_elems, const size_t type_num_bits) {
  cuda_driver::ensure_context_exists();
  static PFN_cuTensorMapEncodeTiled_v12000 cuDriverTensorMapEncodeTiled = []() {
    void *driver_ptr = cuda_driver::get_symbol("cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);
  }();
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {globalX, globalY};
  uint64_t stride[rank - 1] = {(stride_elems * type_num_bits) / 8};
  uint32_t boxSize[rank] = {shmemX, shmemY};
  uint32_t elemStride[rank] = {1, 1};
  void *dataPtr = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(dptr) + (offset_elems * type_num_bits) / 8);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(dataPtr) % TMA_GMEM_ALIGNMENT == 0, "tensor data pointer must be 16B aligned");
  const int TMA_needed_size = (TMA_GMEM_ALIGNMENT * 8) / type_num_bits;
  TORCH_CHECK(globalX % TMA_needed_size == 0, "shape not supported");
  const CUresult r = cuDriverTensorMapEncodeTiled(
      &tensorMap, tensorDataType, rank, dataPtr, size, stride, boxSize, elemStride,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE, CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
}

// quantize_transpose<true> (quantize_transpose_nvfp4.cuh) for a bf16 [rows, cols] input, round to nearest, no noop flag,
// rowwise output always, the transpose when col_data is non-empty; amax_col only with the transpose, as TE passes
// output->columnwise_amax.  Variant <CY, PRE, NB, SCS, CREG, BIGT>: see gen_w2d.py.
template <int CY, int PRE, int NB, bool SCS, bool CREG, bool BIGT>
void launch_w2d(const at::Tensor &x, const at::Tensor &row_data, const at::Tensor &row_si, const at::Tensor &col_data,
                const at::Tensor &col_si, const at::Tensor &amax_row, const at::Tensor &amax_col, bool swz,
                cudaStream_t stream) {
  using namespace transformer_engine::dispatch::nvfp4;
  using namespace transformer_engine::dispatch::nvfp4::quantize_transpose_kernel;
  using IType = bf16;
  const bool return_rowwise = row_data.numel() > 0;
  const bool return_transpose = col_data.numel() > 0;
  TORCH_CHECK(return_rowwise, "row-wise output required");
  const size_t rows = static_cast<size_t>(x.size(0)), cols = static_cast<size_t>(x.size(1));
  TORCH_CHECK(rows % 32 == 0 && cols % 32 == 0, "rows and cols must be multiples of 32");
  TORCH_CHECK(!swz || (rows % 128 == 0 && cols % 128 == 0), "swizzled scales need 128-aligned dims");
  TORCH_CHECK(rows % CY == 0, "rows must be a multiple of the variant's rows per CTA");
  const dim3 grid(DIVUP(cols, CHUNK_DIM_X), DIVUP(rows, static_cast<size_t>(CY)));
  const size_t scale_stride = static_cast<size_t>(row_si.size(1));
  const size_t scale_stride_transpose = return_transpose ? static_cast<size_t>(col_si.size(1)) : 0;
  nvfp4_scale_t *const scales_ptr = reinterpret_cast<nvfp4_scale_t *>(row_si.data_ptr());
  nvfp4_scale_t *const scales_transpose_ptr =
      return_transpose ? reinterpret_cast<nvfp4_scale_t *>(col_si.data_ptr()) : nullptr;
  const float *const amax_rowwise_ptr = amax_row.data_ptr<float>();
  const float *const amax_colwise_ptr = return_transpose ? amax_col.data_ptr<float>() : nullptr;

  alignas(64) CUtensorMap tensor_map_input{};
  alignas(64) CUtensorMap tensor_map_output{};
  alignas(64) CUtensorMap tensor_map_output_transpose{};
  tensor_map_2d(tensor_map_input, x.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                cols, 0, sizeof(IType) * 8);
  tensor_map_2d(tensor_map_output, row_data.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rows, cols, BUFF_DIM_Y,
                BUFF_DIM_X, cols, 0, 4);
  if (return_transpose) {
    tensor_map_2d(tensor_map_output_transpose, col_data.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, cols, rows,
                  BUFF_DIM_X, BUFF_DIM_Y, rows, 0, 4);
  }
  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = NB * buff_elems;
  constexpr size_t buff_size_aligned_in = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out = DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_scales = (static_cast<size_t>(CY) * CHUNK_DIM_X) / 16 * sizeof(nvfp4_scale_t);
  // TE's: in + row out + column out + column scales + alignment; BIGT drops the column out buffers when the transpose is
  // written (the kernel then places nothing there), SCS leaves the compact column scales unused
  const bool bigt_on = BIGT && return_transpose && CY == 128;
  const bool scs_on = SCS && swz && (CY == 128 || CY == 64);    // the compact column scales then go unused
  const size_t dshmem_size = buff_size_aligned_in + buff_size_aligned_out + (bigt_on ? 0 : buff_size_aligned_out) +
                             (scs_on ? 0 : buff_size_scales) + TMA_SHMEM_ALIGNMENT;
  constexpr bool CA = false;
  constexpr float (*OP)(float, const Empty &) = nullptr;
  auto go = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
    kernel<<<grid, THREADS_NUM, dshmem_size, stream>>>(
        tensor_map_input, tensor_map_output, tensor_map_output_transpose, scales_ptr, scales_transpose_ptr, nullptr,
        amax_rowwise_ptr, amax_colwise_ptr, rows, cols, scale_stride, scale_stride_transpose, nullptr,
        return_transpose ? reinterpret_cast<uint8_t *>(col_data.data_ptr()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  using namespace transformer_engine::dispatch::nvfp4::ak_w2d;
  if (return_transpose) {
    if (swz) go(quantize_transpose_nvfp4_2D_kernel_v<CA, Empty, OP, IType, false, true, true, true, CY, PRE, NB, SCS, CREG, BIGT>);
    else go(quantize_transpose_nvfp4_2D_kernel_v<CA, Empty, OP, IType, false, true, true, false, CY, PRE, NB, SCS, CREG, BIGT>);
  } else {
    if (swz) go(quantize_transpose_nvfp4_2D_kernel_v<CA, Empty, OP, IType, false, true, false, true, CY, PRE, NB, SCS, CREG, BIGT>);
    else go(quantize_transpose_nvfp4_2D_kernel_v<CA, Empty, OP, IType, false, true, false, false, CY, PRE, NB, SCS, CREG, BIGT>);
  }
}
// the register-tile schedule (w2d_reg.cuh): same inputs as launch_w2d; WR warps along the rows, LD the load flavour
template <int WR, int LD>
void launch_w2r(const at::Tensor &x, const at::Tensor &row_data, const at::Tensor &row_si, const at::Tensor &col_data,
                const at::Tensor &col_si, const at::Tensor &amax_row, const at::Tensor &amax_col, bool swz,
                cudaStream_t stream) {
  using namespace transformer_engine::dispatch::nvfp4;
  const bool return_transpose = col_data.numel() > 0;
  const size_t rows = static_cast<size_t>(x.size(0)), cols = static_cast<size_t>(x.size(1));
  TORCH_CHECK(row_data.numel() > 0, "row-wise output required");
  TORCH_CHECK(rows % (64 * WR) == 0 && cols % 128 == 0, "rows must be a multiple of 64 WR, cols of 128");
  TORCH_CHECK(!swz || rows % 128 == 0, "swizzled scales need 128-aligned dims");
  TORCH_CHECK(row_data.is_contiguous() && row_data.numel() * 2 == static_cast<int64_t>(rows * cols), "row data");
  TORCH_CHECK(!return_transpose || (col_data.is_contiguous() && col_data.numel() * 2 == static_cast<int64_t>(rows * cols)),
              "column data");
  TORCH_CHECK(row_si.is_contiguous() && row_si.dim() == 2 && row_si.size(0) >= static_cast<int64_t>(rows) &&
              row_si.size(1) * 16 >= static_cast<int64_t>(cols), "row-wise scales");
  TORCH_CHECK(!return_transpose || (col_si.is_contiguous() && col_si.dim() == 2 &&
                                    col_si.size(0) >= static_cast<int64_t>(cols) &&
                                    col_si.size(1) * 16 >= static_cast<int64_t>(rows)), "column-wise scales");
  const size_t scale_stride = static_cast<size_t>(row_si.size(1));
  const size_t scale_stride_t = return_transpose ? static_cast<size_t>(col_si.size(1)) : 0;
  TORCH_CHECK(swz || (scale_stride % 8 == 0 && scale_stride_t % (4 * WR) == 0), "compact scale strides");
  const dim3 grid(cols / 128, rows / (64 * WR));
  auto go = [&](auto kernel) {
    kernel<<<grid, 64 * WR, 0, stream>>>(
        reinterpret_cast<const uint4 *>(x.data_ptr()), reinterpret_cast<uint32_t *>(row_data.data_ptr()),
        reinterpret_cast<nvfp4_scale_t *>(row_si.data_ptr()),
        return_transpose ? reinterpret_cast<uint2 *>(col_data.data_ptr()) : nullptr,
        return_transpose ? reinterpret_cast<nvfp4_scale_t *>(col_si.data_ptr()) : nullptr, nullptr,
        amax_row.data_ptr<float>(), return_transpose ? amax_col.data_ptr<float>() : nullptr, rows, cols, scale_stride,
        scale_stride_t);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  using namespace transformer_engine::dispatch::nvfp4::ak_w2d;
  if (return_transpose) {
    if (swz) go(w2d_reg_kernel<WR, true, true, LD>);
    else go(w2d_reg_kernel<WR, false, true, LD>);
  } else {
    if (swz) go(w2d_reg_kernel<WR, true, false, LD>);
    else go(w2d_reg_kernel<WR, false, false, LD>);
  }
}
}  // namespace ak_w2d

#define AK_W2D_INSTANTIATE(CY, PRE, NB, SCS, CREG, BIGT)                                                         \
  template void ak_w2d::launch_w2d<CY, PRE, NB, SCS, CREG, BIGT>(const at::Tensor &, const at::Tensor &,         \
                                                  const at::Tensor &,                                            \
                                                  const at::Tensor &, const at::Tensor &, const at::Tensor &,    \
                                                  const at::Tensor &, bool, cudaStream_t);

#define AK_W2R_INSTANTIATE(WR, LD)                                                                                    \
  template void ak_w2d::launch_w2r<WR, LD>(const at::Tensor &, const at::Tensor &, const at::Tensor &,                \
                                           const at::Tensor &, const at::Tensor &, const at::Tensor &,                \
                                           const at::Tensor &, bool, cudaStream_t);
