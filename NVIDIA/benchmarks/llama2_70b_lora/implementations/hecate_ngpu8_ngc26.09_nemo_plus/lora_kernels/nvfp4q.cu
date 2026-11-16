/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// Fast NVFP4 quantize of bf16 activations (the math is in nvfp4q_math.cuh).
#include "nvfp4q_math.cuh"

namespace {

// Quantize kUnroll units in each warp step. A unit is 256 elements of one row. Lane l holds
// elements 8l..8l+7 of the unit, so lanes 2p and 2p+1 hold the 16-element block p.
// The units go from the end of the tensor to the start. The amax pass read the tensor from the
// start, so the end is most likely still in L2.
template <bool SR>
__global__ void __launch_bounds__(kThreads) quantize_kernel(const QuantizeArgs args) {
  __shared__ float shared[kWarps];
  float amax = 0.0f;
  for (int b = threadIdx.x; b < args.n_partial; b += kThreads) amax = fmaxf(amax, args.partial[b]);
  amax = block_max<kThreads>(amax, shared);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (args.amax_out != nullptr) args.amax_out[0] = amax;
    write_gemm_alpha(args.alpha_weight_amax, args.alpha_out, amax);
  }
  ScaleConstants constants;
  const float s_enc = global_encode_scale(amax);
  constants.s_enc_times_fp4_inv = s_enc * kFp4MaxInv;
  constants.s_dec = 1.0f / s_enc;
  constants.num_tiles_x = args.cols >> 6;

  uint64_t seed = 0;
  uint64_t offset = 0;
  if constexpr (SR) unpack_philox(args.philox, seed, offset);

  const uint32_t lane = threadIdx.x & 31;
  const uint32_t units_per_row = args.cols >> 8;
  const uint32_t n_units = args.rows * units_per_row;
  const uint32_t cta_cols = args.cols >> 7;
  const uint32_t warp_first = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * kUnroll;
  const uint32_t step = gridDim.x * kWarps * kUnroll;
  // Lane 0 sends an L2 prefetch for the units of a later warp step.
  const auto prefetch_step = [&](uint32_t ahead) {
    if (lane == 0 && ahead < n_units) {
      const uint32_t low_unit = n_units - ahead - kUnroll;
      prefetch_l2(args.x + low_unit * kUnitVecs, kUnroll * kUnitVecs * sizeof(uint4));
    }
  };
  for (int d = 0; d < args.prefetch; ++d) prefetch_step(warp_first + d * step);

  for (uint32_t first = warp_first; first < n_units; first += step) {
    if (args.prefetch > 0) prefetch_step(first + args.prefetch * step);
    const uint32_t first_unit = n_units - 1 - first;
    uint4 v[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) v[k] = load_stream(args.x + (first_unit - k) * kUnitVecs + lane);
    uint32_t row[kUnroll];
    uint32_t ucol[kUnroll];
    unit_positions(first_unit, units_per_row, row, ucol);
#pragma unroll
    for (int k = 0; k < kUnroll; k += 2) {
      const uint4 pair_v[2] = {v[k], v[k + 1]};
      const uint32_t pair_row[2] = {row[k], row[k + 1]};
      const uint32_t pair_ucol[2] = {ucol[k], ucol[k + 1]};
      float coefficient[2];
      pair_coefficients(pair_v, pair_row, pair_ucol, lane, constants, args.scales, coefficient);
      uint32_t rbits[2][2] = {{0, 0}, {0, 0}};
      if constexpr (SR) {
        pair_random_words(pair_row, pair_ucol, lane, cta_cols, seed, offset, rbits);
      }
      pair_convert<SR>(pair_v, coefficient, rbits, args.data, first_unit - k, lane);
    }
  }
}

// TE extract_seed_and_offset, for tests of the random state.
__global__ void rng_state_kernel(const at::PhiloxCudaState state, int64_t *out) {
  uint64_t seed;
  uint64_t offset;
  unpack_philox(state, seed, offset);
  out[0] = static_cast<int64_t>(seed);
  out[1] = static_cast<int64_t>(offset);
}

// ---------------------------------------------------------------------------------------------
// Host side.
int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

// Amax launch settings: the threads of a block and the L2 prefetch distance in loop steps.
// With -1, the code selects variant 0 for tensors up to two times the L2 size (the quantize
// pass then reads most of the tensor from L2), else variant 2.
struct AmaxVariant {
  int threads;
  int prefetch;
};

constexpr AmaxVariant kAmaxVariants[] = {{1024, 0}, {1024, 2}, {512, 2}};
constexpr int kNumAmaxVariants = sizeof(kAmaxVariants) / sizeof(kAmaxVariants[0]);

int select_amax_variant(const at::Tensor &x, int64_t requested) {
  TORCH_CHECK(requested >= -1 && requested < kNumAmaxVariants, "bad amax_variant");
  if (requested >= 0) return static_cast<int>(requested);
  const int64_t l2_bytes = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  return x.numel() * 2 <= 2 * l2_bytes ? 0 : 2;
}

template <int THREADS, bool FINISH>
int amax_blocks_per_sm() {
  static int blocks = 0;
  if (blocks == 0) {
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, amax_kernel<THREADS, FINISH>, THREADS, 0) == cudaSuccess);
  }
  return blocks;
}

// Grid of one wave: each block stays on its SM until the end of the kernel.
template <bool FINISH>
int amax_grid(int64_t n_vec, int variant) {
  const int threads = kAmaxVariants[variant].threads;
  const int blocks_per_sm = threads == 512 ? amax_blocks_per_sm<512, FINISH>()
                                           : amax_blocks_per_sm<1024, FINISH>();
  const int64_t needed = (n_vec + threads - 1) / threads;
  const int64_t limit = static_cast<int64_t>(blocks_per_sm) * sm_count();
  return static_cast<int>(std::max<int64_t>(1, std::min(needed, limit)));
}

// partial needs amax_grid<FINISH>(numel / 8, variant) floats.
template <bool FINISH>
void launch_amax(const at::Tensor &x, int variant, float *partial, unsigned int *counter,
                 float *out, int n_out) {
  const int64_t n_vec = x.numel() / 8;
  const int grid = amax_grid<FINISH>(n_vec, variant);
  const auto *data = reinterpret_cast<const uint4 *>(x.data_ptr());
  const int prefetch = kAmaxVariants[variant].prefetch;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kAmaxVariants[variant].threads == 512) {
    amax_kernel<512, FINISH><<<grid, 512, 0, stream>>>(data, n_vec, partial, counter, out, n_out,
                                                       prefetch);
  } else {
    amax_kernel<1024, FINISH><<<grid, 1024, 0, stream>>>(data, n_vec, partial, counter, out,
                                                         n_out, prefetch);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool SR>
void launch_quantize(const QuantizeArgs &args) {
  static int blocks_per_sm = 0;
  if (blocks_per_sm == 0) {
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, quantize_kernel<SR>, kThreads, 0) == cudaSuccess);
  }
  const int64_t n_units = static_cast<int64_t>(args.rows) * (args.cols >> 8);
  const int64_t needed = (n_units + kWarps * kUnroll - 1) / (kWarps * kUnroll);
  const int64_t limit = static_cast<int64_t>(blocks_per_sm) * sm_count();
  const int grid = static_cast<int>(std::max<int64_t>(1, std::min(needed, limit)));
  quantize_kernel<SR><<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::PhiloxCudaState next_philox_state() {
  // Same calls as TE init_philox_state(gen, rng_elts_per_thread = 1024).
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->philox_cuda_state(kRngEltsPerThread);
}

void check_input(const at::Tensor &x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, "x must be a CUDA bf16 tensor");
  TORCH_CHECK(x.is_contiguous() && x.dim() >= 2, "x must be contiguous with 2 or more dims");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  TORCH_CHECK(x.numel() < (int64_t{1} << 31), "x must have less than 2^31 elements");
  const int64_t cols = x.size(-1);
  TORCH_CHECK(cols > 0 && cols % 256 == 0, "cols must be a positive multiple of 256");
  const int64_t rows = x.numel() / cols;
  TORCH_CHECK(rows > 0 && rows % 128 == 0, "rows must be a positive multiple of 128");
}

// Quantize x with the amax in partial[0..n_partial). Returns (data, swizzled scales).
std::vector<at::Tensor> quantize_from_partial(const at::Tensor &x, const float *partial,
                                              int n_partial, float *amax_out,
                                              bool stochastic_rounding, int prefetch,
                                              const GemmAlpha &alpha = GemmAlpha{}) {
  const int64_t cols = x.size(-1);
  const int64_t rows = x.numel() / cols;
  auto data_shape = x.sizes().vec();
  data_shape.back() = cols / 2;
  at::Tensor data = at::empty(data_shape, x.options().dtype(at::kByte));
  at::Tensor scales = at::empty({rows, cols / 16}, x.options().dtype(at::kByte));
  QuantizeArgs args;
  args.x = reinterpret_cast<const uint4 *>(x.data_ptr());
  args.data = reinterpret_cast<uint32_t *>(data.data_ptr());
  args.scales = scales.data_ptr<uint8_t>();
  args.partial = partial;
  args.n_partial = n_partial;
  args.amax_out = amax_out;
  args.rows = static_cast<uint32_t>(rows);
  args.cols = static_cast<uint32_t>(cols);
  args.prefetch = prefetch;
  args.alpha_weight_amax = alpha.weight_amax;
  args.alpha_out = alpha.out;
  if (stochastic_rounding) {
    args.philox = next_philox_state();
    launch_quantize<true>(args);
  } else {
    launch_quantize<false>(args);
  }
  return {data, scales};
}

// Amax and quantize: two kernels, no memset. Returns (data, swizzled scales, amax).
std::vector<at::Tensor> quantize(const at::Tensor &x, bool stochastic_rounding,
                                 int64_t amax_variant, int64_t prefetch,
    const std::optional<at::Tensor> &alpha_weight_amax,
    const std::optional<at::Tensor> &alpha_out) {
  check_input(x);
  c10::cuda::CUDAGuard guard(x.device());
  const int variant = select_amax_variant(x, amax_variant);
  const int grid = amax_grid<false>(x.numel() / 8, variant);
  at::Tensor partial = at::empty({grid}, x.options().dtype(at::kFloat));
  at::Tensor amax = at::empty({1}, x.options().dtype(at::kFloat));
  launch_amax<false>(x, variant, partial.data_ptr<float>(), nullptr, nullptr, 0);
  auto result = quantize_from_partial(x, partial.data_ptr<float>(), grid, amax.data_ptr<float>(),
                                      stochastic_rounding, static_cast<int>(prefetch),
                                      gemm_alpha(alpha_weight_amax, alpha_out));
  result.push_back(amax);
  return result;
}

// Quantize with a given global amax (a 1-element float tensor). Returns (data, scales).
std::vector<at::Tensor> quantize_with_amax(const at::Tensor &x, const at::Tensor &amax,
                                           bool stochastic_rounding, int64_t prefetch) {
  check_input(x);
  TORCH_CHECK(amax.is_cuda() && amax.scalar_type() == at::kFloat && amax.numel() >= 1,
              "amax must be a CUDA float tensor");
  c10::cuda::CUDAGuard guard(x.device());
  return quantize_from_partial(x, amax.data_ptr<float>(), 1, nullptr, stochastic_rounding,
                               static_cast<int>(prefetch));
}

// Max |x| into out[0..n_out), with no memset. partial needs amax_partial_size() floats.
// counter is one int32 with the value 0 before the call, and the kernel sets it back to 0.
// Use one partial/counter pair for each stream.
void amax_into(const at::Tensor &x, at::Tensor &out, at::Tensor &partial, at::Tensor &counter,
               int64_t amax_variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "x must be a contiguous CUDA bf16 tensor");
  TORCH_CHECK(x.numel() % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x must have a multiple of 8 elements and be 16-byte aligned");
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.numel() >= 1 && out.numel() <= 2,
              "out must have 1 or 2 floats");
  TORCH_CHECK(counter.scalar_type() == at::kInt && counter.numel() >= 1, "counter must be int32");
  c10::cuda::CUDAGuard guard(x.device());
  const int variant = select_amax_variant(x, amax_variant);
  TORCH_CHECK(partial.scalar_type() == at::kFloat &&
                  partial.numel() >= amax_grid<true>(x.numel() / 8, variant),
              "partial is too small");
  launch_amax<true>(x, variant, partial.data_ptr<float>(),
                    reinterpret_cast<unsigned int *>(counter.data_ptr<int>()),
                    out.data_ptr<float>(), static_cast<int>(out.numel()));
}

// Floats that amax_into needs in partial: one wave of the largest variant grid.
int64_t amax_partial_size() {
  const int64_t many = int64_t{1} << 40;
  int64_t size = 0;
  for (int variant = 0; variant < kNumAmaxVariants; ++variant) {
    size = std::max<int64_t>(size, amax_grid<true>(many, variant));
  }
  return size;
}

// Only the amax partial kernel of one variant. For the tuning sweep.
at::Tensor amax_partials(const at::Tensor &x, int64_t amax_variant) {
  check_input(x);
  c10::cuda::CUDAGuard guard(x.device());
  const int variant = select_amax_variant(x, amax_variant);
  at::Tensor partial = at::empty({amax_grid<false>(x.numel() / 8, variant)},
                                 x.options().dtype(at::kFloat));
  launch_amax<false>(x, variant, partial.data_ptr<float>(), nullptr, nullptr, 0);
  return partial;
}

// The int64 [seed, offset] that TE quantize makes for stochastic rounding, with the same
// generator advance. For tests.
at::Tensor rng_state() {
  at::PhiloxCudaState state = next_philox_state();
  at::Tensor out = at::empty({2}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  rng_state_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(state, out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Quantize x with the per-block amax partials of a producer kernel (amax_prod). The global amax
// is the max of the partials, as in quantize(). Returns (data, swizzled scales, amax).
std::vector<at::Tensor> quantize_from_partials(const at::Tensor &x, const at::Tensor &partials,
                                               bool stochastic_rounding, int64_t prefetch,
    const std::optional<at::Tensor> &alpha_weight_amax,
    const std::optional<at::Tensor> &alpha_out) {
  check_input(x);
  TORCH_CHECK(partials.is_cuda() && partials.scalar_type() == at::kFloat &&
                  partials.is_contiguous() && partials.numel() >= 1 &&
                  partials.numel() < (int64_t{1} << 30),
              "partials must be a contiguous CUDA float tensor");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor amax = at::empty({1}, x.options().dtype(at::kFloat));
  auto result = quantize_from_partial(x, partials.data_ptr<float>(),
                                      static_cast<int>(partials.numel()), amax.data_ptr<float>(),
                                      stochastic_rounding, static_cast<int>(prefetch),
                                      gemm_alpha(alpha_weight_amax, alpha_out));
  result.push_back(amax);
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize, "Amax and NVFP4 quantize with swizzled scales", py::arg("x"),
        py::arg("stochastic_rounding"), py::arg("amax_variant") = -1, py::arg("prefetch") = 1,
        py::arg("alpha_weight_amax") = py::none(), py::arg("alpha_out") = py::none());
  m.def("quantize_with_amax", &quantize_with_amax, "NVFP4 quantize with a given amax",
        py::arg("x"), py::arg("amax"), py::arg("stochastic_rounding"), py::arg("prefetch") = 1);
  m.def("amax_into", &amax_into, "Max |x| with no memset", py::arg("x"), py::arg("out"),
        py::arg("partial"), py::arg("counter"), py::arg("amax_variant") = -1);
  m.def("amax_partial_size", &amax_partial_size, "Floats that amax_into needs in partial");
  m.def("amax_partials", &amax_partials, "Amax partial kernel only (tuning)", py::arg("x"),
        py::arg("amax_variant"));
  m.def("rng_state", &rng_state, "TE-equivalent stochastic-rounding RNG state");
  m.def("quantize_from_partials", &quantize_from_partials,
        "NVFP4 quantize with the amax partials of a producer kernel", py::arg("x"),
        py::arg("partials"), py::arg("stochastic_rounding"), py::arg("prefetch") = 1,
        py::arg("alpha_weight_amax") = py::none(), py::arg("alpha_out") = py::none());
}
