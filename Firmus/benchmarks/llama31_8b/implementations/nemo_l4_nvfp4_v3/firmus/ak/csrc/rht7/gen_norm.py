"""gen_norm.py (Llama 3.1 8B campaign, 2026-10-07, NVFP4P version 14): writes norm_variant.cuh -- the RHT cast of
variants.cuh (TE's row_col_rht_gemm_device as rescheduled by gen_variants.py, kernel and launcher) with the RMSNorm
apply folded in front of it: the cast reads the norm's input x (the residual sum) instead of the norm's output z, and
the row-quant warps form z = bf16(gamma * (rs * x)) -- ak_l8b_nvfp4n's apply v6 arithmetic: two IEEE fp32 products,
cvt.rn.bf16x2.f32 -- from each TMA-loaded stage, write it back into the stage for the column-wise RHT MMA, and cast the
row-wise copy from the same registers.  The MMA warp waits for that write (one mbarrier per stage, all 256 row-quant
threads arriving after their fence.proxy.async) besides the TMA's.  Nothing else changes: the same tiles, scales, column
epilogue, schedule and variant bits.

Every edit is an exact text substitution that must match once.
Usage: python3 gen_norm.py <variants.cuh> <norm_variant.cuh>"""
import argparse

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("src")
_ap.add_argument("out")
_args = _ap.parse_args()
src, out = _args.src, _args.out
T = open(src).read()

k0 = T.index("template <class MShape, class NShape, class KShape, class ClusterShape, class ClusterTileShape,")
k1 = T.index("} // NOLINT(readability/fn_size)") + len("} // NOLINT(readability/fn_size)")
l0 = T.index("template <bool kEnableStochasticRounding, bool kEnableRHTColQuant, bool kEnableRowQuant, bool kEnableSwizzleSFOutput,")
l1 = T.rindex("}  // namespace detail")
kern, launch = T[k0:k1], T[l0:l1].rstrip() + "\n"


def sub(text, a, b, n=1):
    c = text.count(a)
    assert c == n, (c, a[:120])
    return text.replace(a, b)


# ---- the kernel ----
kern = sub(kern, "row_col_rht_gemm_device_v(", "row_col_rht_gemm_device_vn(")
kern = sub(kern, "    const size_t* rng_state) {",
           "    const size_t* rng_state,\n    const __nv_bfloat16* norm_gamma,\n    const float* norm_rs) {")
kern = sub(kern, "using SharedStorage = SharedStorage<", "using SharedStorage = SharedStorageN<")
kern = sub(kern,
           "    cute::initialize_barrier(shared_storage.tma_barrier[0], /* num_threads */ 1);\n",
           "    cute::initialize_barrier(shared_storage.tma_barrier[0], /* num_threads */ 1);\n"
           "    // NVFP4P 14: one barrier per stage -- the row-quant threads' write of z into it\n"
           "    CUTLASS_PRAGMA_NO_UNROLL\n"
           "    for (int ak_s = 0; ak_s < K_PIPE_MAX; ++ak_s) {\n"
           "      cute::initialize_barrier(shared_storage.xform_bar[ak_s], NumEpilogueRowQuantThreadCount);\n"
           "    }\n")
kern = sub(kern,
           "          int read_stage = mainloop_pipe_consumer_state.index();\n",
           "          int read_stage = mainloop_pipe_consumer_state.index();\n"
           "          // NVFP4P 14: the stage holds z only once every row-quant thread has written its part back\n"
           "          cute::wait_barrier(shared_storage.xform_bar[read_stage], mainloop_pipe_consumer_state.phase());\n")
kern = sub(kern,
           "      cute::Tensor tQArA = make_tensor_like<TA>(make_layout(tQAsA(_, _, _, _0{}).shape())); // (Copy, Copy_M, Copy_N)\n",
           "      cute::Tensor tQArA = make_tensor_like<TA>(make_layout(tQAsA(_, _, _, _0{}).shape())); // (Copy, Copy_M, Copy_N)\n"
           "      // NVFP4P 14: the same partition for the write of z back into the stage, and this thread's features and\n"
           "      // token within a 128 x 128 tile (the copy's partition of the tile's coordinates): 64 consecutive features\n"
           "      // of one token, checked once\n"
           "      auto ak_tiled_r2s = make_tiled_copy(S2RAtomA{}, S2RThreadLayout{}, S2RValLayout{});\n"
           "      auto ak_thr_r2s = ak_tiled_r2s.get_slice(local_thread_idx);\n"
           "      cute::Tensor ak_tQAsA_w = ak_thr_r2s.partition_D(sA);\n"
           "      cute::Tensor ak_cA = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));\n"
           "      cute::Tensor ak_tQAcA = thr_s2r.partition_S(ak_cA);\n"
           "      static_assert(decltype(size(ak_tQAcA))::value == 64, \"64 values per row-quant thread\");\n"
           "      const int ak_m0 = int(get<0>(ak_tQAcA(0)));\n"
           "      const int ak_k0 = int(get<1>(ak_tQAcA(0)));\n"
           "      CUTLASS_PRAGMA_UNROLL\n"
           "      for (int ak_j = 0; ak_j < 64; ++ak_j) {\n"
           "        if (int(get<0>(ak_tQAcA(ak_j))) != ak_m0 + ak_j || int(get<1>(ak_tQAcA(ak_j))) != ak_k0) { __trap(); }\n"
           "      }\n")
kern = sub(kern,
           "          auto tQApQA_mn = tQApQA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);\n"
           "          auto barrier_token = mainloop_pipeline.consumer_try_wait(\n",
           "          auto tQApQA_mn = tQApQA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);\n"
           "          // NVFP4P 14: this stage's rsigma (this thread's token), loaded before the wait for the stage\n"
           "          const float ak_r = __ldg(norm_rs + int64_t(scheduler.tile_n_base() + uint32_t(k_tile)) * 128 + ak_k0);\n"
           "          auto barrier_token = mainloop_pipeline.consumer_try_wait(\n")
kern = sub(kern,
           "      auto sfa_converter = cutlass::NumericConverter<TSFA, ElementAccumulator>{};\n"
           "      do {\n"
           "        uint32_t skip_wait = K_TILE_MAX <= 0;\n",
           "      auto sfa_converter = cutlass::NumericConverter<TSFA, ElementAccumulator>{};\n"
           "      int ak_tpar = 0;   // NVFP4P 14 (bit 128): the gamma buffer of this work tile\n"
           "      do {\n"
           "        uint32_t skip_wait = K_TILE_MAX <= 0;\n"
           "        if constexpr ((kVar_ & 128) != 0) {   // this work tile's 128 gamma values, fp32, into shared memory\n"
           "          if (local_thread_idx < 128) {\n"
           "            shared_storage.ak_gamma[ak_tpar][local_thread_idx] =\n"
           "                __bfloat162float(norm_gamma[int64_t(scheduler.tile_m()) * 128 + local_thread_idx]);\n"
           "          }\n"
           "          asm volatile(\"bar.sync 14, 256;\" ::: \"memory\");   // the 8 row-quant warps\n"
           "        }\n")
kern = sub(kern,
           "        scheduler.update_work_tile_info();\n"
           "      }while (scheduler.is_valid());\n",
           "        scheduler.update_work_tile_info();\n"
           "        ak_tpar ^= 1;\n"
           "      }while (scheduler.is_valid());\n")
kern = sub(kern,
           "          copy(tiled_s2r, tQAsA(_, _, _, mainloop_pipe_consumer_state.index()), tQArA);\n"
           "          cutlass::arch::fence_view_async_shared();\n",
           "          copy(tiled_s2r, tQAsA(_, _, _, mainloop_pipe_consumer_state.index()), tQArA);\n"
           "          {  // NVFP4P 14: x -> z = bf16(gamma * (rs * x)) in registers, written back for the column-wise MMA\n"
           "            const int64_t ak_f0 = int64_t(scheduler.tile_m()) * 128 + ak_m0;\n"
           "            uint32_t* ak_hv = reinterpret_cast<uint32_t*>(tQArA.data());\n"
           "            if constexpr ((kVar_ & 128) != 0) {   // gamma from this work tile's fp32 copy in shared memory\n"
           "              const float2* ak_gs = reinterpret_cast<const float2*>(&shared_storage.ak_gamma[ak_tpar][ak_m0]);\n"
           "              CUTLASS_PRAGMA_UNROLL\n"
           "              for (int ak_q = 0; ak_q < 32; ++ak_q) {\n"
           "                const float2 ak_g2 = ak_gs[ak_q];\n"
           "                ak_hv[ak_q] = ak_norm_pair_f<kVar_>(ak_hv[ak_q], ak_g2.x, ak_g2.y, ak_r);\n"
           "              }\n"
           "            } else {\n"
           "              const uint4* ak_gp = reinterpret_cast<const uint4*>(norm_gamma + ak_f0);\n"
           "              CUTLASS_PRAGMA_UNROLL\n"
           "              for (int ak_q = 0; ak_q < 8; ++ak_q) {\n"
           "                const uint4 ak_g4 = __ldg(ak_gp + ak_q);\n"
           "                ak_hv[4 * ak_q + 0] = ak_norm_pair<kVar_>(ak_hv[4 * ak_q + 0], ak_g4.x, ak_r);\n"
           "                ak_hv[4 * ak_q + 1] = ak_norm_pair<kVar_>(ak_hv[4 * ak_q + 1], ak_g4.y, ak_r);\n"
           "                ak_hv[4 * ak_q + 2] = ak_norm_pair<kVar_>(ak_hv[4 * ak_q + 2], ak_g4.z, ak_r);\n"
           "                ak_hv[4 * ak_q + 3] = ak_norm_pair<kVar_>(ak_hv[4 * ak_q + 3], ak_g4.w, ak_r);\n"
           "              }\n"
           "            }\n"
           "            copy(ak_tiled_r2s, tQArA, ak_tQAsA_w(_, _, _, mainloop_pipe_consumer_state.index()));\n"
           "          }\n"
           "          cutlass::arch::fence_view_async_shared();\n"
           "          cute::arrive_barrier(shared_storage.xform_bar[mainloop_pipe_consumer_state.index()]);\n")

# ---- the launcher ----
launch = sub(launch, "void row_col_rht_gemm_ntt_w_sfc_v(", "void row_col_rht_gemm_ntt_w_sfc_vn(")
launch = sub(launch, "    int k_tile_size = 1024) {",
             "    int k_tile_size,\n    const __nv_bfloat16* norm_gamma,\n    const float* norm_rs) {")
launch = sub(launch, "AccPipelineBytes; // Reserve for barriers and other uses",
             "AccPipelineBytes + 256 + 1024; // Reserve for barriers and other uses (+ NVFP4P 14's barriers and gamma)")
launch = sub(launch, "  int smem_size = sizeof(\n    SharedStorage<", "  int smem_size = sizeof(\n    SharedStorageN<")
launch = sub(launch, "  auto* kernel_ptr = &row_col_rht_gemm_device_v<", "  auto* kernel_ptr = &row_col_rht_gemm_device_vn<")
launch = sub(launch, "     mma, a_global_amax, d_global_amax, rng_state);",
             "     mma, a_global_amax, d_global_amax, rng_state, norm_gamma, norm_rs);")
launch = sub(launch, "  NVTE_CHECK_CUDA(cudaFuncSetAttribute(*kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));",
             "  NVTE_CHECK(smem_size <= kBlackwellSmemSize, \"NVFP4P 14: shared storage \", smem_size, \" > \", kBlackwellSmemSize);\n"
             "  NVTE_CHECK_CUDA(cudaFuncSetAttribute(*kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));")

head = '''// norm_variant.cuh: generated by gen_norm.py from variants.cuh (see its docstring) -- the RHT cast with the RMSNorm
// apply folded in front of it (NVFP4P version 14).
namespace ak_rht {
namespace detail {

// x -> z for two features of one token: z = bf16(gamma * (rs * x)), ak_l8b_nvfp4n apply v6's arithmetic (y = rs * x,
// t = gamma * y, IEEE fp32 products; cvt.rn.bf16x2.f32); bit 2 of the variant: the products as packed mul.rn.f32x2
template <int kVar>
CUTLASS_DEVICE uint32_t ak_norm_pair(uint32_t xw, uint32_t gw, float r) {
  const float x0 = __uint_as_float(xw << 16), x1 = __uint_as_float(xw & 0xffff0000u);
  const float g0 = __uint_as_float(gw << 16), g1 = __uint_as_float(gw & 0xffff0000u);
  float t0, t1;
  if constexpr ((kVar & 2) != 0) {
    float y0, y1;
    asm("{\\n\\t.reg .b64 ta, tb, td;\\n\\t"
        "mov.b64 ta, {%2, %3};\\n\\t"
        "mov.b64 tb, {%4, %4};\\n\\t"
        "mul.rn.f32x2 td, ta, tb;\\n\\t"
        "mov.b64 {%0, %1}, td;\\n\\t}"
        : "=f"(y0), "=f"(y1) : "f"(x0), "f"(x1), "f"(r));
    asm("{\\n\\t.reg .b64 ta, tb, td;\\n\\t"
        "mov.b64 ta, {%2, %3};\\n\\t"
        "mov.b64 tb, {%4, %5};\\n\\t"
        "mul.rn.f32x2 td, ta, tb;\\n\\t"
        "mov.b64 {%0, %1}, td;\\n\\t}"
        : "=f"(t0), "=f"(t1) : "f"(g0), "f"(g1), "f"(y0), "f"(y1));
  } else {
    t0 = __fmul_rn(g0, __fmul_rn(r, x0));
    t1 = __fmul_rn(g1, __fmul_rn(r, x1));
  }
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(t1), "f"(t0));
  return d;
}

// as ak_norm_pair with gamma given as fp32 (bit 128: gamma staged in shared memory as fp32 per work tile)
template <int kVar>
CUTLASS_DEVICE uint32_t ak_norm_pair_f(uint32_t xw, float g0, float g1, float r) {
  const float x0 = __uint_as_float(xw << 16), x1 = __uint_as_float(xw & 0xffff0000u);
  float t0, t1;
  if constexpr ((kVar & 2) != 0) {
    float y0, y1;
    asm("{\\n\\t.reg .b64 ta, tb, td;\\n\\t"
        "mov.b64 ta, {%2, %3};\\n\\t"
        "mov.b64 tb, {%4, %4};\\n\\t"
        "mul.rn.f32x2 td, ta, tb;\\n\\t"
        "mov.b64 {%0, %1}, td;\\n\\t}"
        : "=f"(y0), "=f"(y1) : "f"(x0), "f"(x1), "f"(r));
    asm("{\\n\\t.reg .b64 ta, tb, td;\\n\\t"
        "mov.b64 ta, {%2, %3};\\n\\t"
        "mov.b64 tb, {%4, %5};\\n\\t"
        "mul.rn.f32x2 td, ta, tb;\\n\\t"
        "mov.b64 {%0, %1}, td;\\n\\t}"
        : "=f"(t0), "=f"(t1) : "f"(g0), "f"(g1), "f"(y0), "f"(y1));
  } else {
    t0 = __fmul_rn(g0, __fmul_rn(r, x0));
    t1 = __fmul_rn(g1, __fmul_rn(r, x1));
  }
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(t1), "f"(t0));
  return d;
}

// SharedStorage plus the per-stage barriers of the write of z
template <
  class ElementA,
  class ElementB,
  class ASmemLayout,
  class BSmemLayout,
  class ClusterShape,
  int AccumulatorPipelineStageCount_,
  int EpilogueUnrollFactor_,
  int SchedulerPipelineStageCount_>
struct SharedStorageN : SharedStorage<ElementA, ElementB, ASmemLayout, BSmemLayout, ClusterShape,
                                      AccumulatorPipelineStageCount_, EpilogueUnrollFactor_, SchedulerPipelineStageCount_> {
  static constexpr int kXformStages = decltype(size<3>(ASmemLayout{}))::value;
  alignas(16) cute::uint64_t xform_bar[kXformStages];
  alignas(16) float ak_gamma[2][128];   // bit 128: the current and the next work tile's gamma, fp32
};

'''
tail = '''
}  // namespace detail
}  // namespace ak_rht
'''
open(out, "w").write(head + kern + "\n\n" + launch + tail)
print("wrote %s: kernel %d lines, launcher %d lines" % (out, kern.count("\n"), launch.count("\n")))
