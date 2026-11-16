// ak_l8b_lmh (Llama 3.1 8B campaign, 2026-10-08, helper line lmh29): the LM head's BF16 forward and dgrad GEMMs through
// cuBLASLt with a pinned algorithm.
//
// The recipe's LM head (Megatron's output layer, no logit dtype) computes its forward with torch.matmul(x, W^T) and its
// dgrad with dy.matmul(W); both run cuBLAS's own pick.  ge28 (EXPERIMENT_LOG sec L16.5) found that, with TE's bf16
// cuBLASLt descriptors for the same problems, the heuristic's entry #2 is byte-equal to torch's output and spends less
// energy under the power cap (forward nvjet_sm103_tst_256x256_64x4_2x2f_2cta_h_bz_TNT -3.75 % J, dgrad
// nvjet_sm103_tst_128x512_64x4_2x1_2cta_v_bz_NNT -1.25 % J).  This module builds exactly TE 131157b5's bf16 descriptors
// (transformer_engine/common/gemm/cublaslt_gemm.cu: compute 32F with an FP32 scale type, host alpha 1 / beta 0, C = D,
// the default epilogue; the preference's maximum workspace after TE's 256-byte alignment, the reduction-scheme mask
// COMPUTE_TYPE and the four pointer alignments) and
//   heuristics(...) -> the heuristic's list for the problem, each algorithm as its 64 bytes (cublasLtMatmulAlgo_t);
//   gemm(...)       -> one cublasLtMatmul with the algorithm given as those 64 bytes (cublasLtMatmulAlgoCheck first).
// The algorithm is opaque (every config attribute reads -1 for these kernels), so the caller pins it by its bytes.
//   kind 0, forward (TE layout "TN"): out[T, V] = x[T, H] @ W[V, H]^T   A = W (T), B = x (N);  m = V, n = T, k = H
//   kind 1, dgrad   (TE layout "NN"): out[T, H] = dy[T, V] @ W[V, H]    A = W (N), B = dy (N); m = H, n = T, k = V
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define LMH_CHECK(x)                                                                              \
  do {                                                                                            \
    cublasStatus_t st_ = (x);                                                                     \
    TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "ak_l8b_lmh: ", #x, " failed with status ", (int)st_); \
  } while (0)

namespace {

cublasLtHandle_t handle() {
  static cublasLtHandle_t h = nullptr;
  static std::once_flag once;
  std::call_once(once, [] { LMH_CHECK(cublasLtCreate(&h)); });
  return h;
}

uint32_t alignment_of(const void* p) {  // TE's _getAlignment: the largest power of two <= 256 dividing the address
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  uint32_t al = 256;
  while (al > 1 && a % al) al /= 2;
  return al;
}

struct Problem {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t A = nullptr, B = nullptr, C = nullptr, D = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  void* ws = nullptr;
  size_t ws_size = 0;
  ~Problem() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (D) cublasLtMatrixLayoutDestroy(D);
    if (C) cublasLtMatrixLayoutDestroy(C);
    if (B) cublasLtMatrixLayoutDestroy(B);
    if (A) cublasLtMatrixLayoutDestroy(A);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

void check_args(int kind, const at::Tensor& W, const at::Tensor& X, const at::Tensor& out, const at::Tensor& ws) {
  TORCH_CHECK(kind == 0 || kind == 1, "ak_l8b_lmh: kind is 0 (forward) or 1 (dgrad)");
  for (const at::Tensor* t : {&W, &X, &out}) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->dim() == 2 && t->is_contiguous(),
                "ak_l8b_lmh: W, X and out must be contiguous 2-D bf16 CUDA tensors");
  }
  TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kByte && ws.is_contiguous(), "ak_l8b_lmh: workspace: uint8 CUDA");
  const int64_t V = W.size(0), H = W.size(1), T = X.size(0);
  if (kind == 0) {
    TORCH_CHECK(X.size(1) == H && out.size(0) == T && out.size(1) == V, "ak_l8b_lmh: forward shapes");
  } else {
    TORCH_CHECK(X.size(1) == V && out.size(0) == T && out.size(1) == H, "ak_l8b_lmh: dgrad shapes");
  }
}

void build(Problem& p, int kind, const at::Tensor& W, const at::Tensor& X, const at::Tensor& out, const at::Tensor& ws) {
  check_args(kind, W, X, out, ws);
  const int64_t V = W.size(0), H = W.size(1), T = X.size(0);
  const cublasOperation_t ta = kind == 0 ? CUBLAS_OP_T : CUBLAS_OP_N, tb = CUBLAS_OP_N;
  const int64_t m = kind == 0 ? V : H, n = T, k = kind == 0 ? H : V;
  const int64_t lda = H;                  // W, row-major [V, H] = column-major [H, V]
  const int64_t ldb = kind == 0 ? H : V;  // x [T, H] -> column-major [H, T]; dy [T, V] -> column-major [V, T]
  const int64_t ldd = m;                  // out [T, m] row-major = column-major [m, T]
  LMH_CHECK(cublasLtMatrixLayoutCreate(&p.A, CUDA_R_16BF, ta == CUBLAS_OP_N ? m : k, ta == CUBLAS_OP_N ? k : m, lda));
  LMH_CHECK(cublasLtMatrixLayoutCreate(&p.B, CUDA_R_16BF, tb == CUBLAS_OP_N ? k : n, tb == CUBLAS_OP_N ? n : k, ldb));
  LMH_CHECK(cublasLtMatrixLayoutCreate(&p.D, CUDA_R_16BF, m, n, ldd));
  LMH_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  LMH_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LMH_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  LMH_CHECK(cublasLtMatrixLayoutCreate(&p.C, CUDA_R_16BF, m, n, ldd));
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  LMH_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  // the workspace, aligned to 256 B as TE aligns it
  const uint32_t orig = alignment_of(ws.data_ptr());
  p.ws = static_cast<uint8_t*>(ws.data_ptr()) + 256 - orig;
  TORCH_CHECK(static_cast<size_t>(ws.numel()) > 256, "ak_l8b_lmh: workspace too small");
  p.ws_size = static_cast<size_t>(ws.numel()) - 256 + orig;
  LMH_CHECK(cublasLtMatmulPreferenceCreate(&p.pref));
  LMH_CHECK(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p.ws_size,
                                                 sizeof(p.ws_size)));
  const uint32_t mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  LMH_CHECK(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));
  const uint32_t aa = alignment_of(W.data_ptr()), ab = alignment_of(X.data_ptr()), ac = alignment_of(out.data_ptr());
  LMH_CHECK(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aa, sizeof(aa)));
  LMH_CHECK(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &ab, sizeof(ab)));
  LMH_CHECK(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &ac, sizeof(ac)));
  LMH_CHECK(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &ac, sizeof(ac)));
}

}  // namespace

// The heuristic's list for the problem (as TE would query it, but up to n_req entries): one 64-byte string per entry,
// in the heuristic's order, plus each entry's workspace size and waves count in a parallel list.
std::vector<py::object> heuristics(int64_t kind, at::Tensor W, at::Tensor X, at::Tensor out, at::Tensor ws, int64_t n_req) {
  c10::cuda::CUDAGuard guard(W.device());
  Problem p;
  build(p, static_cast<int>(kind), W, X, out, ws);
  std::vector<cublasLtMatmulHeuristicResult_t> res(static_cast<size_t>(n_req));
  int got = 0;
  LMH_CHECK(cublasLtMatmulAlgoGetHeuristic(handle(), p.op, p.A, p.B, p.C, p.D, p.pref, static_cast<int>(n_req),
                                           res.data(), &got));
  std::vector<py::object> outl;
  for (int i = 0; i < got; ++i) {
    std::string blob(reinterpret_cast<const char*>(&res[i].algo), sizeof(cublasLtMatmulAlgo_t));
    py::tuple t = py::make_tuple(py::bytes(blob), static_cast<int64_t>(res[i].workspaceSize), res[i].wavesCount,
                                 static_cast<int64_t>(res[i].state));
    outl.push_back(t);
  }
  return outl;
}

// out = the GEMM with the given algorithm (its 64 bytes); the current stream; CUDA-graph capturable.
void gemm(int64_t kind, at::Tensor W, at::Tensor X, at::Tensor out, at::Tensor ws, py::bytes algo_bytes) {
  c10::cuda::CUDAGuard guard(W.device());
  Problem p;
  build(p, static_cast<int>(kind), W, X, out, ws);
  const std::string blob = algo_bytes;
  TORCH_CHECK(blob.size() == sizeof(cublasLtMatmulAlgo_t), "ak_l8b_lmh: an algorithm is ", sizeof(cublasLtMatmulAlgo_t),
              " bytes, got ", blob.size());
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo, blob.data(), sizeof(algo));
  cublasLtMatmulHeuristicResult_t chk = {};
  LMH_CHECK(cublasLtMatmulAlgoCheck(handle(), p.op, p.A, p.B, p.C, p.D, &algo, &chk));
  TORCH_CHECK(chk.workspaceSize <= p.ws_size, "ak_l8b_lmh: the algorithm needs ", chk.workspaceSize, " B of workspace");
  const float alpha = 1.0f, beta = 0.0f;
  LMH_CHECK(cublasLtMatmul(handle(), p.op, &alpha, W.data_ptr(), p.A, X.data_ptr(), p.B, &beta, out.data_ptr(), p.C,
                           out.data_ptr(), p.D, &algo, p.ws, p.ws_size, at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("heuristics", &heuristics, "the heuristic's algorithms (64-byte strings) for the LM-head problem");
  m.def("gemm", &gemm, "the LM-head GEMM with the given algorithm");
  m.attr("ALGO_BYTES") = static_cast<int64_t>(sizeof(cublasLtMatmulAlgo_t));
}
