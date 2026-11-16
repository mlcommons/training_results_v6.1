# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import weakref
from typing import Optional

import torch

from . import arch as _arch

_ENABLED = os.environ.get("OPT_NVFP4_PRODUCER_QUANT", "1") == "1"
_IDENT = os.environ.get("OPT_NVFP4_PQ_IDENT", "0") == "1"
_SWIGLU = os.environ.get("OPT_NVFP4_PQ_SWIGLU", "0") == "1"
_DIR = os.path.dirname(os.path.abspath(__file__))
_ext = None

IDENT, RMSNORM, SWIGLU, DSWIGLU = 0, 1, 2, 3


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = _arch.cuda_arch_list()
        try:
            _ext = load(
                name=_arch.ext_name("opt_nvfp4_producer_quant"),
                sources=[
                    os.path.join(_DIR, "nvfp4_producer_quant.cu"),
                    os.path.join(_DIR, "nvfp4_producer_quant_binding.cpp"),
                ],
                extra_include_paths=[_DIR],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
                build_directory=os.environ.get("OPT_EXT_BUILD_DIR") or None,
                verbose=False,
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _ext


@torch.library.custom_op("optimized::nvfp4_quant_stats", mutates_args=("partials", "sched"))
def nvfp4_quant_stats(
    producer: int,
    M: int,
    N: int,
    in0: torch.Tensor,
    in1: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor],
    eps: float,
    partials: torch.Tensor,
    rht: torch.Tensor,
    sched: torch.Tensor,
    do_col: bool,
) -> None:
    _load().stats(producer, M, N, in0, in1, gamma, eps, partials, rht, sched, do_col)


@torch.library.custom_op(
    "optimized::nvfp4_quant_cast",
    mutates_args=("sched", "row_data", "row_scale", "row_amax", "col_data", "col_scale", "col_amax"),
)
def nvfp4_quant_cast(
    producer: int,
    M: int,
    N: int,
    in0: torch.Tensor,
    in1: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor],
    partials: torch.Tensor,
    rht: torch.Tensor,
    sched: torch.Tensor,
    row_data: Optional[torch.Tensor],
    row_scale: Optional[torch.Tensor],
    row_amax: Optional[torch.Tensor],
    col_data: Optional[torch.Tensor],
    col_scale: Optional[torch.Tensor],
    col_amax: Optional[torch.Tensor],
    swizzled: bool,
    stochastic: bool,
) -> None:
    _load().cast(
        producer, M, N, in0, in1, gamma, partials, rht, sched, row_data, row_scale, row_amax,
        col_data, col_scale, col_amax, swizzled, stochastic,
    )


@torch.library.custom_op(
    "optimized::nvfp4_rmsnorm_quant",
    mutates_args=("rstd", "state_f32", "state_i32", "row_data", "row_scale", "row_amax",
                  "col_data", "col_scale", "col_amax"),
)
def nvfp4_rmsnorm_quant(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    rstd: Optional[torch.Tensor],
    rht: torch.Tensor,
    state_f32: torch.Tensor,
    state_i32: torch.Tensor,
    factor: float,
    cast: bool,
    row_data: Optional[torch.Tensor],
    row_scale: Optional[torch.Tensor],
    row_amax: Optional[torch.Tensor],
    col_data: Optional[torch.Tensor],
    col_scale: Optional[torch.Tensor],
    col_amax: Optional[torch.Tensor],
    swizzled: bool,
) -> None:
    _load().rmsnorm_quant(x, gamma, eps, rstd, rht, state_f32, state_i32, factor, cast, row_data,
                          row_scale, row_amax, col_data, col_scale, col_amax, swizzled)


class _NormState:

    __slots__ = ("f32", "i32", "gamma_ref")

    def __init__(self, device, gamma):
        self.f32 = torch.zeros(4, dtype=torch.float32, device=device)
        self.i32 = torch.zeros(2, dtype=torch.int32, device=device)
        self.gamma_ref = weakref.ref(_root(gamma))


def _root(t):
    return t if t._base is None else t._base


_norm_states = {}


def rmsnorm_quant(quantizer, x, gamma, eps, site=None):
    N = x.shape[-1]
    M = x.numel() // N
    key = (gamma.data_ptr(), x.device.index) if site is None else site
    st = _norm_states.get(key)
    if st is not None and st.gamma_ref() is None:
        st = None
    if st is None:
        for k in [k for k, v in _norm_states.items() if v.gamma_ref() is None]:
            del _norm_states[k]
        st = _NormState(x.device, gamma)
        _norm_states[key] = st
    out = quantizer.make_empty(list(x.shape), dtype=torch.bfloat16, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    rht = quantizer.rht_matrix
    torch.ops.optimized.nvfp4_rmsnorm_quant(
        x, gamma, float(eps), None, rht, st.f32, st.i32, 1.0, False,
        None, None, None, None, None, None, True)
    torch.ops.optimized.nvfp4_rmsnorm_quant(
        x, gamma, float(eps), rstd, rht, st.f32, st.i32, 1.0, True,
        out._rowwise_data, out._rowwise_scale_inv, out._amax_rowwise,
        out._columnwise_data, out._columnwise_scale_inv, out._amax_columnwise,
        bool(out._with_gemm_swizzled_scales))
    return out, rstd


_sched = {}


def _sched_state(device):
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    st = _sched.get(key)
    if st is None:
        st = torch.zeros(4, dtype=torch.int32, device=device)
        _sched[key] = st
    return st


def produce(producer, quantizer, out_shape, M, N, in0, in1=None, gamma=None, eps=0.0):
    assert producer != RMSNORM
    ext = _load()
    out = quantizer.make_empty(out_shape, dtype=torch.bfloat16, device=in0.device)
    P = ext.partials_needed(producer, M, N)
    partials = torch.empty(2 * P, dtype=torch.float32, device=in0.device)
    do_col = out._columnwise_data is not None
    rht = quantizer.rht_matrix
    sched = _sched_state(in0.device)
    torch.ops.optimized.nvfp4_quant_stats(
        producer, M, N, in0, in1, gamma, float(eps), partials, rht, sched, do_col
    )
    torch.ops.optimized.nvfp4_quant_cast(
        producer, M, N, in0, in1, gamma, partials, rht, sched,
        out._rowwise_data, out._rowwise_scale_inv, out._amax_rowwise,
        out._columnwise_data, out._columnwise_scale_inv, out._amax_columnwise,
        bool(out._with_gemm_swizzled_scales), bool(quantizer.stochastic_rounding),
    )
    return out


_orig = {}
_counts = {"ident": 0, "rmsnorm": 0, "swiglu": 0, "dswiglu": 0}
_announced = False


def _announce():
    global _announced
    if not _announced:
        _announced = True
        if int(os.environ.get("RANK", "0")) == 0:
            print("[opt nvfp4_producer_quant] NVFP4 quantization fused into producers active"
                  " (norm outputs: single pass, current scaling)", flush=True)


def _eligible_quantizer(q) -> bool:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if type(q) is not NVFP4Quantizer:
        return False
    if not (q.with_rht and q.with_post_rht_amax):
        return False
    if q.with_amax_reduction or q.row_scaled_nvfp4 or q.nvfp4_use_4over6 or q.with_2d_quantization:
        return False
    if q.nvfp4_e4m3_max != 448 or not (q.rowwise_usage or q.columnwise_usage):
        return False
    if os.environ.get("NVTE_USE_FAST_MATH", "0") not in ("", "0"):
        return False
    rht = getattr(q, "rht_matrix", None)
    return rht is not None and rht.dtype == torch.bfloat16 and rht.numel() == 256


def _plain_bf16(t, last_mod) -> bool:
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

    return (
        isinstance(t, torch.Tensor)
        and not isinstance(t, QuantizedTensorStorage)
        and t.is_cuda
        and t.dtype == torch.bfloat16
        and t.dim() >= 2
        and t.is_contiguous()
        and t.shape[-1] % last_mod == 0
        and (t.numel() // t.shape[-1]) % 128 == 0
    )


_wrapped = {}


def _build():
    if _wrapped:
        return
    import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    _load()
    _orig.update(rmsnorm_fwd=tex.rmsnorm_fwd, swiglu=tex.swiglu, dswiglu=tex.dswiglu,
                 quantize_impl=NVFP4Quantizer.quantize_impl)
    orig_rmsnorm_fwd = _orig["rmsnorm_fwd"]
    orig_swiglu = _orig["swiglu"]
    orig_dswiglu = _orig["dswiglu"]
    orig_quantize_impl = _orig["quantize_impl"]

    def rmsnorm_fwd(x, w, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma):
        if (
            ln_out is None
            and not zero_centered_gamma
            and _eligible_quantizer(quantizer)
            and not quantizer.stochastic_rounding
            and quantizer.rowwise_usage
            and quantizer.columnwise_usage
            and _plain_bf16(x, 128)
            and x.shape[-1] == 4096
            and isinstance(w, torch.Tensor)
            and w.dtype == torch.bfloat16
            and w.is_contiguous()
            and w.numel() == 4096
            and int(otype) == int(tex.DType.kBFloat16)
        ):
            _announce()
            _counts["rmsnorm"] += 1
            out, rstd = rmsnorm_quant(quantizer, x, w, eps)
            return [out, None, rstd]
        return orig_rmsnorm_fwd(x, w, eps, ln_out, quantizer, otype, sm_margin,
                                zero_centered_gamma)

    def swiglu(inp, quantizer, *args, **kwargs):
        if (
            _SWIGLU
            and not args
            and not kwargs
            and _eligible_quantizer(quantizer)
            and _plain_bf16(inp, 256)
        ):
            _announce()
            _counts["swiglu"] += 1
            F = inp.shape[-1] // 2
            M = inp.numel() // inp.shape[-1]
            out = produce(SWIGLU, quantizer, list(inp.shape[:-1]) + [F], M, F, inp)
            return out
        return orig_swiglu(inp, quantizer, *args, **kwargs)

    def dswiglu(grad, inp, quantizer, *args, **kwargs):
        if (
            not args
            and not kwargs
            and _eligible_quantizer(quantizer)
            and _plain_bf16(inp, 256)
            and _plain_bf16(grad, 128)
            and grad.numel() * 2 == inp.numel()
            and grad.shape[-1] * 2 == inp.shape[-1]
        ):
            _announce()
            _counts["dswiglu"] += 1
            N = inp.shape[-1]
            M = inp.numel() // N
            out = produce(DSWIGLU, quantizer, list(inp.shape), M, N, grad, in1=inp)
            return out
        return orig_dswiglu(grad, inp, quantizer, *args, **kwargs)

    def quantize_impl(self, tensor):
        if _IDENT and _eligible_quantizer(self) and _plain_bf16(tensor, 128):
            _announce()
            _counts["ident"] += 1
            N = tensor.shape[-1]
            M = tensor.numel() // N
            out = produce(IDENT, self, list(tensor.shape), M, N, tensor)
            return out
        return orig_quantize_impl(self, tensor)

    _wrapped.update(rmsnorm_fwd=rmsnorm_fwd, swiglu=swiglu, dswiglu=dswiglu,
                    quantize_impl=quantize_impl)


def set_active(active: bool):
    import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
    import transformer_engine_torch as tex
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.ops.basic.rmsnorm as te_rmsnorm_op
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    _build()
    src = _wrapped if active else _orig
    for mod in (tex, te_cpp):
        mod.rmsnorm_fwd = src["rmsnorm_fwd"]
        mod.swiglu = src["swiglu"]
        mod.dswiglu = src["dswiglu"]
    te_rmsnorm_op.rmsnorm_fwd = src["rmsnorm_fwd"]
    NVFP4Quantizer.quantize_impl = src["quantize_impl"]


def install():
    if _ENABLED:
        _arch.check_te_version("nvfp4_producer_quant (TE's NVFP4 cast arithmetic and Philox draws mirrored by hand)",
                               "OPT_NVFP4_PRODUCER_QUANT=0")
        set_active(True)


install()


def _make_quantizer(sr: bool, swizzled: bool = True):
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    q = NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True,
                       stochastic_rounding=sr)
    q.internal = True
    q.optimize_for_gemm = swizzled
    return q


def _cmp(name, a, b) -> str:
    if a is None or b is None:
        return f"{name}={'both-none' if a is b else 'ONE-NONE'}"
    if a.shape != b.shape:
        return f"{name}=SHAPE{tuple(a.shape)}vs{tuple(b.shape)}"
    nbad = int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item())
    return f"{name}={'ok' if nbad == 0 else 'BAD:' + str(nbad)}"


def _compare_out(out, ref) -> list:
    return [_cmp(n, getattr(out, "_" + n), getattr(ref, "_" + n))
            for n in ("rowwise_data", "rowwise_scale_inv", "columnwise_data", "columnwise_scale_inv",
                      "amax_rowwise", "amax_columnwise")]


def _rmsnorm_check(m, swizzled, dev) -> bool:
    import transformer_engine_torch as tex

    n, eps = 4096, 1e-5
    g = torch.Generator(device=dev).manual_seed(m * 17 + 3)
    x = (torch.randn(m, n, device=dev, dtype=torch.float32, generator=g)
         * torch.exp(torch.randn(m, 1, device=dev, generator=g))).to(torch.bfloat16)
    gamma = (1.0 + 0.1 * torch.randn(n, device=dev, generator=g)).to(torch.bfloat16)
    x32 = x.float()
    exact = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()
    q = _make_quantizer(False, swizzled)
    ref_out, _, ref_rstd = _orig["rmsnorm_fwd"](x, gamma, eps, None, q, tex.DType.kBFloat16, 0, False)
    q = _make_quantizer(False, swizzled)
    out, rstd = rmsnorm_quant(q, x, gamma, eps, site=("selftest", m))
    torch.cuda.synchronize()
    err = lambda t: float(((t.dequantize(dtype=torch.float32) - exact).norm() / exact.norm()).item())
    e_ours, e_te = err(out), err(ref_out)
    frac = lambda a, b: float((a.view(torch.uint8) != b.view(torch.uint8)).float().mean().item())
    f_row, f_col = frac(out._rowwise_data, ref_out._rowwise_data), frac(out._columnwise_data, ref_out._columnwise_data)
    rstd_rel = float(((rstd - ref_rstd).abs() / ref_rstd.abs()).max().item())
    a_rel = max(float(((a - b).abs() / b.abs()).max().item()) for a, b in
                ((out._amax_rowwise, ref_out._amax_rowwise), (out._amax_columnwise, ref_out._amax_columnwise)))
    ok = (abs(e_ours - e_te) <= 2e-4 and f_row <= 0.01 and f_col <= 0.03 and rstd_rel <= 1e-6
          and a_rel <= 2.0 ** -8)
    print(f"SELFTEST rmsnorm swizzled={swizzled} shape=({m},{n}) {'OK' if ok else 'MISMATCH'} "
          f"decoded_rel_fro ours={e_ours:.5e} te={e_te:.5e} exact-ref | differing payload bytes row={f_row:.4%} "
          f"col={f_col:.4%} | rstd max rel {rstd_rel:.2e} | amax max rel diff {a_rel:.2e}", flush=True)
    return ok


def selftest(shapes=((16384, 28672), (2048, 4096)), srs=(True, False), seeds=(1234, 7),
             swizzled=(True,)) -> int:
    import transformer_engine
    import transformer_engine_torch as tex

    _build()
    orig_dswiglu = _orig["dswiglu"]
    dev = torch.device("cuda", torch.cuda.current_device())
    print(f"nvfp4_producer_quant selftest: torch {torch.__version__}, TE {transformer_engine.__version__}, "
          f"device {torch.cuda.get_device_name(dev)} ({_arch.sm()})", flush=True)
    ok_all = True
    for sw in swizzled:
        for (m, _n) in shapes:
            ok_all = _rmsnorm_check(m, sw, dev) and ok_all
    for sw in swizzled:
        for sr in srs:
            for (m, n) in shapes:
                g = torch.Generator(device=dev).manual_seed(m * 31 + n)
                inp = torch.randn(m, n, device=dev, dtype=torch.float32, generator=g)
                inp = (inp * torch.exp(torch.randn(m, n, device=dev, generator=g) * 2.0)).to(torch.bfloat16)
                grad = torch.randn(m, n // 2, device=dev, dtype=torch.float32, generator=g)
                grad = (grad * torch.exp(torch.randn(m, n // 2, device=dev, generator=g) * 2.0)).to(torch.bfloat16)
                for seed in seeds:
                    for name, ref_fn, our_fn in (
                        ("dswiglu", lambda q: orig_dswiglu(grad, inp, q),
                         lambda q: produce(DSWIGLU, q, [m, n], m, n, grad, in1=inp)),
                        ("ident", lambda q: tex.quantize(grad, q),
                         lambda q: produce(IDENT, q, [m, n // 2], m, n // 2, grad)),
                    ):
                        torch.cuda.manual_seed(seed)
                        ref = ref_fn(_make_quantizer(sr, sw))
                        torch.cuda.manual_seed(seed)
                        out = our_fn(_make_quantizer(sr, sw))
                        torch.cuda.synchronize()
                        parts = _compare_out(out, ref)
                        note = ""
                        if name == "dswiglu" and not parts[0].endswith("=ok") and all(
                                p.endswith("=ok") for p in parts[1:]):
                            same = torch.equal(out.dequantize(dtype=torch.float32),
                                               ref.dequantize(dtype=torch.float32))
                            if same:
                                nz = parts[0].split(":")[-1]
                                parts[0] = "rowwise_data=ok"
                                note = f" (row bytes differing only at zero-decoding codes: {nz})"
                        ok = all(p.endswith("=ok") for p in parts)
                        ok_all = ok_all and ok
                        shape = (m, n) if name == "dswiglu" else (m, n // 2)
                        print(f"SELFTEST {name} swizzled={sw} sr={sr} shape={shape} seed={seed} "
                              f"{'OK' if ok else 'MISMATCH'} " + " ".join(parts) + note, flush=True)
                        del ref, out
    print("SELFTEST " + ("PASS" if ok_all else "FAIL"), flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(selftest(shapes=((2048, 4096),) if "--quick" in sys.argv else ((16384, 28672), (2048, 4096))))
    print("usage: python -m opt_kernels.nvfp4_producer_quant --selftest [--quick]")
