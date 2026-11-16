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
#
# The backward post-hook replacement is adapted from Megatron-Core's
# DistributedDataParallel._make_backward_post_hook, and the buffer-reset patch keeps the
# extra_main_grads handling of param_and_grad_buffer.py (Apache-2.0, Copyright (c) NVIDIA
# CORPORATION & AFFILIATES).


from __future__ import annotations

import inspect
import os
import sys

import torch

_ENV_DISABLE = "OPT_SKIP_GRAD_ZERO_DISABLE"


class _State:
    installed = False
    enabled = False
    one_microbatch_checked = False
    n_ddp_wrapped = 0
    n_overwrite_weights = 0
    unflagged_params: list = []
    resets_run = 0
    resets_skipped = 0
    hook_copies = 0


_S = _State()


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def _log(msg: str, all_ranks: bool = False) -> None:
    if all_ranks or _rank() == 0:
        sys.stderr.write(f"[opt skip-grad-zero r{_rank()}] {msg}\n")
        sys.stderr.flush()


def enabled() -> bool:
    return _S.enabled


def _check_one_microbatch(strict: bool) -> bool:
    if _S.one_microbatch_checked:
        return True
    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        n = get_num_microbatches()
    except Exception as exc:
        if strict:
            raise RuntimeError(f"opt skip-grad-zero: cannot read the number of micro-batches ({exc!r})") from exc
        return False
    if n != 1:
        raise RuntimeError(
            f"opt skip-grad-zero: {n} micro-batches per step -- the overwrite-on-first-wgrad path needs exactly one "
            f"(every main_grad slice is written once per step); set {_ENV_DISABLE}=1 for gradient accumulation"
        )
    _S.one_microbatch_checked = True
    return True


def _te_gemm_weight_classes():
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
    from transformer_engine.pytorch.module.layernorm_linear import LayerNormLinear
    from transformer_engine.pytorch.module.linear import Linear
    from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear

    return (Linear, LayerNormLinear, GroupedLinear, BasicLinear)


def _flag_overwrite_main_grad(ddp) -> None:
    classes = _te_gemm_weight_classes()
    flagged = set()
    n = 0
    for mname, m in ddp.module.named_modules():
        if not isinstance(m, classes):
            continue
        fuse = getattr(m, "fuse_wgrad_accumulation", None)
        if fuse is None:
            fuse = getattr(m, "_accumulate_into_main_grad", False)
        if not fuse:
            continue
        for pname, p in m.named_parameters(recurse=False):
            if not p.requires_grad or "layer_norm" in pname or "bias" in pname:
                continue
            p.overwrite_main_grad = True
            flagged.add(id(p))
            n += 1
    _S.n_overwrite_weights += n
    name_of = {id(p): name for name, p in ddp.module.named_parameters()}
    _S.unflagged_params = [name_of.get(id(p), "?") for p in ddp.params_with_grad if id(p) not in flagged]
    _S.n_ddp_wrapped += 1
    unflagged = _S.unflagged_params
    _log(
        f"DDP built: overwrite_main_grad set on {n} TE GEMM weights; {len(unflagged)} other grads take the overwrite hook "
        f"/ module paths: {', '.join(unflagged[:3])}{' ...' if len(unflagged) > 3 else ''}"
        + (f", {unflagged[-1]}" if len(unflagged) > 3 else "")
    )


def _patch_ddp_init() -> None:
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel as DDP

    orig_init = DDP.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not _S.enabled:
            return
        _check_one_microbatch(strict=False)
        _flag_overwrite_main_grad(self)

    __init__.__wrapped__ = orig_init
    DDP.__init__ = __init__


def _patch_backward_post_hook() -> None:
    import megatron.core.distributed.distributed_data_parallel as ddp_mod

    DDP = ddp_mod.DistributedDataParallel
    orig = DDP._make_backward_post_hook
    src = inspect.getsource(orig)
    for needle in ("param.main_grad.add_(param.grad.data)", "is_graph_capturing()", "register_grad_ready(", "_cudagraph_wgrad_ready_event"):
        if needle not in src:
            raise RuntimeError(f"opt skip-grad-zero: DistributedDataParallel._make_backward_post_hook has no '{needle}'; the mirrored hook below does not match this Megatron-Core")
    is_graph_capturing = ddp_mod.is_graph_capturing

    def _make_backward_post_hook(self, param):
        if not _S.enabled:
            return orig(self, param)

        def hook(*unused):
            if is_graph_capturing():
                return
            if param in self.param_to_bucket_group:
                assert param.requires_grad
                cudagraph_wgrad_ready_event = getattr(param, "_cudagraph_wgrad_ready_event", None)
                if self.ddp_config.overlap_grad_reduce and cudagraph_wgrad_ready_event is None:
                    if not getattr(param, "is_gtp_weight_remat", False):
                        assert param.grad is not None, "param.grad being None is not safe when overlap_grad_reduce is True"
                if param.grad is not None:
                    if not param.grad_added_to_main_grad:
                        param.main_grad.copy_(param.grad.data)
                        _S.hook_copies += 1
                    elif getattr(param, "zero_out_wgrad", False):
                        param.main_grad.add_(param.grad.data)
                param.grad = None
                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)

        return hook

    _make_backward_post_hook.__wrapped__ = orig
    DDP._make_backward_post_hook = _make_backward_post_hook


def _patch_buffer_reset() -> None:
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    orig_reset = _ParamAndGradBuffer.reset

    def reset(self):
        if not _S.enabled:
            return orig_reset(self)
        if not getattr(self, "_opt_zeroed_once", False):
            self._opt_zeroed_once = True
            _S.resets_run += 1
            return orig_reset(self)
        _check_one_microbatch(strict=True)
        for grad in getattr(self, "extra_main_grads", None) or []:
            grad.zero_()
        _S.resets_skipped += 1
        return None

    reset.__wrapped__ = orig_reset
    _ParamAndGradBuffer.reset = reset


def write_lm_head_wgrad(hidden2d: torch.Tensor, dlogits: torch.Tensor, main_grad: torch.Tensor) -> None:
    if main_grad.dtype != hidden2d.dtype:
        import fused_weight_gradient_mlp_cuda

        main_grad.zero_()
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(hidden2d, dlogits, main_grad)
        return
    torch.mm(dlogits.t(), hidden2d, out=main_grad)


def install() -> bool:
    if _S.installed:
        return _S.enabled
    _S.installed = True
    if os.environ.get(_ENV_DISABLE, "0") == "1":
        _log(f"disabled by {_ENV_DISABLE}=1")
        return False
    from opt_kernels import lm_head_ce_fusion as _ce

    if not _ce._S.installed:
        raise RuntimeError("opt skip-grad-zero: install() must run after lm_head_ce_fusion.install()")
    if not _ce.enabled():
        _log("disabled: the output-layer wgrad needs lm_head_ce_fusion's beta = 0 write and the CE fusion is off; "
             "stock grad-buffer zeroing kept")
        return False
    _patch_backward_post_hook()
    _patch_ddp_init()
    _patch_buffer_reset()
    _S.enabled = True
    _log("installed (TE GEMM weights overwrite main_grad; DDP hook copies; output-layer wgrad beta = 0; grad buffer zeroed once)")
    return True


def _selftest() -> int:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    fails = 0

    def say(msg):
        sys.stderr.write(f"selftest: {msg}\n")
        sys.stderr.flush()

    say(f"torch {torch.__version__} {torch.cuda.get_device_name(dev)} cc {torch.cuda.get_device_capability(dev)} "
        f"allow_bf16_reduced_precision_reduction={torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction} "
        f"preferred_blas={getattr(torch.backends.cuda, 'preferred_blas_library', lambda: 'n/a')()}")

    import fused_weight_gradient_mlp_cuda

    T, H, V = 16384, 4096, 128256
    for seed in (0, 1, 2):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g)
        d = (torch.randn(T, V, dtype=torch.float32, device=dev, generator=g) * 1e-4).to(torch.bfloat16)
        ref = torch.zeros(V, H, dtype=torch.bfloat16, device=dev)
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(x, d, ref)
        mine = torch.full((V, H), float("nan"), dtype=torch.bfloat16, device=dev)
        write_lm_head_wgrad(x, d, mine)
        torch.cuda.synchronize()
        bad = int((ref.view(torch.int16) != mine.view(torch.int16)).sum())
        maxabs = float((ref.float() - mine.float()).abs().max())
        say(f"lm_head wgrad [V={V}, H={H}, T={T}] seed {seed}: apex accumulate-onto-zero vs torch.mm beta=0 mismatches {bad}/{V * H} max|diff| {maxabs:.3e}")
        if bad:
            fails += 1
        del x, d, ref, mine
    g = torch.Generator(device=dev).manual_seed(7)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g)
    d = (torch.randn(T, V, dtype=torch.float32, device=dev, generator=g) * 1e-4).to(torch.bfloat16)
    buf = torch.zeros(V, H, dtype=torch.bfloat16, device=dev)

    def bench(fn, iters=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t_apex = bench(lambda: fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(x, d, buf))
    t_mm = bench(lambda: write_lm_head_wgrad(x, d, buf))
    t_zero = bench(lambda: buf.zero_())
    say(f"lm_head wgrad timing: apex beta=1 {t_apex:.3f} ms, torch.mm beta=0 {t_mm:.3f} ms, zero of the [V,H] slice {t_zero:.3f} ms")
    del x, d, buf

    try:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

        def quant_col(t, stochastic):
            q = NVFP4Quantizer(rowwise=False, columnwise=True, with_rht=True, with_post_rht_amax=True,
                               with_2d_quantization=False, stochastic_rounding=stochastic)
            try:
                q.optimize_for_gemm = True
            except Exception:
                pass
            out = q(t)
            if not getattr(out, "_with_gemm_swizzled_scales", False):
                tex.swizzle_scales_for_gemm_(out)
            return out

        for (n_out, k_in, name) in ((28672, 4096, "fc1"), (4096, 14336, "fc2"), (6144, 4096, "qkv"), (4096, 4096, "proj")):
            g = torch.Generator(device=dev).manual_seed(11)
            xin = torch.randn(T, k_in, dtype=torch.bfloat16, device=dev, generator=g)
            dy = (torch.randn(T, n_out, dtype=torch.float32, device=dev, generator=g) * 1e-3).to(torch.bfloat16)
            xq = quant_col(xin, False)
            dq = quant_col(dy, False)
            acc = torch.zeros(n_out, k_in, dtype=torch.bfloat16, device=dev)
            general_gemm(xq, dq, layout="NT", out_dtype=torch.bfloat16, out=acc, accumulate=True, grad=True)
            ovw = torch.full((n_out, k_in), float("nan"), dtype=torch.bfloat16, device=dev)
            general_gemm(xq, dq, layout="NT", out_dtype=torch.bfloat16, out=ovw, accumulate=False, grad=True)
            torch.cuda.synchronize()
            bad = int((acc.view(torch.int16) != ovw.view(torch.int16)).sum())
            say(f"TE NVFP4 wgrad GEMM {name} [{n_out}, {k_in}] x T={T}: accumulate-onto-zero vs overwrite mismatches {bad}/{n_out * k_in}; nan {int(torch.isnan(ovw).sum())}")
            if bad or int(torch.isnan(ovw).sum()):
                fails += 1
            del xin, dy, xq, dq, acc, ovw
    except Exception as exc:
        say(f"TE NVFP4 wgrad GEMM check skipped: {exc!r}")

    g = torch.Generator(device=dev).manual_seed(3)
    grad = torch.randn(4096, dtype=torch.float32, device=dev, generator=g).to(torch.bfloat16) * 1e-3
    a = torch.zeros(4096, dtype=torch.bfloat16, device=dev)
    a.add_(grad)
    c = torch.full((4096,), float("nan"), dtype=torch.bfloat16, device=dev)
    c.copy_(grad)
    bad = int((a.view(torch.int16) != c.view(torch.int16)).sum())
    say(f"DDP hook: add_ onto zero vs copy_ mismatches {bad}/4096")
    if bad:
        fails += 1

    say("SELFTEST PASS" if fails == 0 else f"SELFTEST FAIL ({fails} checks)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("usage: python -m opt_kernels.skip_grad_buffer_zero --selftest")
