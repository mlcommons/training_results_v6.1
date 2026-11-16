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
# The cross-entropy kernel is a derivative of Transformer Engine's common/triton/cross_entropy.py;
# the wgrad accumulation in backward follows Megatron-Core's tensor_parallel/layers.py; the
# rerun-state-machine block and the GPTModel._postprocess rewrite follow Megatron-Bridge's
# losses.py and Megatron-Core's gpt_model.py (all Apache-2.0, Copyright (c) NVIDIA CORPORATION
# & AFFILIATES).
from __future__ import annotations

import inspect
import os
import sys
import textwrap
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from opt_kernels import skip_grad_buffer_zero as _sgz

IGNORE_INDEX = -100
_MAX_BLOCK = 65536 // 2


class _S:
    installed = False
    enabled = False
    block = 32768
    warps = 32
    grad_block = 16384
    evict = 0
    grad_form = 0
    loss_form = 0
    debug = 0
    calls = 0
    bwd_calls = 0
    checked = set()
    orig_loss_fn = None
    gaf_warned = False


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def _log(msg: str) -> None:
    sys.stderr.write(f"[opt lm-head-ce r{_rank()}] {msg}\n")
    sys.stderr.flush()


@triton.jit
def _lmhead_ce_prep_kernel(
    Y_ptr, Y_stride_b, Y_stride_s, NB, n_rows, ignore_index,
    out_ptr,
    BLOCK: tl.constexpr,
):
    cnt = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(0, n_rows, BLOCK):
        r = (i + tl.arange(0, BLOCK)).to(tl.int64)
        valid = r < n_rows
        s_idx = r // NB
        b_idx = r - s_idx * NB
        y = tl.load(Y_ptr + b_idx * Y_stride_b + s_idx * Y_stride_s, mask=valid, other=ignore_index)
        cnt += (y != ignore_index).to(tl.int32)
    c = tl.sum(cnt).to(tl.float32)
    cl = tl.maximum(c, 1.0)
    one = cl * 0.0 + 1.0
    tl.store(out_ptr + 0, c)
    tl.store(out_ptr + 1, cl)
    tl.store(out_ptr + 2, tl.div_rn(one, cl))


@triton.jit
def _lmhead_ce_kernel(
    X_ptr, X_stride,
    Y_ptr, Y_stride_b, Y_stride_s,
    loss_ptr, L_stride_b, L_stride_s,
    scale_ptr,
    NB, n_cols, ignore_index,
    BLOCK_SIZE: tl.constexpr,
    GRAD_BLOCK: tl.constexpr,
    WRITE_GRAD: tl.constexpr,
    GRAD_FORM: tl.constexpr,
    LOSS_FORM: tl.constexpr,
    EVICT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    s_idx = row // NB
    b_idx = row - s_idx * NB
    y = tl.load(Y_ptr + b_idx * Y_stride_b + s_idx * Y_stride_s)
    X_ptr += row * X_stride
    loss_out = loss_ptr + b_idx * L_stride_b + s_idx * L_stride_s
    if y == ignore_index:
        if WRITE_GRAD:
            for i in range(0, n_cols, GRAD_BLOCK):
                offs = i + tl.arange(0, GRAD_BLOCK)
                tl.store(X_ptr + offs, tl.zeros([GRAD_BLOCK], dtype=tl.float32).to(X_ptr.dtype.element_ty),
                         mask=offs < n_cols)
        tl.store(loss_out, 0.0)
        return
    m = float("-inf")
    d = 0.0
    x_y = tl.load(X_ptr + y).to(tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        if EVICT == 1:
            x = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf"), eviction_policy="evict_last").to(tl.float32)
        else:
            x = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
    if WRITE_GRAD:
        scale = tl.load(scale_ptr)
        lse = m + tl.log(d)
        for i in range(0, n_cols, GRAD_BLOCK):
            offs = i + tl.arange(0, GRAD_BLOCK)
            if EVICT == 1:
                x = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf"), eviction_policy="evict_first").to(tl.float32)
            else:
                x = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
            if GRAD_FORM == 0:
                p = tl.exp(x - m) / d
                g = tl.where(offs != y, p, p - 1.0) * scale
            elif GRAD_FORM == 1:
                p = tl.exp(x - lse)
                g = tl.where(offs != y, p, p - 1.0) * scale
            elif GRAD_FORM == 2:
                p = tl.exp(x - m) / d
                g = tl.where(offs != y, p * scale, p * scale - scale)
            else:
                p = tl.exp(x - m) / d
                g = tl.where(offs != y, p, p - 1.0)
                g = g * scale
            if EVICT == 1:
                tl.store(X_ptr + offs, g.to(X_ptr.dtype.element_ty), mask=offs < n_cols, eviction_policy="evict_first")
            else:
                tl.store(X_ptr + offs, g.to(X_ptr.dtype.element_ty), mask=offs < n_cols)
    if LOSS_FORM == 0:
        loss = -(x_y - m - tl.log(d))
    elif LOSS_FORM == 1:
        loss = (m + tl.log(d)) - x_y
    else:
        loss = tl.log(d) - (x_y - m)
    tl.store(loss_out, loss)


@triton.jit
def _scale_inplace_kernel(x_ptr, g_ptr, n, BLOCK: tl.constexpr):
    g = tl.load(g_ptr).to(tl.float32)
    if g != 1.0:
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        for start in range(pid * BLOCK, n, nprog * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(x_ptr + offs, (x * g).to(x_ptr.dtype.element_ty), mask=mask)


def _scale_inplace(x: torch.Tensor, g: torch.Tensor) -> None:
    assert x.is_contiguous()
    n = x.numel()
    grid = (min(triton.cdiv(n, 4096), 2048),)
    _scale_inplace_kernel[grid](x, g, n, BLOCK=4096, num_warps=4)


def ce_prep(labels: torch.Tensor, n_rows: int, nb: int, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    out = torch.empty(3, dtype=torch.float32, device=labels.device)
    ysb, yss = (labels.stride(0), labels.stride(1)) if labels.dim() == 2 else (0, labels.stride(0))
    _lmhead_ce_prep_kernel[(1,)](labels, ysb, yss, nb, n_rows, ignore_index, out, BLOCK=1024, num_warps=4)
    return out


def ce_rows_inplace(
    logits: torch.Tensor, labels: torch.Tensor, loss_bs: torch.Tensor, scale: torch.Tensor,
    nb: int, write_grad: bool, ignore_index: int = IGNORE_INDEX,
    block: Optional[int] = None, warps: Optional[int] = None, grad_form: Optional[int] = None,
    loss_form: Optional[int] = None, grad_block: Optional[int] = None, evict: Optional[int] = None,
) -> None:
    T, V = logits.shape
    assert logits.stride(1) == 1
    assert labels.dtype == torch.int64
    if labels.dim() == 2:
        ysb, yss = labels.stride(0), labels.stride(1)
        assert labels.shape[0] == nb and labels.shape[0] * labels.shape[1] == T
    else:
        ysb, yss = 0, labels.stride(0)
        assert nb == 1 and labels.numel() == T
    if loss_bs.dim() == 2:
        lsb, lss = loss_bs.stride(0), loss_bs.stride(1)
    else:
        lsb, lss = 0, loss_bs.stride(0)
    block = block or _S.block
    warps = warps or _S.warps
    block = min(block, triton.next_power_of_2(V))
    grad_block = min(grad_block or _S.grad_block, triton.next_power_of_2(V))
    _lmhead_ce_kernel[(T,)](
        logits, logits.stride(0), labels, ysb, yss, loss_bs, lsb, lss, scale, nb, V, ignore_index,
        BLOCK_SIZE=block, GRAD_BLOCK=grad_block, WRITE_GRAD=write_grad,
        EVICT=_S.evict if evict is None else evict,
        GRAD_FORM=_S.grad_form if grad_form is None else grad_form,
        LOSS_FORM=_S.loss_form if loss_form is None else loss_form,
        num_warps=warps,
    )


def _get_dummy_wgrad(shape, dtype, zero=False):
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    return get_dummy_wgrad(list(shape), dtype, zero=zero) if zero else get_dummy_wgrad(list(shape), dtype)


class FusedLMHeadCE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden2d, weight, labels, nb, write_grad, gradient_accumulation_fusion):
        T, H = hidden2d.shape
        V = weight.shape[0]
        logits = torch.matmul(hidden2d, weight.t())
        prep = ce_prep(labels, T, nb)
        loss_bs = torch.empty((nb, T // nb), dtype=torch.float32, device=logits.device)
        ce_rows_inplace(logits, labels, loss_bs, prep[2:3], nb, write_grad)
        loss_sum = loss_bs.sum()
        n_tokens = prep[0]
        loss = loss_sum / prep[1]
        _S.calls += 1
        if _S.debug and _S.calls <= _S.debug:
            _debug_compare(hidden2d, weight, labels, nb, write_grad, logits, loss_bs, prep)
        ctx.save_for_backward(hidden2d, weight, logits)
        ctx.main_grad = weight.main_grad if (gradient_accumulation_fusion and hasattr(weight, "main_grad")) else None
        ctx.gaf = gradient_accumulation_fusion
        ctx.write_grad = write_grad
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(loss_sum, n_tokens)
        return loss, loss_sum, n_tokens

    @staticmethod
    def backward(ctx, g_loss, g_sum, g_cnt):
        hidden2d, weight, dlogits = ctx.saved_tensors
        assert ctx.write_grad, "opt lm_head_ce: backward without a gradient pass in the forward"
        _S.bwd_calls += 1
        if ctx.gaf:
            weight.main_grad = ctx.main_grad
        if g_loss is None:
            g_loss = torch.ones((), dtype=torch.float32, device=dlogits.device)
        g_loss = g_loss.reshape(-1)[:1].to(torch.float32)
        grad_input = torch.matmul(dlogits, weight)
        _scale_inplace(grad_input, g_loss)
        _scale_inplace(hidden2d, g_loss)
        if ctx.gaf:
            if _sgz.enabled():
                _sgz.write_lm_head_wgrad(hidden2d, dlogits, weight.main_grad)
            else:
                import fused_weight_gradient_mlp_cuda

                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(hidden2d, dlogits, weight.main_grad)
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(hidden2d, dlogits, weight.main_grad)
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            if hasattr(weight, "grad_added_to_main_grad"):
                grad_weight = _get_dummy_wgrad(weight.main_grad.shape, hidden2d.dtype,
                                               zero=getattr(weight, "zero_out_wgrad", False))
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = dlogits.t().matmul(hidden2d)
        return grad_input, grad_weight, None, None, None, None


def _debug_compare(hidden2d, weight, labels, nb, write_grad, mine_grad, mine_loss_bs, prep):
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        import transformer_engine.pytorch.triton.cross_entropy as tece

        T, H = hidden2d.shape
        V = weight.shape[0]
        logits = torch.matmul(hidden2d, weight.t())
        if labels.dim() == 2:
            tgt = labels.transpose(0, 1).contiguous().view(-1)
        else:
            tgt = labels.contiguous()
        inp3 = logits.view(T // nb, nb, V)
        out = tece.cross_entropy_forward(inp3, tgt, 0.0, False, None, IGNORE_INDEX, True)
        ref_loss, saved, stats, tgt2, _, rank, world = out
        ref_loss_bs = ref_loss.reshape(T // nb, nb).transpose(0, 1).contiguous()
        msg = [f"call {_S.calls}: T={T} V={V} nb={nb} n_non_ignore={prep[0].item():.0f}"]
        eq_loss = torch.equal(mine_loss_bs, ref_loss_bs)
        msg.append(f"loss bitwise_eq={eq_loss} maxabs={(mine_loss_bs - ref_loss_bs).abs().max().item():.3e}")
        if write_grad:
            grad_rows = torch.where(tgt != IGNORE_INDEX, prep[2], torch.zeros_like(prep[2])).contiguous()
            tece.cross_entropy_backward(saved, stats, tgt2, None, grad_rows, 0.0, False, rank, world, IGNORE_INDEX)
            ref_grad = saved.view(T, V)
            eq = torch.equal(mine_grad, ref_grad)
            diff = (mine_grad.float() - ref_grad.float()).abs()
            msg.append(f"dlogits bitwise_eq={eq} maxabs={diff.max().item():.3e} n_diff={(diff > 0).sum().item()}")
        _log("DEBUG " + " | ".join(msg))
        del logits, saved, stats
    except Exception as exc:
        _log(f"DEBUG compare failed: {exc!r}")


def _check_output_layer(model, output_layer, weight) -> None:
    key = id(output_layer)
    if key in _S.checked:
        return
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    problems = []
    if not isinstance(output_layer, ColumnParallelLinear):
        problems.append(f"output_layer is {type(output_layer).__name__}, not Megatron ColumnParallelLinear")
    tp = getattr(output_layer, "tp_group", None)
    tp_size = tp.size() if tp is not None else 1
    if tp_size != 1:
        problems.append(f"tensor-parallel size {tp_size} (vocab-parallel CE not implemented)")
    if getattr(output_layer, "sequence_parallel", False):
        problems.append("sequence_parallel output layer")
    if getattr(output_layer, "bias", None) is not None:
        problems.append("output layer has a bias")
    if getattr(output_layer, "explicit_expert_comm", False):
        problems.append("explicit_expert_comm")
    if getattr(output_layer, "gtp_remat_size", 1) != 1:
        problems.append("gtp_remat_size != 1")
    out_dtype = getattr(output_layer, "output_dtype", None)
    if out_dtype not in (None, torch.bfloat16):
        problems.append(f"output_dtype {out_dtype}")
    cfg = output_layer.config
    if getattr(cfg, "defer_embedding_wgrad_compute", False):
        problems.append("defer_embedding_wgrad_compute")
    if getattr(cfg, "use_mup", False):
        problems.append("use_mup (logit scaling)")
    if getattr(cfg, "_cpu_offloading_context", None) is not None and getattr(cfg._cpu_offloading_context, "inside_context", False):
        problems.append("cpu offloading context")
    if getattr(cfg, "mtp_num_layers", None):
        problems.append("multi-token prediction")
    if getattr(cfg, "fp16_lm_cross_entropy", False) or getattr(model, "fp16_lm_cross_entropy", False):
        problems.append("fp16_lm_cross_entropy")
    if weight.dtype != torch.bfloat16:
        problems.append(f"weight dtype {weight.dtype}")
    if problems:
        raise RuntimeError("opt lm_head_ce: fused lm_head + CE cannot be applied: " + "; ".join(problems))
    _S.checked.add(key)
    _log(f"output layer accepted: {type(output_layer).__name__} {tuple(weight.shape)} {weight.dtype}, "
         f"gradient_accumulation_fusion={output_layer.gradient_accumulation_fusion}, "
         f"main_grad={getattr(getattr(weight, 'main_grad', None), 'dtype', None)}, "
         f"kernel block={_S.block} warps={_S.warps} grad_block={_S.grad_block} evict={_S.evict} "
         f"grad_form={_S.grad_form} loss_form={_S.loss_form}")


def fused_path_applies(model, labels, in_inference_mode, runtime_gather_output) -> bool:
    if not _S.enabled or labels is None or in_inference_mode:
        return False
    from megatron.core.config_logger import has_config_logger_enabled

    reason = None
    if runtime_gather_output:
        reason = "runtime_gather_output"
    elif has_config_logger_enabled(model.config):
        reason = "the config logger"
    if reason is None:
        return True
    if torch.is_grad_enabled() and _sgz.enabled():
        raise RuntimeError(
            f"opt lm_head_ce: {reason} takes the stock output-layer path in a training step while "
            "skip_grad_buffer_zero is enabled (the output layer's main_grad would accumulate across steps); "
            "set OPT_SKIP_GRAD_ZERO_DISABLE=1 or OPT_LMHEAD_CE_DISABLE=1"
        )
    return False


def fused_lm_head_ce(model, hidden_states, output_weight, labels):
    output_layer = model.output_layer
    weight = output_weight if output_weight is not None else output_layer.weight
    _check_output_layer(model, output_layer, weight)
    s, b, H = hidden_states.shape
    assert labels.shape == (b, s), f"labels {tuple(labels.shape)} vs hidden [s={s}, b={b}]"
    hidden2d = hidden_states.reshape(s * b, H)
    write_grad = torch.is_grad_enabled() and (hidden_states.requires_grad or weight.requires_grad)
    loss, loss_sum, n_tokens = FusedLMHeadCE.apply(
        hidden2d, weight, labels, b, write_grad, output_layer.gradient_accumulation_fusion
    )
    return (loss, loss_sum, n_tokens)


def masked_next_token_loss(loss_mask, output_tensor, check_for_nan_in_loss=True, check_for_spiky_loss=False):
    if not (isinstance(output_tensor, (tuple, list)) and len(output_tensor) == 3
            and all(isinstance(t, torch.Tensor) and t.dim() == 0 for t in output_tensor)):
        return _S.orig_loss_fn(loss_mask, output_tensor, check_for_nan_in_loss=check_for_nan_in_loss,
                               check_for_spiky_loss=check_for_spiky_loss)
    loss, loss_sum, n_tokens = output_tensor
    from functools import partial

    from megatron.core.rerun_state_machine import get_rerun_state_machine

    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        rerun_state_machine.validate_result(result=loss_sum, rejection_func=torch.isnan,
                                            message="found NaN in local forward loss calculation",
                                            tolerance=0.0, fatal=True)
        rerun_state_machine.validate_result(result=loss_sum, rejection_func=torch.isinf,
                                            message="found Inf in local forward loss calculation",
                                            tolerance=0.0, fatal=True)
    if check_for_spiky_loss:
        spiky_loss_factor = getattr(rerun_state_machine, "spiky_loss_factor", 10.0)
        rerun_state_machine.validate_result(
            result=loss_sum,
            rejection_func=partial(rerun_state_machine.is_unexpectedly_large, threshold=spiky_loss_factor, context="loss"),
            message="Spiky loss", tolerance=0.0, fatal=False)
    reporting_loss = torch.stack([loss_sum, n_tokens])
    from megatron.core import parallel_state

    cp = parallel_state.get_context_parallel_world_size()
    if cp > 1:
        loss = loss / cp
    return (loss, {"lm loss": reporting_loss})


def _patch_postprocess() -> None:
    import megatron.core.models.gpt.gpt_model as gm

    fn = gm.GPTModel._postprocess
    src = textwrap.dedent(inspect.getsource(fn))
    needle = (
        "    logits, _ = self.output_layer(\n"
        "        hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output\n"
        "    )\n"
    )
    if src.count(needle) != 1:
        raise RuntimeError("opt lm_head_ce: GPTModel._postprocess does not contain the expected output_layer call")
    insert = (
        "    # lm_head_ce_fusion: one Function for the lm_head GEMM + cross-entropy (no logits copy,\n"
        "    # no separate CE backward pass, token gradient written once); labels carry the loss mask as ignore_index.\n"
        "    if _opt_lm_head_ce.fused_path_applies(self, labels, in_inference_mode, runtime_gather_output):\n"
        "        return _opt_lm_head_ce.fused_lm_head_ce(self, hidden_states, output_weight, labels)\n"
    )
    src = src.replace(needle, insert + needle)
    ns = gm.__dict__
    ns["_opt_lm_head_ce"] = sys.modules[__name__]
    code = compile(src, gm.__file__, "exec")
    local_ns = {}
    exec(code, ns, local_ns)
    gm.GPTModel._postprocess = local_ns["_postprocess"]


def _patch_get_batch() -> None:
    import megatron.bridge.training.gpt_step as gs

    orig = gs.get_batch

    def get_batch(*args, **kwargs):
        out = orig(*args, **kwargs)
        tokens, labels, loss_mask, attention_mask, position_ids, packed = out
        if labels is not None and loss_mask is not None and _S.enabled:
            labels = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
        return tokens, labels, loss_mask, attention_mask, position_ids, packed

    get_batch.__wrapped__ = orig
    gs.get_batch = get_batch


def _patch_loss_function() -> None:
    import megatron.bridge.training.gpt_step as gs
    import megatron.bridge.training.losses as losses

    _S.orig_loss_fn = losses.masked_next_token_loss
    losses.masked_next_token_loss = masked_next_token_loss
    if getattr(gs, "masked_next_token_loss", None) is _S.orig_loss_fn:
        gs.masked_next_token_loss = masked_next_token_loss


def install(tp_size: int = 1, sequence_parallel: bool = False, calculate_per_token_loss: bool = False) -> bool:
    if _S.installed:
        return _S.enabled
    _S.installed = True
    if os.environ.get("OPT_LMHEAD_CE_DISABLE", "0") == "1":
        _log("disabled by OPT_LMHEAD_CE_DISABLE=1")
        return False
    unsupported = []
    if tp_size != 1:
        unsupported.append(f"tensor-parallel size {tp_size} (vocab-parallel CE not implemented)")
    if sequence_parallel:
        unsupported.append("sequence_parallel output layer")
    if calculate_per_token_loss:
        unsupported.append("calculate_per_token_loss (the fused token gradient is pre-normalised by 1/n)")
    if unsupported:
        _log("not installed: " + "; ".join(unsupported) + "; Megatron-Bridge's native output layer + cross-entropy run")
        return False
    _S.block = int(os.environ.get("OPT_LMHEAD_CE_BLOCK", str(_MAX_BLOCK)))
    _S.warps = int(os.environ.get("OPT_LMHEAD_CE_WARPS", "32"))
    _S.grad_block = int(os.environ.get("OPT_LMHEAD_CE_GRAD_BLOCK", str(_S.grad_block)))
    _S.evict = int(os.environ.get("OPT_LMHEAD_CE_EVICT", str(_S.evict)))
    _S.grad_form = int(os.environ.get("OPT_LMHEAD_CE_GRAD_FORM", "0"))
    _S.loss_form = int(os.environ.get("OPT_LMHEAD_CE_LOSS_FORM", "0"))
    _S.debug = int(os.environ.get("OPT_LMHEAD_CE_DEBUG", "0"))
    _patch_postprocess()
    _patch_get_batch()
    _patch_loss_function()
    _S.enabled = True
    _log(f"installed (block={_S.block} warps={_S.warps} grad_block={_S.grad_block} evict={_S.evict} grad_form={_S.grad_form} loss_form={_S.loss_form} debug={_S.debug})")
    return True


def enabled() -> bool:
    return _S.enabled


def _bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
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


def _bench_pre(pre, fn, iters=8, warmup=2):
    ts = []
    for i in range(warmup + iters):
        pre()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if i >= warmup:
            ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def _selftest(quick: bool = False) -> int:
    import transformer_engine.pytorch.triton.cross_entropy as tece

    torch.manual_seed(0)
    dev = "cuda"
    T, V, H = 16384, 128256, 4096
    B, S = 2, T // 2
    print(f"selftest: torch {torch.__version__} triton {triton.__version__} {torch.cuda.get_device_name()} cc {torch.cuda.get_device_capability()}", flush=True)
    X = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    W = torch.randn(V, H, device=dev, dtype=torch.bfloat16) * 0.02
    labels_bs = torch.randint(0, V, (B, S), device=dev, dtype=torch.int64)
    labels_bs[0, :37] = IGNORE_INDEX
    labels_bs[1, 100:103] = IGNORE_INDEX
    logits = torch.matmul(X, W.t())
    tgt = labels_bs.transpose(0, 1).contiguous().view(-1)
    n_non_ignore = (tgt != IGNORE_INDEX).sum()
    buf = logits.clone()
    ref = tece.cross_entropy_forward(buf.view(S, B, V), tgt, 0.0, False, None, IGNORE_INDEX, False)
    ref_loss, saved, stats_t, tgt2, _, rank, world = ref
    inv = (torch.ones((), device=dev) / n_non_ignore.float()).item()
    inv_t = torch.full((), 1.0, device=dev) / torch.clamp(n_non_ignore.float(), min=1.0)
    grad_rows = torch.where(tgt != IGNORE_INDEX, inv_t, torch.zeros((), device=dev)).contiguous()
    tece.cross_entropy_backward(saved, stats_t, tgt2, None, grad_rows, 0.0, False, rank, world, IGNORE_INDEX)
    ref_grad = saved.view(T, V)
    ref_loss_bs = ref_loss.reshape(S, B).transpose(0, 1).contiguous()
    ref_sum = torch.sum(ref_loss_bs.view(-1).float() * (labels_bs.view(-1) != IGNORE_INDEX).float())
    print(f"reference: n_non_ignore={n_non_ignore.item()} inv={inv:.9e} loss_sum={ref_sum.item():.7f} stats shape={tuple(stats_t.shape)}", flush=True)
    prep = ce_prep(labels_bs, T, B)
    ok_prep = (prep[0].item() == n_non_ignore.item()) and torch.equal(prep[2].reshape(()), inv_t.reshape(()))
    print(f"prep kernel: count={prep[0].item():.0f} inv={prep[2].item():.9e} bitwise_eq_torch={ok_prep}", flush=True)
    rc = 0
    match = None
    loss_bs = torch.empty((B, S), dtype=torch.float32, device=dev)
    for gf in range(4):
        for lf in range(3):
            buf.copy_(logits)
            ce_rows_inplace(buf, labels_bs, loss_bs, prep[2:3], B, True, grad_form=gf, loss_form=lf)
            eq_g = torch.equal(buf, ref_grad)
            eq_l = torch.equal(loss_bs, ref_loss_bs)
            dg = (buf.float() - ref_grad.float()).abs().max().item()
            dl = (loss_bs - ref_loss_bs).abs().max().item()
            print(f"variant grad_form={gf} loss_form={lf}: dlogits bitwise_eq={eq_g} maxabs={dg:.3e} | loss bitwise_eq={eq_l} maxabs={dl:.3e}", flush=True)
            if eq_g and eq_l and match is None:
                match = (gf, lf)
    print(f"MATCH grad_form/loss_form = {match}", flush=True)
    if match is None:
        rc = 1
    buf.copy_(logits)
    ce_rows_inplace(buf, labels_bs, loss_bs, prep[2:3], B, True)
    row0 = 0 * B + 0
    print(f"masked row zero grad={bool(buf[row0].float().abs().max() == 0)} zero loss={bool(loss_bs[0, 0] == 0)}", flush=True)
    mysum = loss_bs.sum()
    print(f"loss_sum bitwise_eq={torch.equal(mysum, ref_sum)} ({mysum.item():.7f} vs {ref_sum.item():.7f}); "
          f"normalised bitwise_eq={torch.equal(mysum / prep[1], ref_sum / torch.clamp(n_non_ignore, min=1))}", flush=True)
    buf.copy_(logits)
    ce_rows_inplace(buf, labels_bs, loss_bs, prep[2:3], B, False)
    print(f"no-grad pass leaves logits intact={torch.equal(buf, logits)} loss_eq={torch.equal(loss_bs, ref_loss_bs)}", flush=True)
    t = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    t2 = t.clone()
    _scale_inplace(t2, torch.ones(1, device=dev))
    g = torch.full((1,), 0.5, device=dev)
    t3 = t.clone()
    _scale_inplace(t3, g)
    print(f"scale kernel: g=1 no-op={torch.equal(t2, t)} g=0.5 eq={torch.equal(t3, (t.float() * 0.5).to(torch.bfloat16))}", flush=True)
    from unittest import mock

    from megatron.core import parallel_state

    Vs, Ts, Ss = 8192, 4096, 2048
    lab0 = labels_bs[:, :Ss].contiguous()
    lab = torch.where(lab0 == IGNORE_INDEX, lab0, lab0 % Vs)
    Ws = W[:Vs].clone()

    def legacy_path(cp):
        Xg = X[:Ts].clone().requires_grad_(True)
        with mock.patch.object(parallel_state, "get_context_parallel_world_size", return_value=cp):
            out = FusedLMHeadCE.apply(Xg.view(-1, H), Ws, lab, B, True, False)
            loss_t, _report = masked_next_token_loss(None, out, check_for_nan_in_loss=False, check_for_spiky_loss=False)
        loss_t *= cp
        loss_t.backward()
        return loss_t.detach(), Xg.grad

    l1, g1 = legacy_path(1)
    ok_cp = True
    for cp in (2, 3, 4, 8):
        lc, gc = legacy_path(cp)
        ok_cp &= torch.equal(g1, gc)
        print(f"legacy-path contract cp={cp}: hidden grad bitwise_eq_cp1={torch.equal(g1, gc)} loss={lc.item():.7f} (cp1 {l1.item():.7f})", flush=True)
    rc += 0 if ok_cp else 1
    print(f"legacy-path contract: gradient independent of cp={ok_cp}", flush=True)
    if quick:
        return rc
    ms = _bench_pre(lambda: buf.copy_(logits), lambda: tece.cross_entropy_forward(buf.view(S, B, V), tgt, 0.0, False, None, IGNORE_INDEX, False))
    print(f"TIME te_fwd(copy_input) ms={ms:.3f}", flush=True)
    ms = _bench_pre(lambda: buf.copy_(logits), lambda: tece.cross_entropy_forward(buf.view(S, B, V), tgt, 0.0, False, None, IGNORE_INDEX, True))
    print(f"TIME te_fwd(overwrite_input) ms={ms:.3f}", flush=True)
    saved2 = logits.clone()
    ms = _bench_pre(lambda: saved2.copy_(logits), lambda: tece.cross_entropy_backward(saved2.view(S, B, V), stats_t, tgt2, None, grad_rows, 0.0, False, rank, world, IGNORE_INDEX))
    print(f"TIME te_bwd ms={ms:.3f}", flush=True)
    del saved2
    best = None
    configs = [(32768, 32, gb, ev) for gb in (4096, 8192, 16384, 32768, 65536) for ev in (0, 1)]
    configs += [(32768, 16, 32768, 0), (8192, 8, 8192, 0)]
    for block, warps, gb, ev in configs:
        try:
            ms = _bench_pre(lambda: buf.copy_(logits), lambda: ce_rows_inplace(buf, labels_bs, loss_bs, prep[2:3], B, True, block=block, warps=warps, grad_block=gb, evict=ev))
            eq = torch.equal(buf, ref_grad) if match is not None else "n/a"
            print(f"TIME ours block={block} warps={warps} grad_block={gb} evict={ev} ms={ms:.3f} GBps={(2 * T * V * 2) / ms / 1e6:.0f} bitwise_eq_TE={eq}", flush=True)
            if eq is True and (best is None or ms < best[0]):
                best = (ms, block, warps, gb, ev)
        except Exception as ex:
            print(f"TIME ours block={block} warps={warps} grad_block={gb} evict={ev} FAILED {ex!r}", flush=True)
    print(f"BEST ours (bitwise-identical configs only) {best}", flush=True)
    ms = _bench_pre(lambda: buf.copy_(logits), lambda: ce_rows_inplace(buf, labels_bs, loss_bs, prep[2:3], B, False))
    print(f"TIME ours no-grad (stats+loss only) ms={ms:.3f}", flush=True)
    ms = _bench(lambda: ce_prep(labels_bs, T, B))
    print(f"TIME prep ms={ms:.4f}", flush=True)
    ms = _bench(lambda: _scale_inplace(t2, torch.ones(1, device=dev)))
    print(f"TIME scale g=1 (no-op) ms={ms:.4f}", flush=True)
    ms = _bench(lambda: _scale_inplace(t3, g))
    print(f"TIME scale g=0.5 ms={ms:.4f}", flush=True)
    ms = _bench(lambda: loss_bs.sum())
    print(f"TIME loss sum ms={ms:.4f}", flush=True)
    def fused_fwd():
        lg = torch.matmul(X, W.t())
        p = ce_prep(labels_bs, T, B)
        ce_rows_inplace(lg, labels_bs, loss_bs, p[2:3], B, True)
        return loss_bs.sum() / p[1]
    def today_fwd():
        lg = torch.matmul(X, W.t())
        return tece.cross_entropy_forward(lg.view(S, B, V), tgt, 0.0, False, None, IGNORE_INDEX, False)
    print(f"TIME fwd fused(gemm+prep+ce+sum+div) ms={_bench(fused_fwd, iters=5):.3f} | today(gemm+te_fwd copy) ms={_bench(today_fwd, iters=5):.3f}", flush=True)
    return rc


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest(quick="--quick" in sys.argv))
    print("usage: python -m opt_kernels.lm_head_ce_fusion --selftest [--quick]")
