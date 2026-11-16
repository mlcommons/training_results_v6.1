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
# The gradient-norm parameter filter is adapted from Megatron-Core's optimizer/optimizer.py and
# the capturable call order from Transformer Engine's pytorch/optimizers/fused_adam.py
# (Apache-2.0, Copyright (c) NVIDIA CORPORATION & AFFILIATES).


import math
import os

import torch

_ENABLED = os.environ.get("OPT_FUSED_ADAM", "1") != "0"
_KERNEL = os.environ.get("OPT_FUSED_ADAM_KERNEL", "triton")
_LOGGED = set()


def _log_once(key, msg):
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"[opt fused_adam] {msg}", flush=True)


class _FusedState:

    def __init__(self, opt):
        import transformer_engine_torch as tex

        self.tex = tex
        adam = opt.optimizer
        self.adam = adam
        master_to_model = {}
        for shard_groups, model_groups in (
            (opt.shard_fp32_from_float16_groups, opt.model_float16_groups),
            (opt.shard_fp32_groups, opt.model_fp32_groups),
        ):
            for sg, mg in zip(shard_groups, model_groups):
                for sp, mp in zip(sg, mg):
                    master_to_model[id(sp)] = mp
        dev = torch.device("cuda", torch.cuda.current_device())
        self.groups = []
        self.norm_grads = []
        for group in adam.param_groups:
            g_l, p_l, m_l, v_l, w_l = [], [], [], [], []
            for p in group["params"]:
                mp = master_to_model[id(p)]
                if len(adam.state[p]) == 0:
                    adam.initialize_state(p, False)
                st = adam.state[p]
                prange = opt._get_model_param_range_map(mp)
                param_range = prange["param"]
                world_range = prange["gbuf_world_in_bucket"]
                assert param_range.size == p.nelement() == world_range.size
                g = mp.main_grad.view(-1)[param_range.start : param_range.end]
                gbuf_index, _, bucket_id = opt.model_param_gbuf_map[mp]
                pbuf = opt.buffers[gbuf_index].buckets[bucket_id].param_data
                pv = pbuf.view(-1)[world_range.start : world_range.end]
                assert g.dtype == torch.bfloat16 and pv.dtype == torch.bfloat16
                assert st["exp_avg"].dtype == torch.float32 == st["exp_avg_sq"].dtype
                assert p.dtype == torch.float32 and p.is_contiguous()
                if p.numel() == 0:
                    continue
                g_l.append(g)
                p_l.append(pv)
                m_l.append(st["exp_avg"])
                v_l.append(st["exp_avg_sq"])
                w_l.append(p.data)
            lr_t = torch.zeros(1, dtype=torch.float32, device=dev)
            step_t = torch.zeros(1, dtype=torch.int32, device=dev)
            self.groups.append((group, [g_l, p_l, m_l, v_l, w_l], lr_t, step_t))
        from megatron.core import tensor_parallel
        from megatron.core.transformer.module import param_is_not_shared

        for group in adam.param_groups:
            for p in group["params"]:
                if p.numel() == 0:
                    continue
                if not param_is_not_shared(p):
                    continue
                if not tensor_parallel.param_is_not_tensor_parallel_duplicate(
                    p,
                    tp_group=getattr(opt, "tp_group", None),
                    expert_tp_group=getattr(opt, "expert_tp_group", None),
                ):
                    continue
                if not tensor_parallel.param_is_not_gtp_duplicate(p):
                    continue
                mp = master_to_model[id(p)]
                prange = opt._get_model_param_range_map(mp)["param"]
                self.norm_grads.append(mp.main_grad.view(-1)[prange.start : prange.end])
        self.noop = torch.zeros(1, dtype=torch.int32, device=dev)
        self.chunk_size = 2048 * 32
        from . import fused_adam as fa

        self.plans = [fa.AdamPlan(*[lists[i] for i in (0, 1, 4, 2, 3)]) for _, lists, _, _ in self.groups]

    def adam_step(self, coeff):
        adam = self.adam
        for (group, lists, lr_t, step_t), plan in zip(self.groups, self.plans):
            group["step"] = group.get("step", 0) + 1
            if len(lists[0]) == 0:
                continue
            beta1, beta2 = group["betas"]
            if _KERNEL != "te":
                plan.step(coeff, group["lr"], beta1, beta2, group["eps"], group["weight_decay"],
                          group["step"], group["bias_correction"], adam.adam_w_mode)
                continue
            lr_t.fill_(group["lr"])
            step_t.fill_(group["step"])
            n = len(lists[0])
            for s in range(0, n, 1024):
                chunk = [l[s : s + 1024] for l in lists]
                self.tex.multi_tensor_adam_capturable_master(
                    self.chunk_size,
                    self.noop,
                    chunk,
                    lr_t,
                    beta1,
                    beta2,
                    group["eps"],
                    step_t,
                    adam.adam_w_mode,
                    1 if group["bias_correction"] else 0,
                    group["weight_decay"],
                    coeff,
                )


def _ineligible_reason(top, opts):
    from megatron.core.optimizer import optimizer as mopt
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    try:
        from transformer_engine.pytorch.optimizers import FusedAdam as TEFusedAdam
    except ImportError:
        return "TE FusedAdam unavailable"
    cfg = top.config
    if cfg.log_num_zeros_in_grad:
        return "log_num_zeros_in_grad"
    if math.isfinite(getattr(cfg, "grad_norm_skip_threshold", math.inf)):
        return "grad_norm_skip_threshold"
    if getattr(cfg, "overlap_param_gather_with_optimizer_step", False):
        return "overlap_param_gather_with_optimizer_step"
    if getattr(cfg, "reuse_grad_buf_for_mxfp8_param_ag", False):
        return "reuse_grad_buf_for_mxfp8_param_ag"
    if getattr(cfg, "optimizer_cuda_graph", False):
        return "optimizer_cuda_graph"
    if not opts:
        return "no optimizers"
    if not cfg.clip_grad > 0.0:
        return "clip_grad <= 0"
    clip = None
    for o in opts:
        if not isinstance(o, DistributedOptimizer):
            return f"{type(o).__name__} is not DistributedOptimizer"
        if getattr(o, "is_stub_optimizer", False):
            return "stub optimizer"
        c = o.config
        if c.use_precision_aware_optimizer or c.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return "precision-aware optimizer"
        if c.optimizer != "adam" or not isinstance(o.optimizer, TEFusedAdam):
            return f"inner optimizer {type(o.optimizer).__name__}"
        a = o.optimizer
        if a.capturable or a.master_weights:
            return "FusedAdam capturable/master_weights already set"
        if not (a.exp_avg_dtype == a.exp_avg_sq_dtype == torch.float32):
            return "non-fp32 moments"
        if o.grad_scaler is not None:
            return "grad scaler"
        dc = o.ddp_config
        if dc.use_megatron_fsdp or dc.fp8_param_gather or getattr(dc, "fp4_param_gather", False):
            return "fsdp / quantized param gather"
        if any(len(g) for g in o.model_fp32_groups):
            return "fp32 model params"
        for mg in o.model_float16_groups:
            for mp in mg:
                if mp.dtype != torch.bfloat16 or getattr(mp, "main_grad", None) is None:
                    return "non-bf16 model param or no main_grad"
                if mp.main_grad.dtype != torch.bfloat16:
                    return "main_grad not bf16 (grad_reduce_in_fp32)"
        if clip is None:
            clip = c.clip_grad
        elif clip != c.clip_grad:
            return "mixed clip_grad"
    if len(opts) > 1 and not top.grads_states_parallel_group_is_shared():
        return "grad-stats groups not shared"
    return None


def _fused_step(top, opts):
    from . import grad_clip_norm as gcn

    fs = getattr(top, "_opt_fused", None)
    if fs is None:
        states = [_FusedState(o) for o in opts]
        grads = [g for s in states for g in s.norm_grads]
        plan = gcn.GradNormPlan(grads)
        fs = (states, plan)
        top._opt_fused = fs
        _log_once(
            "active",
            f"active: {len(opts)} DistributedOptimizer(s), "
            f"{sum(len(s.groups) for s in states)} param group(s), "
            f"{sum(len(l[0]) for s in states for _, l, _, _ in s.groups)} shard tensors, "
            f"norm over {plan.numel} bf16 elements in {plan.offsets.numel()} chunks "
            f"(aligned={plan.aligned}); adam kernel={_KERNEL} "
            f"aligned={[p.aligned for s in states for p in s.plans if not p.empty]}",
        )
    states, plan = fs
    clip = opts[0].config.clip_grad
    group = opts[0].get_grad_stats_parallel_group()
    sumsq = plan.sumsq_()
    torch.distributed.all_reduce(sumsq, op=torch.distributed.ReduceOp.SUM, group=group)
    nc = gcn.norm_and_coeff(sumsq, clip if clip > 0.0 else math.inf)
    grad_norm = nc[0:1]
    coeff = nc[1:2]
    for s in states:
        s.adam_step(coeff)
    for o in opts:
        should_sync = not o.ddp_config.overlap_param_gather and not getattr(
            o, "_defer_param_sync", False
        )
        if should_sync:
            o.start_param_sync_for_bucket_group_subset()
    return grad_norm


def _check(top, opts):
    ok = getattr(top, "_opt_fused_ok", None)
    if ok is None:
        from megatron.core.optimizer import optimizer as mopt

        separate = [g for g in mopt.SEPARATE_GRAD_NORM_GROUPS if top.has_grad_norm_group(g)]
        reason = _ineligible_reason(top, opts)
        if reason is None and separate:
            reason = f"separate grad-norm groups {separate}"
        flag = torch.tensor([0 if reason is None else 1], dtype=torch.int32, device="cuda")
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        ok = int(flag.item()) == 0
        top._opt_fused_ok = ok
        if not ok:
            _log_once(
                "fallback",
                f"fallback to Megatron's optimizer step (rank-0 reason: {reason}; "
                "some rank was ineligible)",
            )
    return ok


def install():
    if not _ENABLED:
        return
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer, MixedPrecisionOptimizer

    orig_chained_step = ChainedOptimizer.step
    orig_mp_step = MixedPrecisionOptimizer.step

    @torch.no_grad()
    def chained_step(self):
        opts = list(self.chained_optimizers)
        if not _check(self, opts):
            return orig_chained_step(self)
        self.grad_norms_by_group = {}
        return True, _fused_step(self, opts), None

    @torch.no_grad()
    def mp_step(self):
        if not isinstance(self, DistributedOptimizer) or not _check(self, [self]):
            return orig_mp_step(self)
        self.grad_norms_by_group = {}
        return True, _fused_step(self, [self]), 0

    ChainedOptimizer.step = chained_step
    MixedPrecisionOptimizer.step = mp_step
    _log_once("installed", "installed (OPT_FUSED_ADAM=0 disables)")


install()
