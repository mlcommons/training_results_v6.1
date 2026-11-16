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
# Re-implements the Megatron-Core DistributedDataParallel / DistributedOptimizer data path
# (gradient reduce-scatter, parameter all-gather, optimizer step) as run-owned code; the
# grad_added_to_main_grad reset follows Megatron-Core's distributed_data_parallel.py
# (Apache-2.0, Copyright (c) NVIDIA CORPORATION & AFFILIATES).


import math
import os

import torch
import triton
import triton.language as tl
from torch.distributed.distributed_c10d import _coalescing_manager

_ENABLED = os.environ.get("OPT_OWN_DP", "1") != "0"
_LOOKAHEAD = int(os.environ.get("OPT_OWN_DP_LOOKAHEAD", "2"))
_DEBUG = os.environ.get("OPT_OWN_DP_DEBUG", "0") == "1"
_LOGGED = set()

PREFETCH = 2
_PREFETCHED = object()

CHUNK = 16384
_BLOCK = 512
_HP = 9


_CKPT_REASON = [None]


def note_checkpoint_config(ckpt):
    reasons = []
    if getattr(ckpt, "save", None):
        reasons.append(f"checkpoint save ({ckpt.save})")
    if getattr(ckpt, "load", None):
        reasons.append(f"checkpoint load ({ckpt.load})")
    _CKPT_REASON[0] = " / ".join(reasons) + " configured: the engine has no checkpoint state" if reasons else None


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _log_once(key, msg):
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    if _rank() == 0:
        print(f"[opt dp_engine] {msg}", flush=True)


def _f32(x):
    return float(torch.tensor(x, dtype=torch.float32).item())


@triton.jit
def _adam_dev_kernel(
    g_ptr, p_ptr, w_ptr, m_ptr, v_ptr, tab_ptr, hp_ptr, coeff_ptr, flag_ptr, n_chunks,
    ADAMW: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr, ALIGNED: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    active = tl.load(flag_ptr) != 0
    coeff = tl.load(coeff_ptr)
    for c in tl.range(pid, n_chunks, nprog):
        row = tab_ptr + c * 7
        og = tl.load(row + 0)
        op = tl.load(row + 1)
        ow = tl.load(row + 2)
        om = tl.load(row + 3)
        ov = tl.load(row + 4)
        n = tl.where(active, tl.load(row + 5), 0)
        gi = tl.load(row + 6)
        if ALIGNED:
            og = tl.multiple_of(og, 8)
            op = tl.multiple_of(op, 8)
            ow = tl.multiple_of(ow, 4)
            om = tl.multiple_of(om, 4)
            ov = tl.multiple_of(ov, 4)
        h = hp_ptr + gi * 9
        lr = tl.load(h + 0)
        beta1 = tl.load(h + 1)
        beta2 = tl.load(h + 2)
        omb1 = tl.load(h + 3)
        omb2 = tl.load(h + 4)
        bc1 = tl.load(h + 5)
        bc2 = tl.load(h + 6)
        eps = tl.load(h + 7)
        wd = tl.load(h + 8)
        for s in tl.range(0, CHUNK, BLOCK):
            idx = s + tl.arange(0, BLOCK)
            mask = idx < n
            g = tl.load(g_ptr + og + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
            w = tl.load(w_ptr + ow + idx, mask=mask, other=0.0, eviction_policy="evict_first")
            m = tl.load(m_ptr + om + idx, mask=mask, other=0.0, eviction_policy="evict_first")
            v = tl.load(v_ptr + ov + idx, mask=mask, other=0.0, eviction_policy="evict_first")
            g = g * coeff
            if ADAMW:
                m = beta1 * m + omb1 * g
                v = beta2 * v + omb2 * g * g
                m_hat = tl.div_rn(m, bc1)
                v_hat = tl.div_rn(v, bc2)
                denom = tl.sqrt_rn(v_hat) + eps
                update = tl.div_rn(m_hat, denom) + wd * w
            else:
                g = g + wd * w
                m = beta1 * m + omb1 * g
                v = beta2 * v + omb2 * g * g
                m_hat = tl.div_rn(m, bc1)
                v_hat = tl.div_rn(v, bc2)
                denom = tl.sqrt_rn(v_hat) + eps
                update = tl.div_rn(m_hat, denom)
            w = w - lr * update
            pb = w.to(tl.bfloat16)
            tl.store(w_ptr + ow + idx, w, mask=mask, eviction_policy="evict_first")
            tl.store(m_ptr + om + idx, m, mask=mask, eviction_policy="evict_first")
            tl.store(v_ptr + ov + idx, v, mask=mask, eviction_policy="evict_first")
            tl.store(p_ptr + op + idx, pb, mask=mask, eviction_policy="evict_first")


@torch.library.custom_op(
    "optimized::adam_dev_step", mutates_args=("p_base", "w_base", "m_base", "v_base"),
    device_types="cuda",
)
def adam_dev_step(
    g_base: torch.Tensor, p_base: torch.Tensor, w_base: torch.Tensor, m_base: torch.Tensor,
    v_base: torch.Tensor, table: torch.Tensor, hp: torch.Tensor, coeff: torch.Tensor,
    flag: torch.Tensor, aligned: bool, adam_w_mode: bool,
) -> None:
    n_chunks = table.shape[0]
    if n_chunks == 0:
        return
    _adam_dev_kernel[(n_chunks,)](
        g_base, p_base, w_base, m_base, v_base, table, hp, coeff, flag, n_chunks,
        ADAMW=adam_w_mode, CHUNK=CHUNK, BLOCK=_BLOCK, ALIGNED=aligned, num_warps=4,
    )


class _AdamPlan:

    def __init__(self, bases, items):
        self.bases = bases
        esz = (2, 2, 4, 4, 4)
        rows = []
        aligned = True
        for it in items:
            ts, gi = it[:5], it[5]
            n = ts[0].numel()
            offs = []
            for t, base, sz in zip(ts, bases, esz):
                assert t.is_contiguous() and t.numel() == n
                d = t.data_ptr() - base.data_ptr()
                assert d % sz == 0
                if t.data_ptr() % 16 != 0:
                    aligned = False
                offs.append(d // sz)
            for c in range(0, n, CHUNK):
                rows.append([o + c for o in offs] + [min(CHUNK, n - c), gi])
        self.aligned = aligned
        dev = bases[0].device
        self.table = torch.tensor(rows, dtype=torch.int64, device=dev).reshape(-1, 7)

    def run(self, hp, coeff, flag, adam_w_mode):
        adam_dev_step(*self.bases, self.table, hp, coeff, flag, self.aligned, bool(adam_w_mode))


class _Unit:
    __slots__ = ("name", "params", "module", "adam", "norm", "ag", "work", "remaining", "slot")

    def __init__(self, name, params, module):
        self.name = name
        self.params = params
        self.module = module
        self.adam = None
        self.norm = None
        self.ag = []
        self.work = None
        self.remaining = len(params)
        self.slot = None


class DPEngine:
    def __init__(self, ddp):
        self.ddp = ddp
        self.group = ddp.intra_dp_cp_group
        self.world = self.group.size()
        self.rank = self.group.rank()
        self.active = False
        self.units = []
        self.unit_of = {}
        self.shard = {}
        self.overwrite_ptrs = set()
        self.emb_param = None
        self.emb_prev = None
        self.emb_shard = None
        self.issued = 0
        self.first_backward_checked = False
        self._captured_works = []
        self.pf_group = None
        self.pf_ev = None
        self.n_pf = 0
        self.pf_pending = False
        self.pf_applied = False
        self.graph_pf = None
        self.pf_graph = None
        self.pf_eager_done = False

    def _ineligible_reason_ddp(self):
        ddp = self.ddp
        dc = ddp.ddp_config
        if _CKPT_REASON[0] is not None:
            return _CKPT_REASON[0]
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

            nmb = get_num_microbatches()
        except Exception:
            nmb = None
        if nmb is not None and nmb != 1:
            return f"{nmb} micro-batches per step (the engine reduces and updates once per backward)"
        if not dc.use_distributed_optimizer:
            return "no distributed optimizer"
        if dc.use_megatron_fsdp or dc.num_distributed_optimizer_instances != 1:
            return "fsdp / multiple optimizer instances"
        if dc.grad_reduce_in_fp32 or dc.average_in_collective:
            return "fp32 grad reduce / average in collective"
        if len(ddp.expert_parallel_buffers) or len(ddp.buffers) != 1:
            return "expert buffers / several grad buffers"
        if ddp.tp_group.size() != 1 or ddp.pp_group.size() != 1:
            return "tensor/pipeline parallel"
        if ddp.dp_cp_group.size() != self.world or ddp.dp_group.size() != self.world:
            return "context parallel"
        if self.world < 2:
            return "single rank"
        for p in ddp.params_with_grad:
            if p.dtype != torch.bfloat16 or p.main_grad.dtype != torch.bfloat16:
                return "non-bf16 param or grad"
            if p.numel() % self.world:
                return f"param numel {p.numel()} not divisible by {self.world}"
            if not p.is_contiguous() or not p.main_grad.is_contiguous():
                return "non-contiguous param"
        self.scale = ddp.buffers[0].gradient_scaling_factor
        if abs(self.scale * self.world - 1.0) > 0 or self.scale <= 0:
            return f"gradient scaling factor {self.scale}"
        return None

    def _build_units(self):
        ddp = self.ddp
        names = {p: n for n, p in ddp.module.named_parameters()}
        mods = dict(ddp.module.named_modules())
        emb, layers, final = [], {}, []
        for p in ddp.params_with_grad:
            n = names[p]
            parts = n.split(".")
            if "embedding" in parts:
                emb.append(p)
            elif "layers" in parts and parts[parts.index("layers") - 1] == "decoder":
                i = int(parts[parts.index("layers") + 1])
                layers.setdefault(i, []).append(p)
            elif "final_layernorm" in parts or "output_layer" in parts:
                final.append(p)
            else:
                raise RuntimeError(f"dp_engine: no unit for parameter {n}")

        def find(suffix):
            hits = [m for k, m in mods.items() if k == suffix or k.endswith("." + suffix)]
            if len(hits) != 1:
                raise RuntimeError(f"dp_engine: module {suffix}: {len(hits)} matches")
            return hits[0]

        units = []
        if emb:
            assert len(emb) == 1
            units.append(_Unit("embedding", emb, find("embedding")))
        for i in sorted(layers):
            units.append(_Unit(f"layer{i}", layers[i], find(f"decoder.layers.{i}")))
        if final:
            units.append(_Unit("final", final, find("decoder.final_layernorm")))
        self.units = units
        for u in units:
            for p in u.params:
                self.unit_of[p] = u
        self.emb_param = emb[0] if emb else None

    def _classify_overwrite(self):
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch.ops import BasicLinear

        te_mod_types = tuple(
            t for t in (getattr(te, "Linear", None), getattr(te, "LayerNormLinear", None),
                        getattr(te, "LayerNormMLP", None), getattr(te, "GroupedLinear", None))
            if t is not None
        )
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

        op_owned = set()
        for m in self.ddp.module.modules():
            if type(m).__name__ == "TEFusedMLP" or isinstance(m, BasicLinear):
                op_owned.update(id(p) for p in m.parameters())
        self.fused_kind = {}
        for m in self.ddp.module.modules():
            w = m._parameters.get("weight") if hasattr(m, "_parameters") else None
            if w is None or not w.requires_grad:
                continue
            if id(w) in op_owned:
                self.overwrite_ptrs.add(w.main_grad.data_ptr())
                self.fused_kind[w] = "te_op"
            elif isinstance(m, te_mod_types):
                w.overwrite_main_grad = True
                self.fused_kind[w] = "te_module"
            elif isinstance(m, (ColumnParallelLinear, RowParallelLinear)):
                self.overwrite_ptrs.add(w.main_grad.data_ptr())
                self.fused_kind[w] = "mcore_linear"
        kinds = {}
        for k in self.fused_kind.values():
            kinds[k] = kinds.get(k, 0) + 1
        self.kind_counts = kinds

    def attach_optimizer(self, opt, group_of):
        if getattr(self, "opt", None) is not None:
            raise RuntimeError("dp_engine: more than one DistributedOptimizer is not supported")
        reason = self._ineligible_reason_ddp()
        if reason is None:
            from .fused_adam_patch import _ineligible_reason

            reason = _ineligible_reason(opt, [opt])
        if reason is None:
            try:
                self._build_units()
            except RuntimeError as e:
                reason = str(e)
        flag = torch.tensor([0 if reason is None else 1], dtype=torch.int32, device="cuda")
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        if int(flag.item()) != 0:
            _log_once("inactive", f"inactive, Megatron DDP / optimizer path kept (rank-0 reason: {reason})")
            return
        self.opt = opt
        self.pf_group = torch.distributed.new_group(
            ranks=torch.distributed.get_process_group_ranks(self.group), backend="nccl")
        torch.distributed.all_reduce(torch.zeros(1, device="cuda"), group=self.pf_group)
        self._build_state(opt, group_of)
        self._classify_overwrite()
        self._install_hooks()
        self.active = True
        nel = sum(p.numel() for p in self.ddp.params_with_grad)
        _log_once(
            "active",
            f"active: DP={self.world}, {len(self.units)} units, {len(self.ddp.params_with_grad)} "
            f"tensors, {nel} params ({nel // self.world} per shard), lookahead {_LOOKAHEAD}, "
            f"adam aligned={[u.adam.aligned for u in self.units]}, "
            f"overwrite kinds={self.kind_counts}",
        )

    def _build_state(self, opt, group_of):
        dev = torch.device("cuda", torch.cuda.current_device())
        adam = opt.optimizer
        self.adam = adam
        self.adam_w_mode = adam.adam_w_mode
        self.groups = list(adam.param_groups)
        assert all(p in group_of for p in self.ddp.params_with_grad)
        if len(opt.model_fp32_groups) and any(len(g) for g in opt.model_fp32_groups):
            raise RuntimeError("dp_engine: fp32 model params")
        W, r = self.world, self.rank
        total = 0
        for u in self.units:
            for p in u.params:
                total += p.numel() // W
        self.w_buf = torch.empty(total, dtype=torch.float32, device=dev)
        self.m_buf = torch.zeros(total, dtype=torch.float32, device=dev)
        self.v_buf = torch.zeros(total, dtype=torch.float32, device=dev)
        gbase = self.ddp.buffers[0].grad_data
        pbase = self.ddp.buffers[0].param_data
        bases = (gbase, pbase, self.w_buf, self.m_buf, self.v_buf)
        self.slots = torch.zeros(len(self.units), dtype=torch.float32, device=dev)
        from . import grad_clip_norm as gcn

        off = 0
        for ui, u in enumerate(self.units):
            items = []
            gshards = []
            for p in u.params:
                n = p.numel()
                k = n // W
                gs = p.main_grad.view(-1)[r * k:(r + 1) * k]
                ps = p.data.view(-1)[r * k:(r + 1) * k]
                w = self.w_buf[off:off + k]
                w.copy_(ps.float())
                items.append((gs, ps, w, self.m_buf[off:off + k], self.v_buf[off:off + k],
                              group_of[p]))
                self.shard[p] = (gs, ps)
                u.ag.append((p, p.data.view(-1), ps))
                gshards.append(gs)
                off += k
            u.adam = _AdamPlan(bases, items)
            u.norm = gcn.GradNormPlan(gshards)
            u.norm.sumsq = self.slots[ui:ui + 1]
            u.slot = ui
        assert off == total
        if self.emb_param is not None:
            self.emb_shard = self.shard[self.emb_param][0]
        self.hp = torch.zeros(len(self.groups), _HP, dtype=torch.float32, device=dev)
        self.flag = torch.zeros(1, dtype=torch.int32, device=dev)
        self.sumsq = torch.zeros(1, dtype=torch.float32, device=dev)
        self.nc = torch.zeros(2, dtype=torch.float32, device=dev)
        self.nc[1] = 1.0
        self.clip = opt.config.clip_grad
        self.stats_group = opt.get_grad_stats_parallel_group()
        self.s_opt = torch.cuda.Stream(device=dev)
        self.s_red = torch.cuda.Stream(device=dev)
        self.s_ag = torch.cuda.Stream(device=dev)
        self.n_pf = min(PREFETCH, len(self.units) - 1)
        self.pf_ev = torch.cuda.Event(external=True)
        self.pending_host = False
        for groups in (opt.shard_fp32_from_float16_groups,):
            for g in groups:
                for mp in g:
                    mp.data = torch.empty(0, dtype=mp.dtype, device=mp.device)

    def _install_hooks(self):
        for ui, u in enumerate(self.units):
            u.module.register_forward_pre_hook(self._make_pre_hook(ui))
        if self.emb_param is not None:
            self.emb_param._opt_dp_emb = self

    def _issue(self, j, after_event=None, group=None):
        if j >= len(self.units):
            return
        u = self.units[j]
        with torch.cuda.stream(self.s_opt):
            if after_event is not None:
                self.s_opt.wait_event(after_event)
            u.adam.run(self.hp, self.nc[1:2], self.flag, self.adam_w_mode)
            if j == len(self.units) - 1:
                self.flag.zero_()
        with torch.cuda.stream(self.s_ag):
            self.s_ag.wait_stream(self.s_opt)
            self._gather_unit(u, group=group)
            ev = torch.cuda.Event()
            ev.record(self.s_ag)
        u.work = ev

    def _make_pre_hook(self, ui):
        def hook(module, *unused):
            if not self.active or not torch.is_grad_enabled():
                return
            cur = torch.cuda.current_stream()
            if ui == 0:
                capturing = torch.cuda.is_current_stream_capturing()
                if self.pf_pending:
                    cur.wait_event(self.pf_ev)
                    self.pf_pending = False
                    self.issued = self.n_pf
                else:
                    self.s_opt.wait_stream(cur)
                    self.s_ag.wait_stream(cur)
                    n = min(_LOOKAHEAD, len(self.units))
                    for j in range(n):
                        self._issue(j)
                    self.issued = n
                if capturing:
                    self.graph_pf = self.issued == self.n_pf and self.units[0].work is _PREFETCHED
                    _log_once("graph_pf", f"training graph captured with first-unit prefetch "
                              f"{'on' if self.graph_pf else 'off'} ({self.n_pf} units)")
            u = self.units[ui]
            assert u.work is not None, f"dp_engine: unit {u.name} all-gather not issued"
            if u.work is not _PREFETCHED:
                cur.wait_event(u.work)
            u.work = None
            if self.issued < len(self.units):
                ev = torch.cuda.Event()
                ev.record(cur)
                self._issue(self.issued, after_event=ev)
                self.issued += 1

        return hook

    def _gather_unit(self, u, group=None):
        group = self.group if group is None else group
        with _coalescing_manager(group, async_ops=True) as cm:
            for p, full, shard in u.ag:
                torch.distributed.all_gather_into_tensor(full, shard, group=group,
                                                         async_op=True)
        cm.wait()
        self._keep(cm)

    def flush(self):
        if not self.active:
            return
        cur = torch.cuda.current_stream()
        cur.wait_stream(self.s_opt)
        cur.wait_stream(self.s_ag)
        done = self.n_pf if self.pf_applied else 0
        with torch.no_grad():
            for i, u in enumerate(self.units):
                if i >= done:
                    u.adam.run(self.hp, self.nc[1:2], self.flag, self.adam_w_mode)
                self._gather_unit(u)
            self.flag.zero_()
        self.pending_host = False
        self.pf_applied = False

    def step(self):
        rows = []
        for group in self.groups:
            group["step"] = group.get("step", 0) + 1
            b1, b2 = group["betas"]
            b1, b2 = _f32(b1), _f32(b2)
            if group["bias_correction"]:
                st = group["step"]
                bc1, bc2 = _f32(1.0 - b1 ** st), _f32(1.0 - b2 ** st)
            else:
                bc1 = bc2 = 1.0
            rows.append([_f32(group["lr"]), b1, b2, _f32(1.0 - b1), _f32(1.0 - b2), bc1, bc2,
                         _f32(group["eps"]), _f32(group["weight_decay"])])
        self.hp.copy_(torch.tensor(rows, dtype=torch.float32))
        self.flag.fill_(1)
        self.pending_host = True
        if self.graph_pf is not False:
            self._prefetch()
        norm = self.nc[0:1].clone()
        if _DEBUG and _rank() == 0:
            print(f"[opt dp_engine] debug: grad_norm {norm.item():.6e} coeff "
                  f"{self.nc[1].item():.6e} lr {rows[0][0]:.6e} step {self.groups[0]['step']}",
                  flush=True)
        return norm

    def _prefetch(self):
        cur = torch.cuda.current_stream()
        if self.pf_graph is None and self.pf_eager_done:
            self._capture_prefetch()
        if self.pf_graph is not None:
            self.s_ag.wait_stream(cur)
            with torch.cuda.stream(self.s_ag):
                self.pf_graph.replay()
        else:
            self.s_opt.wait_stream(cur)
            self.s_ag.wait_stream(cur)
            self._issue_prefetch()
            self.pf_eager_done = True
        for j in range(self.n_pf):
            self.units[j].work = _PREFETCHED
        self.pf_ev.record(self.s_ag)
        self.pf_pending = True
        self.pf_applied = True

    def _issue_prefetch(self):
        for j in range(self.n_pf):
            self._issue(j, group=self.pf_group)

    def _capture_prefetch(self):
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        cs = torch.cuda.Stream()
        with torch.cuda.graph(g, stream=cs, capture_error_mode="thread_local"):
            c = torch.cuda.current_stream()
            self.s_opt.wait_stream(c)
            self.s_ag.wait_stream(c)
            self._issue_prefetch()
            c.wait_stream(self.s_opt)
            c.wait_stream(self.s_ag)
        torch.cuda.synchronize()
        self.pf_graph = g
        _log_once("pf_graph", f"first-unit prefetch captured ({self.n_pf} units)")

    def release_graphs(self):
        if self.pf_graph is not None:
            torch.cuda.synchronize()
            self.pf_graph = None
            _log_once("pf_release", "first-unit prefetch graph released")

    def grad_hook(self, param):
        if param is self.emb_param:
            param.grad = None
            return
        if param.grad is not None and not param.grad_added_to_main_grad:
            if param.grad.data_ptr() != param.main_grad.data_ptr():
                param.main_grad.copy_(param.grad.data)
        if not self.first_backward_checked and param.grad_added_to_main_grad:
            if param not in self.fused_kind:
                raise RuntimeError("dp_engine: a parameter accumulated into main_grad through "
                                   "an unhandled path (would read a stale buffer)")
        param.grad = None
        self.grad_ready(param)

    def grad_ready(self, param):
        u = self.unit_of[param]
        gs = self.shard[param][0]
        cur = torch.cuda.current_stream()
        with torch.cuda.stream(self.s_red):
            self.s_red.wait_stream(cur)
            work = torch.distributed.reduce_scatter_tensor(
                gs, param.main_grad.view(-1), group=self.group, async_op=True
            )
            work.wait()
            self._keep(work)
            u.remaining -= 1
            if u.remaining == 0:
                torch._foreach_mul_([self.shard[p][0] for p in u.params], self.scale)
                u.norm.sumsq_()

    def embedding_grad_buffer(self, weight, idx):
        mg = weight.main_grad
        flat = idx.view(-1)
        if self.emb_prev is None:
            self.emb_prev = torch.zeros(flat.numel(), dtype=torch.int64, device=flat.device)
        assert self.emb_prev.numel() == flat.numel()
        mg.index_fill_(0, self.emb_prev, 0)
        self.emb_shard.zero_()
        return mg

    def embedding_grad_ready(self, weight, idx):
        self.emb_prev.copy_(idx.view(-1))
        self.grad_ready(weight)

    def finish_grad_sync(self):
        if not self.first_backward_checked:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

            if get_num_microbatches() != 1:
                raise RuntimeError("dp_engine: needs one microbatch per step (attach_optimizer's eligibility "
                                   "check should have kept the engine inactive)")
            self.first_backward_checked = True
        left = [u.name for u in self.units if u.remaining != 0]
        if left:
            raise RuntimeError(f"dp_engine: grads never became ready for units {left}")
        cur = torch.cuda.current_stream()
        cur.wait_stream(self.s_red)
        cur.wait_stream(self.s_opt)
        cur.wait_stream(self.s_ag)
        torch.sum(self.slots, dim=0, keepdim=True, out=self.sumsq)
        work = torch.distributed.all_reduce(self.sumsq, group=self.stats_group, async_op=True)
        work.wait()
        self._keep(work)
        from .grad_clip_norm import clip_coeff

        clip_coeff(self.sumsq, self.nc, float(self.clip if self.clip > 0.0 else math.inf))
        for u in self.units:
            u.remaining = len(u.params)

    def _keep(self, work):
        if torch.cuda.is_current_stream_capturing():
            self._captured_works.append(work)

    def reset_for_step(self):
        for u in self.units:
            u.remaining = len(u.params)
        self.issued = 0


_ENGINES = {}


def _engine_of(ddp):
    return _ENGINES.get(id(ddp))


def _single_engine():
    if len(_ENGINES) != 1:
        return None
    e = next(iter(_ENGINES.values()))
    return e if e.active else None


def install():
    if not _ENABLED:
        _log_once("off", "disabled (OPT_OWN_DP=0): Megatron DDP / DistributedOptimizer path")
        return
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel as DDP
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer, MixedPrecisionOptimizer
    from megatron.core.tensor_parallel import layers as mcore_layers
    from transformer_engine.pytorch.ops import BasicLinear

    orig_init = DDP.__init__
    orig_bwd_hook = DDP._make_backward_post_hook
    orig_fwd_hook = DDP._make_forward_pre_hook
    orig_enable = DDP.enable_forward_pre_hook
    orig_disable = DDP.disable_forward_pre_hook
    orig_start_param_sync = DDP.start_param_sync
    orig_start_grad_sync = DDP.start_grad_sync
    orig_finish_grad_sync = DDP.finish_grad_sync
    orig_zero = DDP.zero_grad_buffer

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        _ENGINES[id(self)] = DPEngine(self)

    def bwd_hook(self, param):
        orig = orig_bwd_hook(self, param)

        def hook(*unused):
            e = _engine_of(self)
            if e is not None and e.active:
                return e.grad_hook(param)
            return orig(*unused)

        return hook

    def fwd_hook(self):
        orig = orig_fwd_hook(self)

        def hook(module, *unused):
            e = _engine_of(self)
            if e is not None and e.active:
                return None
            return orig(module, *unused)

        return hook

    def enable(self):
        e = _engine_of(self)
        if e is not None and e.active:
            return
        return orig_enable(self)

    def disable(self, param_sync=True):
        e = _engine_of(self)
        if e is not None and e.active:
            if param_sync:
                e.flush()
            return
        return orig_disable(self, param_sync=param_sync)

    def start_param_sync(self, *unused, force_sync=False, force_dispatch=False):
        e = _engine_of(self)
        if e is not None and e.active:
            if force_sync:
                e.flush()
            return
        return orig_start_param_sync(self, *unused, force_sync=force_sync,
                                     force_dispatch=force_dispatch)

    def start_grad_sync(self, *unused):
        e = _engine_of(self)
        if e is not None and e.active:
            return
        return orig_start_grad_sync(self, *unused)

    def finish_grad_sync(self, force_all_reduce=False):
        e = _engine_of(self)
        if e is not None and e.active:
            return e.finish_grad_sync()
        return orig_finish_grad_sync(self, force_all_reduce=force_all_reduce)

    def zero_grad_buffer(self):
        e = _engine_of(self)
        if e is not None and e.active:
            if getattr(self.config, "cuda_graph_impl", "none") != "transformer_engine":
                for param in self.params_with_grad:
                    param.grad_added_to_main_grad = False
            e.reset_for_step()
            return
        return orig_zero(self)

    DDP.__init__ = init
    DDP._make_backward_post_hook = bwd_hook
    DDP._make_forward_pre_hook = fwd_hook
    DDP.enable_forward_pre_hook = enable
    DDP.disable_forward_pre_hook = disable
    DDP.start_param_sync = start_param_sync
    DDP.start_grad_sync = start_grad_sync
    DDP.finish_grad_sync = finish_grad_sync
    DDP.zero_grad_buffer = zero_grad_buffer

    orig_do_init = DistributedOptimizer.__init__

    def do_init(self, *a, **kw):
        inner = a[0] if a else kw.get("optimizer")
        group_of = {}
        if inner is not None:
            for gi, g in enumerate(inner.param_groups):
                for p in g["params"]:
                    group_of[p] = gi
        orig_do_init(self, *a, **kw)
        engines = [e for e in _ENGINES.values()]
        if len(engines) == 1 and not getattr(self, "is_stub_optimizer", False):
            engines[0].attach_optimizer(self, group_of)

    DistributedOptimizer.__init__ = do_init

    prev_chained_step = ChainedOptimizer.step
    prev_mp_step = MixedPrecisionOptimizer.step

    @torch.no_grad()
    def chained_step(self):
        e = _single_engine()
        opts = list(self.chained_optimizers)
        if e is not None and len(opts) == 1 and opts[0] is getattr(e, "opt", None):
            self.grad_norms_by_group = {}
            return True, e.step(), None
        return prev_chained_step(self)

    @torch.no_grad()
    def mp_step(self):
        e = _single_engine()
        if e is not None and self is getattr(e, "opt", None):
            self.grad_norms_by_group = {}
            return True, e.step(), 0
        return prev_mp_step(self)

    ChainedOptimizer.step = chained_step
    MixedPrecisionOptimizer.step = mp_step

    orig_fb = BasicLinear._functional_backward

    def functional_backward(*a, **kw):
        e = _single_engine()
        gw = kw.get("grad_weight")
        if (e is not None and gw is not None and kw.get("accumulate_into_grad_weight")
                and gw.data_ptr() in e.overwrite_ptrs):
            kw["accumulate_into_grad_weight"] = False
        return orig_fb(*a, **kw)

    BasicLinear._functional_backward = staticmethod(functional_backward)

    fwmc = mcore_layers.fused_weight_gradient_mlp_cuda

    class _WgradOverwrite:
        def __getattr__(self, name):
            return getattr(fwmc, name)

        @staticmethod
        def wgrad_gemm_accum_fp16(total_input, grad_output, main_grad):
            e = _single_engine()
            if e is not None and main_grad.data_ptr() in e.overwrite_ptrs:
                return mcore_layers._wgrad_gemm(main_grad, grad_output, total_input)
            return fwmc.wgrad_gemm_accum_fp16(total_input, grad_output, main_grad)

    mcore_layers.fused_weight_gradient_mlp_cuda = _WgradOverwrite()

    try:
        from megatron.bridge.training import eval as eval_module

        orig_eval = eval_module.evaluate

        def evaluate(*a, **kw):
            e = _single_engine()
            if e is not None:
                e.flush()
            return orig_eval(*a, **kw)

        eval_module.evaluate = evaluate
    except ImportError:
        pass

    try:
        from megatron.bridge.training import train as bridge_train

        orig_delete = bridge_train._delete_cuda_graphs

        def delete_cuda_graphs(*a, **kw):
            for e in _ENGINES.values():
                e.release_graphs()
            return orig_delete(*a, **kw)

        bridge_train._delete_cuda_graphs = delete_cuda_graphs
    except (ImportError, AttributeError):
        pass

    _log_once("installed", "installed (OPT_OWN_DP=0 disables)")


def _install_debug_norm_print():
    from megatron.core.optimizer.optimizer import ChainedOptimizer, MixedPrecisionOptimizer

    for cls in (ChainedOptimizer, MixedPrecisionOptimizer):
        prev = cls.step

        def make(prev):
            def step(self):
                out = prev(self)
                if _rank() == 0 and out[1] is not None:
                    gn = out[1]
                    gn = gn.item() if torch.is_tensor(gn) else gn
                    print(f"[opt dp_engine] debug: baseline grad_norm {gn:.6e}", flush=True)
                return out

            return step

        cls.step = make(prev)


install()
if _DEBUG and not _ENABLED:
    _install_debug_norm_print()
