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

"""Run the distributed optimizer per bucket on a low-priority side stream inside the full-iteration CUDA graph.

The reference steps the optimizer eagerly after the graph (grad copies, L2 norm, clip, FusedAdam, param copies, then the
param all-gathers from the next forward's pre-hooks).  Here each bucket's grad-norm partial runs on the side stream once its
reduce-scatter is drained (a single scalar all-reduce remains after the last), the clip coefficient is applied inside the
bucket's Adam as TE's `inv_scale`, optimizer.step() only snapshots lr / step into device scalars (betas / eps / wd are baked
and checked), and Adam + grad zero + param all-gather of every bucket run on the side stream in first-needed order as the
prologue of the NEXT forward_backward call.  Buckets not needed at step start are released by an event recorded when the
forward reaches a layer's core attention (yield) so their traffic lands under attention.  The MoE overflow flags are reduced
inside the graph right after the decoder forward so the host enqueues the next replay before the backward ends (overflow /
rerun semantics unchanged).  With FLAT_ADAM=1 each bucket's Adam is one flat_adam_cuda launch (bit-identical to TE).

Gates: OPT_SIDE_STREAM=0 leaves the reference path; OPT_SIDE_STREAM_YIELD / OPT_SIDE_STREAM_EARLY_CHECK default on,
OPT_SIDE_STREAM_HIPRI off (see its definition).  Anything outside the fast path keeps the original optimizer path (logged).
"""
import logging
import os
import sys

import torch
import torch.distributed as dist

_OUT = sys.__stdout__
logger = logging.getLogger(__name__)

ENABLED = os.getenv("OPT_SIDE_STREAM", "1") == "1"
YIELD = os.getenv("OPT_SIDE_STREAM_YIELD", "1") == "1"
# Off by default: with the chain captured at a higher priority than the side stream, the side stream's nodes only get
# SMs while the chain is stalled -- exactly when it waits for AG(b) -- so Adam/zero/AG of every bucket are fully exposed
# (measured +43.7 ms/step).  Equal priorities share the SMs; the yield below is done with events, not priorities.
HIPRI = os.getenv("OPT_SIDE_STREAM_HIPRI", "0") == "1"
EARLY_CHECK = os.getenv("OPT_SIDE_STREAM_EARLY_CHECK", "1") == "1"
VERBOSE = os.getenv("OPT_SIDE_STREAM_VERBOSE", "0") == "1"
# One flat Adam launch per bucket shard (flat_adam_cuda) instead of TE's per-tensor-list launches.
FLAT = os.getenv("FLAT_ADAM", "1") == "1"
FLAT_SELFCHECK = os.getenv("FLAT_ADAM_SELFCHECK", "1") == "1"
# Buckets per attention slot and how many slots before first use an Adam must be launched.
# One bucket per slot: an all-gather is 1.26 GiB of copy-engine traffic (1.6-2.2 ms) against a ~4 ms layer, so two per
# slot queue 3.2-4.4 ms of all-gather per layer, the copy queue falls behind the forward and the chain waits for AG(b)
# at every bucket (+41.8 ms/step).  One per slot keeps the reference's cadence of one all-gather per layer.
# Lead: three slots.  With two, the chain still waited 0.2-0.9 ms at 8-12 of the 27 buckets in the back half of the
# forward (3.4-5.7 ms/iter): a need point falls mid-layer, so two slots are ~1.3 layers (~5 ms) of lead while Adam(b)
# under the forward's kernels plus the 7-copy all-gather take 3.8-4.5 ms from the slot event to the last copy, and peer
# skew eats the rest.  Three slots (~2.3 layers, ~9 ms) cover that; only the phase moves one layer earlier.
YIELD_BUCKETS_PER_SLOT = int(os.getenv("OPT_SIDE_STREAM_YIELD_PER_SLOT", "1"))
YIELD_LEAD_SLOTS = int(os.getenv("OPT_SIDE_STREAM_YIELD_LEAD", "3"))

_INSTALLED = False
_ORIG = {}


def _rank():
    return int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))


def _log(msg, all_ranks=False):
    if all_ranks or _rank() == 0:
        _OUT.write(f"OPT_SIDE_STREAM rank={_rank()} {msg}\n")
        _OUT.flush()


class _GroupInfo:
    __slots__ = ("opt", "group", "lr_t", "step_t", "beta1", "beta2", "eps", "wd", "bias_correction", "mode", "name")

    def __init__(self, opt, group, name):
        self.opt = opt
        self.group = group
        self.name = name
        self.lr_t = torch.zeros(1, dtype=torch.float32, device="cuda")
        self.step_t = torch.zeros(1, dtype=torch.int32, device="cuda")
        betas = group.get("betas_", group["betas"])
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(group["eps"])
        self.wd = float(group.get("weight_decay_", group["weight_decay"]))
        self.bias_correction = 1 if group.get("bias_correction_", group["bias_correction"]) else 0
        self.mode = int(opt.optimizer.adam_w_mode)

    def effective(self):
        g = self.group
        betas = g.get("betas_", g["betas"])
        return (
            float(betas[0]),
            float(betas[1]),
            float(g["eps"]),
            float(g.get("weight_decay_", g["weight_decay"])),
            1 if g.get("bias_correction_", g["bias_correction"]) else 0,
        )


class _BucketGroupWork:
    __slots__ = ("bg", "index", "shards", "calls", "numel", "flat")

    def __init__(self, bg, index, shards):
        self.bg = bg
        self.index = index
        self.shards = shards            # this rank's reduce-scatter output slice of every bucket's grad_data
        self.calls = []                 # (kernel, [lists], ginfo)
        self.numel = sum(s.numel() for s in shards)
        self.flat = None                # [_FlatShard] (one per bucket) when the flat Adam kernel is in charge


class _FlatShard:
    """One bucket's shard as the flat Adam kernel sees it."""
    __slots__ = ("bucket", "g", "p", "master", "m", "v", "seg_start", "seg_group", "nseg", "numel", "npieces", "grid")

    def __init__(self, bucket, g, p, master, m, v, seg_start, seg_group, nseg, npieces, grid):
        self.bucket = bucket
        self.g = g                      # bf16 grad shard (contiguous slice of bucket.grad_data)
        self.p = p                      # bf16 param shard (the same slice of bucket.param_data)
        self.master = master            # fp32 [numel] flat master weights (the per-param tensors are views into it)
        self.m = m                      # fp32 [numel] flat exp_avg
        self.v = v                      # fp32 [numel] flat exp_avg_sq
        self.seg_start = seg_start      # int64 [nseg + 1]
        self.seg_group = seg_group      # int32 [nseg], -1 = padding
        self.nseg = nseg
        self.numel = g.numel()
        self.npieces = npieces
        self.grid = grid


class _State:
    def __init__(self):
        self.ready = False
        self.disabled_reason = None
        self.chained = None
        self.dist_opts = []
        self.ddps = []
        self.groups = []
        self.bg_list = []
        self.work = {}
        self.managed_buffers = set()
        self.order = []
        self.need = {}
        self.order_known = False
        self.slot_of = {}
        self.slot_map = {}
        self.n_slots = 0
        self.pending = None
        self.applying = False
        self.launch_adam = False
        self.training_call = False
        self.attn_counter = 0
        self.iter_dispatched = set()
        self.partials_done = set()
        self.side = None
        self.flags_stream = None
        self.check_stream = None
        self.flags_event = None
        self.flags_static = None
        self.gen_dev = None
        self.gen_host = 0
        self.flags_recorded_this_call = False
        self.flags_in_graph = False
        self.flags_valid = False
        self.moe_layers = []
        self.hooks_registered = False
        self.flat_on = False
        self.flat_reason = ""
        self.flat_groups = None         # per-group argument lists of the flat kernel (lrs, steps, b1, b2, eps, wd, bc, mode)
        self.flat_shards = 0
        self.stats = {
            "deferred_steps": 0, "applied_steps": 0, "dropped_pending": 0, "flushed": 0, "late_dispatch": 0,
            "early_checks": 0, "early_check_fallback": 0, "gen_mismatch": 0, "prologue_calls": 0,
            "yield_launches": 0, "rs_partials": 0, "captures": 0, "flat_launches": 0, "te_adam_launches": 0,
        }

    # ------------------------------------------------------------------ setup
    def init(self, chained):
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from megatron.core.distributed.param_and_grad_buffer import shard_buffer
        from transformer_engine.pytorch.optimizers import FusedAdam
        import transformer_engine_torch as tex

        self.chained = chained
        opts = getattr(chained, "chained_optimizers", [chained])
        dist_opts = [o for o in opts if isinstance(o, DistributedOptimizer) and not getattr(o, "is_stub_optimizer", False)]
        if len(dist_opts) != len(opts) or not dist_opts:
            return self._disable("not every chained optimizer is a DistributedOptimizer")
        ddps = []
        for o in dist_opts:
            if o.ddp_config.use_megatron_fsdp or o.ddp_config.num_distributed_optimizer_instances != 1:
                return self._disable("Megatron-FSDP or several distributed-optimizer instances")
            if o.config.use_precision_aware_optimizer or o.ddp_config.fp8_param_gather or getattr(o.ddp_config, "fp4_param_gather", False):
                return self._disable("precision-aware optimizer or quantized param gather")
            if o.config.clip_grad <= 0.0 or o.config.log_num_zeros_in_grad or o.grad_scaler is not None:
                return self._disable("clip_grad == 0, num-zeros logging or a grad scaler")
            fused = o.optimizer
            if not isinstance(fused, FusedAdam) or fused.capturable or fused.master_weights:
                return self._disable("inner optimizer is not TE FusedAdam in its fp32-main-param mode")
            for chunk in o.model_chunks:
                if chunk not in ddps:
                    ddps.append(chunk)
        if any(len(b.extra_main_grads) > 0 for ddp in ddps for b in ddp.buffers + ddp.expert_parallel_buffers):
            return self._disable("buffers with extra main grads")
        if not (ddps[0].ddp_config.overlap_grad_reduce and ddps[0].ddp_config.overlap_param_gather):
            return self._disable("overlap_grad_reduce / overlap_param_gather off")
        if ddps[0].ddp_config.align_param_gather or getattr(ddps[0], "overlap_param_gather_with_optimizer_step", False):
            return self._disable("align_param_gather / overlap_param_gather_with_optimizer_step")
        self.dist_opts = dist_opts
        self.ddps = ddps

        # Bucket groups in reduce-scatter (backward) order; one work record each.
        for ddp in ddps:
            for bg in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
                shards = []
                for bucket in bg.buckets:
                    shards.append(
                        shard_buffer(bucket.grad_data, bg.intra_distributed_optimizer_instance_size)[
                            bg.intra_distributed_optimizer_instance_rank
                        ]
                    )
                self.work[id(bg)] = _BucketGroupWork(bg, len(self.bg_list), shards)
                self.bg_list.append(bg)
            for b in ddp.buffers + ddp.expert_parallel_buffers:
                self.managed_buffers.add(id(b))

        def bucket_group_of(model_param):
            for ddp in ddps:
                bg = ddp.param_to_bucket_group.get(model_param)
                if bg is not None:
                    return bg
            raise KeyError("model param without a bucket group")

        # Per (bucket group, param group) tensor lists for the capturable master Adam.
        per_bg_group = {}
        n_bf16 = n_fp32 = 0
        flat_pieces = []   # (g slice, param shard, m, v, main param, group index) of every bf16 shard
        for oi, opt in enumerate(dist_opts):
            fused = opt.optimizer
            for gidx, group in enumerate(fused.param_groups):
                ginfo = _GroupInfo(opt, group, f"opt{oi}.g{gidx}")
                self.groups.append(ginfo)
                lists_bf16 = {}
                lists_fp32 = {}
                for mp, sp, mainp in zip(
                    opt.model_float16_groups[gidx], opt.shard_float16_groups[gidx], opt.shard_fp32_from_float16_groups[gidx]
                ):
                    if sp is None or mainp is None or mp.dtype != torch.bfloat16:
                        return self._disable("quantized or non-bf16 model param shard")
                    rng = opt._get_model_param_range_map(mp)["param"]
                    g = mp.main_grad.view(-1)[rng.start:rng.end]
                    st = fused.state[mainp]
                    if "exp_avg" not in st or "exp_avg_sq" not in st:
                        return self._disable("FusedAdam state not initialised")
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                    if not (g.numel() == sp.numel() == mainp.numel() == m.numel() == v.numel()) or m.dtype != torch.float32:
                        return self._disable("shard / state shape or dtype mismatch")
                    if g.numel() == 0:
                        continue
                    bg = bucket_group_of(mp)
                    lists_bf16.setdefault(id(bg), ([], [], [], [], []))
                    for lst, t in zip(lists_bf16[id(bg)], (g, sp, m, v, mainp)):
                        lst.append(t)
                    flat_pieces.append((g, sp, m, v, mainp, len(self.groups) - 1))
                    n_bf16 += 1
                for mp, sp in zip(opt.model_fp32_groups[gidx], opt.shard_fp32_groups[gidx]):
                    rng = opt._get_model_param_range_map(mp)["param"]
                    g = mp.main_grad.view(-1)[rng.start:rng.end]
                    st = fused.state[sp]
                    if "exp_avg" not in st or "exp_avg_sq" not in st:
                        return self._disable("FusedAdam state not initialised (fp32)")
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                    if not (g.numel() == sp.numel() == m.numel() == v.numel()) or g.dtype != torch.float32:
                        return self._disable("fp32 shard / state shape or dtype mismatch")
                    if g.numel() == 0:
                        continue
                    bg = bucket_group_of(mp)
                    lists_fp32.setdefault(id(bg), ([], [], [], []))
                    for lst, t in zip(lists_fp32[id(bg)], (g, sp, m, v)):
                        lst.append(t)
                    n_fp32 += 1
                for key, lists in lists_bf16.items():
                    self.work[key].calls.append((tex.multi_tensor_adam_capturable_master, [list(l) for l in lists], ginfo))
                for key, lists in lists_fp32.items():
                    self.work[key].calls.append((tex.multi_tensor_adam_capturable, [list(l) for l in lists], ginfo))

        n_bg = len(self.bg_list)
        self.partials = torch.zeros(n_bg, dtype=torch.float32, device="cuda")
        self.total = torch.zeros(1, dtype=torch.float32, device="cuda")
        self.grad_norm = torch.zeros(1, dtype=torch.float32, device="cuda")
        self.coeff = torch.ones(1, dtype=torch.float32, device="cuda")
        self.noop = torch.ones(1, dtype=torch.int32, device="cuda")
        self.dummy_overflow_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.flags_static = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.gen_dev = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.max_norm = float(dist_opts[0].config.clip_grad)
        self._setup_flat(flat_pieces, n_fp32)
        self.grad_stats_group = chained.get_grad_stats_parallel_group()
        self.side = torch.cuda.Stream(priority=0)  # torch's lowest priority
        self.flags_stream = torch.cuda.Stream(priority=0)
        self.check_stream = torch.cuda.Stream()
        try:
            self.flags_event = torch.cuda.Event(external=True)
        except TypeError:
            self.flags_event = None
            _log("torch.cuda.Event has no `external` flag: the overflow check stays synchronous (finding)")
        # Initialise the NCCL communicators the captured side-stream collectives use, outside any capture.
        dist.all_reduce(torch.zeros(4, dtype=torch.int32, device="cuda"))
        dist.all_reduce(torch.zeros(1, dtype=torch.float32, device="cuda"), group=self.grad_stats_group)
        self._register_model_hooks()
        self._warm_kernels()
        torch.cuda.synchronize()
        self.ready = True
        _log(
            f"ready: {len(dist_opts)} dist-opts, {len(self.groups)} param groups, {n_bg} bucket groups, "
            f"{n_bf16} bf16 + {n_fp32} fp32 shards, adam launches/step={sum(len(w.calls) for w in self.work.values())}, "
            f"flat={self.flat_on} flat shards/step={self.flat_shards} ({self.flat_reason}), "
            f"attention slots={self.n_slots}, moe layers={len(self.moe_layers)}, max_norm={self.max_norm}, "
            f"yield={YIELD} hipri={HIPRI} early_check={EARLY_CHECK and self.flags_event is not None} "
            f"groups={[(g.name, g.beta1, g.beta2, g.eps, g.wd, g.bias_correction, g.mode) for g in self.groups]}"
        )

    def _disable(self, reason):
        self.disabled_reason = reason
        self.ready = False
        _log(f"DISABLED (original optimizer path kept): {reason}")
        return None

    def _register_model_hooks(self):
        from megatron.core.utils import get_attr_wrapped_model

        attn = []
        for ddp in self.ddps:
            model = get_attr_wrapped_model(ddp, "decoder", allow_none=False, return_model_obj=True)
            decoder = model.decoder
            for layer in decoder.layers:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None and hasattr(mlp, "token_dispatcher") and hasattr(mlp.token_dispatcher, "check_over_budget"):
                    self.moe_layers.append(mlp)
                core = getattr(getattr(layer, "self_attention", None), "core_attention", None)
                if core is not None:
                    attn.append(core)
            decoder.register_forward_hook(self._on_decoder_forward)
        for k, mod in enumerate(attn):
            mod.register_forward_pre_hook(lambda m, args, k=k: self._on_attention(k))
        self.n_slots = len(attn)
        self.hooks_registered = True

    def _warm_kernels(self):
        """Run every captured kernel once outside capture (noop flag set: no state changes)."""
        from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm

        self.noop.fill_(1)
        seen = set()
        for w in self.work.values():
            for kernel, lists, ginfo in w.calls:
                key = (kernel, lists[0][0].dtype, lists[1][0].dtype)
                if key in seen:
                    continue
                seen.add(key)
                self._launch_adam(kernel, lists, ginfo)
        if self.flat_on:
            for w in self.work.values():
                if w.flat:
                    self._launch_flat(w.flat[0])
                    break
        w0 = self.work[id(self.bg_list[0])]
        for shard in w0.shards:
            multi_tensor_applier(multi_tensor_l2norm, self.dummy_overflow_buf, [[shard]], False)

    # ------------------------------------------------------------ per-iteration
    def pre_iteration(self):
        """Host side of one training forward_backward call (eager, capture or replay): consume the pending
        step, fill the device scalars.  Runs before FullCudaGraphWrapper replays / captures."""
        self._consume_pending()
        self.gen_host += 1
        self.attn_counter = 0
        self.iter_dispatched = set()
        self.partials_done = set()
        self.flags_recorded_this_call = False

    def _consume_pending(self):
        warm = any("betas_" in g.group for g in self.groups)
        if warm and self.pending is not None:
            # WarmupCallback neutralised the optimizer (betas 1, lr 0): the reference's warmup steps leave
            # params and moments untouched, so a deferred warmup step is dropped rather than applied.
            self.pending = None
            self.stats["dropped_pending"] += 1
        self.applying = self.pending is not None
        p = self.pending
        self.pending = None
        if self.applying:
            for k, gi in enumerate(self.groups):
                eff = gi.effective()
                baked = (gi.beta1, gi.beta2, gi.eps, gi.wd, gi.bias_correction)
                if any(abs(a - b) > 1e-12 for a, b in zip(eff, baked)):
                    raise RuntimeError(
                        f"OPT_SIDE_STREAM: optimizer hyperparameters of {gi.name} changed from the captured values "
                        f"{baked} to {eff}; the captured Adam launches would be wrong"
                    )
                step = gi.group.get("step", 0)
                if isinstance(step, torch.Tensor):
                    gi.group["step"] = step + 1
                    step_val = int(gi.group["step"].item()) if gi.group["step"].numel() == 1 else 0
                else:
                    gi.group["step"] = int(step) + 1
                    step_val = gi.group["step"]
                gi.lr_t.fill_(float(p["lr"][k]))
                gi.step_t.fill_(step_val)
            self.noop.fill_(0)
            self.stats["applied_steps"] += 1
        else:
            self.noop.fill_(1)

    def defer_step(self):
        """optimizer.step() replacement: snapshot what the step needs; the launches are the next prologue."""
        self.pending = {"lr": [float(gi.group["lr"]) for gi in self.groups]}
        self.stats["deferred_steps"] += 1
        return True, self.grad_norm, None

    def drop_pending(self):
        if self.pending is not None:
            self.pending = None
            self.stats["dropped_pending"] += 1

    def flush_pending(self):
        """Apply a pending step eagerly (before a forced parameter sync: eval, checkpoint)."""
        if self.pending is None:
            return
        self._consume_pending()
        comp = torch.cuda.current_stream()
        self.side.wait_stream(comp)
        with torch.cuda.stream(self.side):
            for bg in self.bg_list:
                self._adam_and_zero(bg)
        comp.wait_stream(self.side)
        self.applying = False
        self.stats["flushed"] += 1

    # ---------------------------------------------------------- side-stream work
    def _launch_adam(self, kernel, lists, ginfo):
        from transformer_engine.pytorch.optimizers import multi_tensor_applier
        from transformer_engine.pytorch.optimizers.fused_adam import _MAX_TENSORS_PER_MULTI_TENSOR_ADAM_CALL as MAXT

        n = len(lists[0])
        for start in range(0, n, MAXT):
            chunk = [l[start:start + MAXT] for l in lists]
            self.stats["te_adam_launches"] += 1
            multi_tensor_applier(
                kernel, self.noop, chunk, ginfo.lr_t, ginfo.beta1, ginfo.beta2, ginfo.eps, ginfo.step_t,
                ginfo.mode, ginfo.bias_correction, ginfo.wd, self.coeff,
            )

    # ----------------------------------------------------- flat per-bucket Adam
    def _flat_off(self, reason):
        self.flat_on = False
        self.flat_reason = reason
        self.flat_shards = 0
        for w in self.work.values():
            w.flat = None
        _log(f"flat Adam OFF ({reason}): TE multi_tensor_adam_capturable_master per tensor list stays in charge")

    def _setup_flat(self, pieces, n_fp32):
        """Re-point every bf16 shard's fp32 master / m / v into one flat buffer per bucket (laid out like the grad shard)
        and build the segment tables; any layout assumption that fails leaves TE's per-list path in charge (nothing is
        modified before every check has passed)."""
        if not FLAT:
            return self._flat_off("disabled by FLAT_ADAM=0")
        if n_fp32 != 0:
            return self._flat_off(f"{n_fp32} fp32 param shards (only the bf16 master path is flattened)")
        try:
            import custom_kernels.flat_adam_cuda as FA
        except Exception as e:  # noqa: BLE001
            return self._flat_off(f"import failed: {e!r}")
        if not FA.ensure_loaded():
            return self._flat_off(FA.state()["why"])
        from megatron.core.distributed.param_and_grad_buffer import shard_buffer

        max_seg, max_groups = FA.state()["max_seg"], FA.state()["max_groups"]
        if len(self.groups) > max_groups:
            return self._flat_off(f"{len(self.groups)} param groups > {max_groups}")
        recs = []
        for bg in self.bg_list:
            size = bg.intra_distributed_optimizer_instance_size
            rank = bg.intra_distributed_optimizer_instance_rank
            for bucket in bg.buckets:
                sg = shard_buffer(bucket.grad_data, size)[rank]
                spd = shard_buffer(bucket.param_data, size)[rank]
                if not (sg.is_contiguous() and spd.is_contiguous() and sg.dtype == torch.bfloat16 and spd.dtype == torch.bfloat16
                        and sg.numel() == spd.numel()):
                    return self._flat_off("bucket shard is not a contiguous bf16 slice of grad_data / param_data")
                recs.append(dict(bg=bg, bucket=bucket, g=sg, p=spd, n=sg.numel(), lo=sg.data_ptr(), hi=sg.data_ptr() + 2 * sg.numel(), pieces=[]))
        for (g, sp, m, v, mainp, gidx) in pieces:
            rec = next((r for r in recs if r["lo"] <= g.data_ptr() < r["hi"]), None)
            if rec is None:
                return self._flat_off("a grad shard piece lies outside every bucket shard")
            off = (g.data_ptr() - rec["lo"]) // 2
            k = g.numel()
            if not (g.is_contiguous() and off + k <= rec["n"]):
                return self._flat_off("a grad shard piece is not contiguous inside its bucket shard")
            if not (sp.is_contiguous() and sp.numel() == k and sp.data_ptr() == rec["p"].data_ptr() + 2 * off):
                return self._flat_off("a param shard piece is not at the grad piece's offset of param_data")
            for t in (m, v, mainp):
                if not (t.is_contiguous() and t.numel() == k and t.dtype == torch.float32 and t.dim() == 1):
                    return self._flat_off("a master / m / v tensor is not a contiguous fp32 1-D tensor of the piece's size")
            rec["pieces"].append((off, k, gidx, m, v, mainp))
        plans = []
        for rec in recs:
            n = rec["n"]
            if n % FA.VEC != 0 or rec["g"].data_ptr() % 16 or rec["p"].data_ptr() % 16:
                return self._flat_off(f"bucket shard numel {n} or base pointers not 16-B aligned")
            ps = sorted(rec["pieces"], key=lambda t: t[0])
            cur = 0
            for off, k, *_ in ps:
                if off < cur:
                    return self._flat_off("overlapping param pieces inside a bucket shard")
                cur = off + k
            try:
                seg_start, seg_group, nseg = FA.plan_segments([(off, k, gidx) for off, k, gidx, *_ in ps], n, rec["g"].device)
            except Exception as e:  # noqa: BLE001
                return self._flat_off(f"segment planning failed: {e!r}")
            if nseg > max_seg:
                return self._flat_off(f"bucket shard has {nseg} segments > {max_seg}")
            plans.append((rec, ps, seg_start, seg_group, nseg))
        # Every check passed: re-point the per-parameter fp32 tensors into flat per-bucket buffers (same objects).
        with torch.no_grad():
            for rec, ps, seg_start, seg_group, nseg in plans:
                n, dev = rec["n"], rec["g"].device
                master = torch.zeros(n, dtype=torch.float32, device=dev)
                m_flat = torch.zeros(n, dtype=torch.float32, device=dev)
                v_flat = torch.zeros(n, dtype=torch.float32, device=dev)
                for off, k, gidx, m, v, mainp in ps:
                    master[off:off + k].copy_(mainp)
                    m_flat[off:off + k].copy_(m)
                    v_flat[off:off + k].copy_(v)
                    mainp.set_(master[off:off + k])
                    m.set_(m_flat[off:off + k])
                    v.set_(v_flat[off:off + k])
                    assert mainp.data_ptr() == master.data_ptr() + 4 * off and mainp.numel() == k
                    assert m.data_ptr() == m_flat.data_ptr() + 4 * off and v.data_ptr() == v_flat.data_ptr() + 4 * off
                fb = _FlatShard(rec["bucket"], rec["g"], rec["p"], master, m_flat, v_flat, seg_start, seg_group, nseg, len(ps),
                                FA.grid_for(n))
                w = self.work[id(rec["bg"])]
                if w.flat is None:
                    w.flat = []
                w.flat.append(fb)
        torch.cuda.synchronize()
        self.flat_groups = (
            [gi.lr_t for gi in self.groups], [gi.step_t for gi in self.groups],
            [gi.beta1 for gi in self.groups], [gi.beta2 for gi in self.groups], [gi.eps for gi in self.groups],
            [gi.wd for gi in self.groups], [gi.bias_correction for gi in self.groups], [gi.mode for gi in self.groups],
        )
        self.flat_on = True
        self.flat_shards = sum(len(w.flat or []) for w in self.work.values())
        self.flat_reason = (
            f"{self.flat_shards} shards, pieces={sum(fb.npieces for w in self.work.values() for fb in (w.flat or []))}, "
            f"segments max={max(fb.nseg for w in self.work.values() for fb in (w.flat or []))}, "
            f"numel min/max={min(fb.numel for w in self.work.values() for fb in (w.flat or []))}/"
            f"{max(fb.numel for w in self.work.values() for fb in (w.flat or []))}, grid={FA.grid_for(plans[0][0]['n'])} "
            f"threads={FA.THREADS} waves={FA.WAVES} write_g={FA.WRITE_G}"
        )
        if FLAT_SELFCHECK:
            self._flat_selfcheck()

    def _launch_flat(self, fb):
        import custom_kernels.flat_adam_cuda as FA

        lrs, steps, b1, b2, eps, wd, bc, mode = self.flat_groups
        FA.flat_adam(fb.g, fb.p, fb.master, fb.m, fb.v, self.noop, self.coeff, fb.seg_start, fb.seg_group,
                     lrs, steps, b1, b2, eps, wd, bc, mode, grid=fb.grid)
        self.stats["flat_launches"] += 1

    @torch.no_grad()
    def _flat_selfcheck(self):
        """Eager, once at init (FLAT_ADAM_SELFCHECK=1, the default): run the flat kernel and TE's per-list launches on the
        real buffers of the first bucket group and compare p / master / m / v bitwise; a mismatch turns the flat path off for
        the run.  g is only reported (TE writes the scaled grad back, the flat kernel does not unless FLAT_ADAM_WRITE_G=1)."""
        w = next((w for w in self.work.values() if w.flat), None)
        if w is None:
            return self._flat_off("self-check: no flat shard")
        names = ("g", "p", "master", "m", "v")
        fbs = list(w.flat)
        snap = [[getattr(fb, nm).clone() for nm in names] for fb in fbs]
        saved = (self.noop.clone(), self.coeff.clone(), [(gi.lr_t.clone(), gi.step_t.clone()) for gi in self.groups])
        try:
            self.noop.fill_(0)
            self.coeff.fill_(0.5)
            for gi in self.groups:
                gi.lr_t.fill_(1.0e-3)
                gi.step_t.fill_(3)
            for fb in fbs:
                self._launch_flat(fb)
            torch.cuda.synchronize()
            flat_res = [[getattr(fb, nm).clone() for nm in names] for fb in fbs]
            for fb, sn in zip(fbs, snap):
                for nm, t in zip(names, sn):
                    getattr(fb, nm).copy_(t)
            for kernel, lists, ginfo in w.calls:
                self._launch_adam(kernel, lists, ginfo)
            torch.cuda.synchronize()
            report = []
            bad = 0
            for fb, fr, sn in zip(fbs, flat_res, snap):
                for nm, a, s0 in zip(names, fr, sn):
                    b = getattr(fb, nm)
                    ai = a.view(torch.int16) if a.dtype == torch.bfloat16 else a.view(torch.int32)
                    bi = b.view(torch.int16) if b.dtype == torch.bfloat16 else b.view(torch.int32)
                    mism = int((ai != bi).sum().item())
                    if nm == "g":
                        s0i = s0.view(torch.int16)
                        report.append(f"g: te_writes_g={bool((bi != s0i).any().item())} flat_writes_g={bool((ai != s0i).any().item())}")
                        continue
                    changed = int((bi != (s0.view(torch.int16) if s0.dtype == torch.bfloat16 else s0.view(torch.int32))).sum().item())
                    maxabs = float((a.float() - b.float()).abs().max().item())
                    report.append(f"{nm}: mismatches={mism}/{a.numel()} changed_by_te={changed} max_abs_diff={maxabs:.3e}")
                    bad += mism
            _log("flat Adam self-check (bucket group 0, lr=1e-3 step=3 clip=0.5): " + "; ".join(report), all_ranks=True)
            if bad:
                self._flat_off(f"self-check: {bad} element mismatches vs TE's kernel")
        finally:
            for fb, sn in zip(fbs, snap):
                for nm, t in zip(names, sn):
                    getattr(fb, nm).copy_(t)
            self.noop.copy_(saved[0])
            self.coeff.copy_(saved[1])
            for gi, (lr_t, step_t) in zip(self.groups, saved[2]):
                gi.lr_t.copy_(lr_t)
                gi.step_t.copy_(step_t)
            torch.cuda.synchronize()

    def _adam_and_zero(self, bg):
        """Current stream must be the side stream."""
        w = self.work[id(bg)]
        if w.flat is not None:
            for fb in w.flat:
                self._launch_flat(fb)
        else:
            for kernel, lists, ginfo in w.calls:
                self._launch_adam(kernel, lists, ginfo)
        for bucket in bg.buckets:
            if not getattr(bucket, "no_zero_fill", False):  # never set in this tree: every bucket is zero-filled
                bucket.grad_data.zero_()

    def _run_bucket(self, bg):
        """Adam (+ zero) if this iteration applies a step, then dispatch the bucket's all-gather.  Side stream."""
        if id(bg) in self.iter_dispatched:
            return
        self.iter_dispatched.add(id(bg))
        if self.launch_adam:
            self._adam_and_zero(bg)
        if not bg.param_gather_dispatched:
            bg.start_param_sync()

    def prologue(self):
        """Start of a training forward_backward call (eager or capture): side-stream Adam/zero/AG."""
        self.stats["prologue_calls"] += 1
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            self.stats["captures"] += 1
        self.launch_adam = capturing or self.applying
        comp = torch.cuda.current_stream()
        if not self.order_known:
            # First-needed order not recorded yet: the DDP pre-hook chain keeps the all-gather schedule;
            # the Adam of every bucket runs first, joined back so the chain's all-gathers see the update.
            if self.launch_adam:
                self.side.wait_stream(comp)
                with torch.cuda.stream(self.side):
                    for bg in self.bg_list:
                        self._adam_and_zero(bg)
                comp.wait_stream(self.side)
            return
        self.side.wait_stream(comp)
        with torch.cuda.stream(self.side):
            for bg in self.order:
                if self.slot_of[id(bg)] < 0:
                    self._run_bucket(bg)

    def tail(self):
        """End of a training forward_backward call: norm all-reduce, clip coefficient, joins."""
        from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm  # noqa: F401

        comp = torch.cuda.current_stream()
        if not self.order_known:
            self._finalize_order()
        with torch.cuda.stream(self.side):
            # Any all-gather dispatched here but never waited on (unused bucket) must rejoin the capture.
            for bg in self.bg_list:
                if bg.param_gather_handle is not None and id(bg) in self.iter_dispatched:
                    bg.param_gather_handle.wait()
                    bg.param_gather_handle = None
                    bg._post_param_sync()
            self.total.copy_(self.partials.sum().view(1))
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM, group=self.grad_stats_group)
            torch.pow(self.total, 0.5, out=self.grad_norm)
            # Same expression as clip_grad_by_total_norm_fp32: max_norm / (norm + 1e-6), clamped to 1.
            self.coeff.copy_(self.max_norm / (self.grad_norm + 1.0e-6))
            self.coeff.clamp_max_(1.0)
        comp.wait_stream(self.side)
        if self.flags_recorded_this_call:
            # Only when the flags were computed this call: joining a stream with no captured work raises
            # cudaErrorStreamCaptureIsolation (seen with OPT_SIDE_STREAM_EARLY_CHECK=0).
            comp.wait_stream(self.flags_stream)

    def on_rs_drained(self, bg):
        """finish_grad_sync hook (main thread, compute stream has waited on this bucket group's RS): grad-norm partial
        of the reduced shard on the side stream, overlapping the waits for the remaining buckets.  Not from the backward
        hook: a side-stream wait on the NCCL handle there made the captured chain wait for every RS (27 x 2.8 ms exposed)."""
        from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm

        if id(bg) in self.partials_done:
            return
        self.partials_done.add(id(bg))
        w = self.work[id(bg)]
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            acc = None
            for shard in w.shards:
                norm, _ = multi_tensor_applier(multi_tensor_l2norm, self.dummy_overflow_buf, [[shard]], False)
                sq = norm * norm
                acc = sq if acc is None else acc + sq
            self.partials[w.index].copy_(acc.view(-1)[0])
        self.stats["rs_partials"] += 1

    def record_need(self, bg):
        if self.order_known or id(bg) in self.need:
            return
        self.need[id(bg)] = self.attn_counter
        self.order.append(bg)
        if len(self.order) == len(self.bg_list):
            self._finalize_order()

    def _finalize_order(self):
        for bg in self.bg_list:
            if id(bg) not in self.need:
                self.need[id(bg)] = 0
                self.order.append(bg)
        self.slot_of = {}
        self.slot_map = {}
        for j, bg in enumerate(self.order):
            slot = -1
            if YIELD and self.n_slots > 0 and j >= YIELD_BUCKETS_PER_SLOT:
                slot = min((j - YIELD_BUCKETS_PER_SLOT) // YIELD_BUCKETS_PER_SLOT, self.need[id(bg)] - YIELD_LEAD_SLOTS)
                slot = min(slot, self.n_slots - 1)
            self.slot_of[id(bg)] = slot
            if slot >= 0:
                self.slot_map.setdefault(slot, []).append(bg)
        self.order_known = True
        _log(
            "first-needed order recorded: "
            + " ".join(
                f"{self.work[id(bg)].index}@{self.need[id(bg)]}->{self.slot_of[id(bg)]}" for bg in self.order
            )
        )

    def _on_attention(self, k):
        if not self.ready or not self.training_call:
            return
        self.attn_counter += 1
        if not self.order_known:
            return
        bgs = self.slot_map.get(k)
        if not bgs:
            return
        comp = torch.cuda.current_stream()
        self.side.wait_stream(comp)
        with torch.cuda.stream(self.side):
            for bg in bgs:
                self._run_bucket(bg)
        self.stats["yield_launches"] += len(bgs)

    def ensure_dispatched(self, bg):
        """finish_param_sync hook: a bucket needed before its scheduled slot is dispatched now (late)."""
        if id(bg) in self.iter_dispatched:
            return
        self.stats["late_dispatch"] += 1
        with torch.cuda.stream(self.side):
            self._run_bucket(bg)

    # ----------------------------------------------------------- overflow flags
    def _compute_flags(self):
        from megatron.core.transformer.moe.paged_stash import check_paged_stash_host_spill, check_paged_stash_overflow

        stash_overflow = check_paged_stash_overflow().view(-1)[0]
        host_spill = check_paged_stash_host_spill().view(-1)[0]
        overbudget = torch.zeros(1, dtype=torch.bool, device=stash_overflow.device).view(-1)[0]
        for mlp in self.moe_layers:
            ob = mlp.token_dispatcher.check_over_budget()
            if ob is not None:
                overbudget |= ob.view(-1)[0]
        return torch.stack([stash_overflow.to(torch.int32), overbudget.to(torch.int32), host_spill.to(torch.int32)], dim=0)

    def _on_decoder_forward(self, module, inputs, output):
        if not self.ready or not self.training_call or not EARLY_CHECK or self.flags_event is None:
            return
        comp = torch.cuda.current_stream()
        self.flags_stream.wait_stream(comp)
        with torch.cuda.stream(self.flags_stream):
            flags = self._compute_flags()
            dist.all_reduce(flags, op=dist.ReduceOp.SUM)
            self.gen_dev.add_(1)
            self.flags_static[:3].copy_(flags)
            self.flags_static[3:4].copy_(self.gen_dev)
            self.flags_event.record(self.flags_stream)
        self.flags_recorded_this_call = True

    def early_overflow_check(self):
        """Returns (stash_overflow_ranks, overbudget_ranks, host_spill_ranks) read after the forward, or None
        when the flags of this iteration are not available that way (caller falls back to the original)."""
        if not (self.ready and self.training_call and EARLY_CHECK and self.flags_event is not None and self.flags_valid):
            return None
        s = self.check_stream
        s.wait_event(self.flags_event)
        with torch.cuda.stream(s):
            f = self.flags_static.to("cpu")
        vals = [int(x) for x in f.tolist()]
        if vals[3] != self.gen_host:
            self.stats["gen_mismatch"] += 1
            return None
        self.stats["early_checks"] += 1
        return vals[0], vals[1], vals[2]


_STATE = _State()


# ===================================================================== patches
def _patch_full_cuda_graph_wrapper():
    from megatron.core import full_cuda_graph as fcg

    W = fcg.FullCudaGraphWrapper
    _ORIG["fcg_init"] = W.__init__
    _ORIG["fcg_call"] = W.__call__

    class _Inner:
        """Wraps the schedule function inside the captured region: prologue / tail around it."""

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            st = _STATE
            training = not kwargs.get("forward_only", False)
            if not (training and st.ready):
                return self.fn(*args, **kwargs)
            st.prologue()
            result = self.fn(*args, **kwargs)
            st.tail()
            return result

    def __init__(self, forward_backward_func, *args, **kwargs):
        if not isinstance(forward_backward_func, _Inner):
            forward_backward_func = _Inner(forward_backward_func)
        _ORIG["fcg_init"](self, forward_backward_func, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        st = _STATE
        training = not kwargs.get("forward_only", False)
        st.training_call = training
        if not (training and st.ready):
            return _ORIG["fcg_call"](self, *args, **kwargs)
        graph_existed = W.cuda_graph["training"] is not None
        st.pre_iteration()
        result = _ORIG["fcg_call"](self, *args, **kwargs)
        if graph_existed:
            st.flags_valid = st.flags_in_graph
        else:
            st.flags_valid = st.flags_recorded_this_call
            if W.cuda_graph["training"] is not None:  # this call captured the training graph
                st.flags_in_graph = st.flags_recorded_this_call
        return result

    W.__init__ = __init__
    W.__call__ = __call__

    if HIPRI:
        # The chain is captured from this stream; a high priority here is what makes the side stream (torch's
        # lowest priority, 0) the low-priority one.  Set before the first capture so the lazy getter never
        # creates the default-priority stream.
        if fcg._shared_capture_stream is None:
            fcg._shared_capture_stream = torch.cuda.Stream(priority=-1)
        else:
            _log("shared capture stream already created: HIPRI not applied")


def _patch_optimizer():
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    _ORIG["chained_step"] = ChainedOptimizer.step
    _ORIG["chained_prepare_sync"] = ChainedOptimizer.prepare_model_params_for_param_sync

    @torch.no_grad()
    def step(self):
        st = _STATE
        if st.disabled_reason is not None:
            return _ORIG["chained_step"](self)
        if not st.ready:
            # The first (eager) step initialises the TE optimizer state; everything after it is deferred.
            result = _ORIG["chained_step"](self)
            try:
                st.init(self)
            except Exception as e:  # noqa: BLE001
                st._disable(f"init failed: {e!r}")
                if VERBOSE:
                    raise
            return result
        return st.defer_step()

    def prepare_model_params_for_param_sync(self):
        st = _STATE
        if st.ready:
            st.flush_pending()
        return _ORIG["chained_prepare_sync"](self)

    ChainedOptimizer.step = step
    ChainedOptimizer.prepare_model_params_for_param_sync = prepare_model_params_for_param_sync


def _patch_bucket_groups():
    from megatron.core.distributed import param_and_grad_buffer as pgb

    BG = pgb._ParamAndGradBucketGroup
    BUF = pgb._ParamAndGradBuffer
    _ORIG["bg_finish_grad_sync"] = BG.finish_grad_sync
    _ORIG["bg_finish_param_sync"] = BG.finish_param_sync
    _ORIG["buf_reset"] = BUF.reset

    def finish_grad_sync(self, force_all_reduce=False):
        _ORIG["bg_finish_grad_sync"](self, force_all_reduce)
        st = _STATE
        if st.ready and st.training_call and not force_all_reduce and id(self) in st.work and self.grad_reduce_finished:
            st.on_rs_drained(self)

    def finish_param_sync(self, skip_next_bucket_dispatch=False):
        st = _STATE
        if st.ready and st.training_call and id(self) in st.work:
            st.record_need(self)
            if st.order_known:
                st.ensure_dispatched(self)
                skip_next_bucket_dispatch = True
        return _ORIG["bg_finish_param_sync"](self, skip_next_bucket_dispatch)

    def reset(self):
        st = _STATE
        if st.ready and st.pending is not None and id(self) in st.managed_buffers:
            # The grads of the pending step are still needed; each bucket is zeroed on the side stream right
            # after its Adam (prologue of the next forward_backward call).
            return
        return _ORIG["buf_reset"](self)

    BG.finish_grad_sync = finish_grad_sync
    BG.finish_param_sync = finish_param_sync
    BUF.reset = reset


def _patch_paged_stash_runner():
    try:
        from megatron.core.transformer.moe.paged_stash import PagedStashRunner
    except Exception:  # noqa: BLE001
        return
    _ORIG["psr_check"] = PagedStashRunner.check_moe_overflow

    def check_moe_overflow(self):
        st = _STATE
        res = st.early_overflow_check()
        if res is None:
            return _ORIG["psr_check"](self)
        stash_overflow_ranks, overbudget_ranks, host_spill_ranks = res
        self._required_recv_capacity = None
        self._required_capacity_factor = None
        if stash_overflow_ranks > 0 or overbudget_ranks > 0:
            # Failure path: the original check recomputes the same flags and its second all-reduce.
            st.stats["early_check_fallback"] += 1
            torch.cuda.synchronize()
            return _ORIG["psr_check"](self)
        return stash_overflow_ranks, overbudget_ranks, host_spill_ranks

    PagedStashRunner.check_moe_overflow = check_moe_overflow


def _patch_warmup_callback():
    try:
        import callback_warmup
    except Exception:  # noqa: BLE001
        return
    _ORIG["warmup_data_init"] = callback_warmup.WarmupCallback.on_data_init_start

    def on_data_init_start(self, context):
        try:
            return _ORIG["warmup_data_init"](self, context)
        finally:
            # The warmup's last (neutralised) step must not be applied to the real run.
            _STATE.drop_pending()

    callback_warmup.WarmupCallback.on_data_init_start = on_data_init_start


def _report():
    st = _STATE
    _log(f"stats: ready={st.ready} disabled={st.disabled_reason} {st.stats}", all_ranks=True)


def install():
    global _INSTALLED
    if _INSTALLED or not ENABLED:
        if not ENABLED:
            _log("disabled by OPT_SIDE_STREAM=0")
        return
    _INSTALLED = True
    _patch_full_cuda_graph_wrapper()
    _patch_optimizer()
    _patch_bucket_groups()
    _patch_paged_stash_runner()
    _patch_warmup_callback()
    import atexit

    atexit.register(_report)
    _log(f"installed (yield={YIELD} hipri={HIPRI} early_check={EARLY_CHECK} flat_adam={FLAT})")
