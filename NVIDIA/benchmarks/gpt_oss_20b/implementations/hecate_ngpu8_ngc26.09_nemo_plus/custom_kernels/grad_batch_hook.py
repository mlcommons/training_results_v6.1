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

"""Batch the DDP per-parameter main_grad writes into 2-4 torch._foreach_copy_ / _foreach_add_ launches per bucket.

Megatron DDP's backward post-hook issues ~1700 `main_grad.add_(grad)` launches per step, one per parameter whose wgrad
producer does not write main_grad itself (mostly the 32 x 24 expert biases): ~3.5 ms/step of serialized chain time in the
full-iteration graph.  Here the hook parks (main_grad view, grad) on the parameter's bucket group, keyed by the producing
stream, and start_grad_sync (right before the bucket's reduce-scatter) flushes them with one multi-tensor kernel per
(stream, op) on that stream -- _foreach_copy_ for parameters carrying overwrite_main_grad (unset since an earlier patch
that skipped the grad-buffer zero fill was removed), _foreach_add_ otherwise -- and the reduce-scatter stream waits on every
flush.  Gradients of >= GRAD_BATCH_BIG_NUMEL elements (the embedding table) keep a direct write and grad_added_to_main_grad
parameters are untouched.  Bitwise identical to the reference: same per-element copy / add, only the launch grouping changes.

Engages at the first zero_grad_buffer call when there is one micro-batch per step, overlap_grad_reduce is on, no
Megatron-FSDP and no extra main grads; otherwise Megatron's hook runs unchanged and the reason is logged.  GRAD_BATCH=0
leaves the reference path; GRAD_BATCH_BIG_NUMEL (default 2^20) is the direct-write threshold.
"""
import atexit
import os
import sys

import torch

ENABLED = os.getenv("GRAD_BATCH", "1") == "1"
BIG_NUMEL = int(os.getenv("GRAD_BATCH_BIG_NUMEL", str(1 << 20)))
_OUT = sys.__stdout__
_ORIG = {}
_INSTALLED = False


def _rank():
    return int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))


def _log(msg, all_ranks=False):
    if all_ranks or _rank() == 0:
        _OUT.write(f"GRAD_BATCH rank={_rank()} {msg}\n")
        _OUT.flush()


class _Group:
    """Parked gradients of one bucket group produced on one stream; flushed on that stream (no cross-stream reuse hazard)."""

    __slots__ = ("stream", "copy_dst", "copy_src", "add_dst", "add_src")

    def __init__(self, stream):
        self.stream = stream
        self.copy_dst = []
        self.copy_src = []
        self.add_dst = []
        self.add_src = []


class _State:
    def __init__(self):
        self.decided = set()  # id(ddp) examined
        self.engaged = False
        self.disabled_reason = None
        self.pending = {}  # id(bucket_group) -> {stream key: _Group}
        self.stats = {
            "zero_grad_buffer_calls": 0, "hook_calls": 0, "deferred": 0, "deferred_elems": 0, "direct_big": 0,
            "direct_fallback": 0, "dummy_zero_adds": 0, "fused_skipped": 0, "flushes": 0, "flush_groups": 0,
            "foreach_copy_launches": 0, "foreach_add_launches": 0, "foreach_tensors": 0, "cross_stream_waits": 0,
            "leftover_flushed_at_finish": 0, "first_step_groups_per_bucket": [],
        }

    # ------------------------------------------------------------------ engage
    def maybe_engage(self, ddp):
        if id(ddp) in self.decided:
            return
        self.decided.add(id(ddp))
        reason = self._check(ddp)
        if reason is not None:
            self.disabled_reason = reason
            self.engaged = False
            _log(f"DISABLED ({reason}): the per-parameter main_grad write stays", all_ranks=True)
            return
        self.engaged = True
        n_params = len(list(ddp.params_with_grad))
        n_bg = len(ddp.bucket_groups) + len(ddp.expert_parallel_bucket_groups)
        _log(f"engaged: {n_params} params in {n_bg} bucket groups; direct-write threshold {BIG_NUMEL} elements")

    @staticmethod
    def _check(ddp):
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

            n_mb = get_num_microbatches()
        except Exception as e:  # noqa: BLE001
            return f"num_microbatches unavailable ({e!r})"
        if n_mb != 1:
            return f"{n_mb} micro-batches per step: the hook accumulates across micro-batches"
        cfg = ddp.ddp_config
        if getattr(cfg, "use_megatron_fsdp", False):
            return "Megatron-FSDP"
        if not cfg.overlap_grad_reduce:
            return "overlap_grad_reduce off: no per-bucket reduce-scatter dispatch to flush at"
        buffers = list(ddp.buffers) + list(ddp.expert_parallel_buffers)
        if not buffers:
            return "no grad buffers"
        if any(len(getattr(b, "extra_main_grads", [])) > 0 for b in buffers):
            return "buffers with extra (higher-precision) main grads"
        for p in ddp.params_with_grad:
            if getattr(p, "main_grad", None) is None:
                return "a parameter without main_grad"
        return None

    # ------------------------------------------------------------------- park
    def defer(self, bucket_group, main_grad, grad, overwrite):
        stream = torch.cuda.current_stream()
        key = (stream.device.index, stream.cuda_stream)
        groups = self.pending.setdefault(id(bucket_group), {})
        g = groups.get(key)
        if g is None:
            g = groups[key] = _Group(stream)
        if overwrite:
            g.copy_dst.append(main_grad)
            g.copy_src.append(grad)
        else:
            g.add_dst.append(main_grad)
            g.add_src.append(grad)
        self.stats["deferred"] += 1
        self.stats["deferred_elems"] += grad.numel()

    # ------------------------------------------------------------------ flush
    def flush(self, bucket_group):
        self._flush_key(id(bucket_group))

    def flush_all(self):
        keys = list(self.pending)
        for key in keys:
            self._flush_key(key)
        return len(keys)

    def _flush_key(self, key):
        groups = self.pending.pop(key, None)
        if not groups:
            return
        st = self.stats
        st["flushes"] += 1
        st["flush_groups"] += len(groups)
        if st["flushes"] <= 32:
            st["first_step_groups_per_bucket"].append(len(groups))
        cur = torch.cuda.current_stream()
        for g in groups.values():
            with torch.cuda.stream(g.stream):
                if g.copy_dst:
                    torch._foreach_copy_(g.copy_dst, g.copy_src)
                    st["foreach_copy_launches"] += 1
                    st["foreach_tensors"] += len(g.copy_dst)
                if g.add_dst:
                    torch._foreach_add_(g.add_dst, g.add_src)
                    st["foreach_add_launches"] += 1
                    st["foreach_tensors"] += len(g.add_dst)
            if g.stream.cuda_stream != cur.cuda_stream or g.stream.device != cur.device:
                cur.wait_stream(g.stream)
                st["cross_stream_waits"] += 1
            g.copy_dst.clear()
            g.copy_src.clear()
            g.add_dst.clear()
            g.add_src.clear()


_STATE = _State()


# ===================================================================== patches
def _patch_ddp():
    from megatron.core.distributed import distributed_data_parallel as ddpm

    DDP = ddpm.DistributedDataParallel
    _ORIG["ddp_zero_grad_buffer"] = DDP.zero_grad_buffer
    _ORIG["ddp_make_backward_post_hook"] = DDP._make_backward_post_hook
    _ORIG["ddp_finish_grad_sync"] = DDP.finish_grad_sync

    def zero_grad_buffer(self):
        _STATE.stats["zero_grad_buffer_calls"] += 1
        _STATE.maybe_engage(self)
        return _ORIG["ddp_zero_grad_buffer"](self)

    def _make_backward_post_hook(self, param):
        """Megatron's hook with the main_grad write parked on the bucket group instead of issued per parameter."""
        prev_hook = _ORIG["ddp_make_backward_post_hook"](self, param)
        ddp = self

        def hook(*unused):
            st = _STATE
            if ddpm.is_graph_capturing():
                return
            if not st.engaged or param not in ddp.param_to_bucket_group:
                return prev_hook(*unused)
            bucket_group = ddp.param_to_bucket_group[param]
            if not bucket_group.is_last_microbatch:
                return prev_hook(*unused)
            st.stats["hook_calls"] += 1
            assert param.requires_grad
            cudagraph_wgrad_ready_event = getattr(param, "_cudagraph_wgrad_ready_event", None)
            if ddp.ddp_config.overlap_grad_reduce and cudagraph_wgrad_ready_event is None:
                if not getattr(param, "is_gtp_weight_remat", False):
                    assert param.grad is not None, "param.grad being None is not safe when overlap_grad_reduce is True"
            grad = param.grad
            if grad is not None:
                if not param.grad_added_to_main_grad:
                    overwrite = getattr(param, "overwrite_main_grad", False)
                    if grad.numel() >= BIG_NUMEL:
                        if overwrite:
                            param.main_grad.copy_(grad.data)
                        else:
                            param.main_grad.add_(grad.data)
                        st.stats["direct_big"] += 1
                    else:
                        st.defer(bucket_group, param.main_grad, grad.data, overwrite)
                elif getattr(param, "zero_out_wgrad", False):
                    param.main_grad.add_(grad.data)  # reference: the fused wgrad wrote main_grad, the grad is a zero dummy
                    st.stats["dummy_zero_adds"] += 1
                else:
                    st.stats["fused_skipped"] += 1
            param.grad = None
            if ddp.ddp_config.overlap_grad_reduce:
                bucket_group.register_grad_ready(param, ddp.force_all_reduce)

        return hook

    def finish_grad_sync(self, *args, **kwargs):
        r = _ORIG["ddp_finish_grad_sync"](self, *args, **kwargs)
        st = _STATE
        if st.engaged and st.pending:
            # Every bucket's dispatch flushed its own grads; anything left here is a path this hook did not foresee.
            n = st.flush_all()
            st.stats["leftover_flushed_at_finish"] += n
            _log(f"WARNING: {n} bucket group(s) still had parked gradients after finish_grad_sync; flushed late", all_ranks=True)
        return r

    DDP.zero_grad_buffer = zero_grad_buffer
    DDP._make_backward_post_hook = _make_backward_post_hook
    DDP.finish_grad_sync = finish_grad_sync


def _patch_bucket_group():
    from megatron.core.distributed import param_and_grad_buffer as pgb

    BG = pgb._ParamAndGradBucketGroup
    _ORIG["bg_start_grad_sync"] = BG.start_grad_sync  # whichever wrapper another hook installed first, else Megatron's

    def start_grad_sync(self, force_all_reduce=False):
        st = _STATE
        if st.engaged:
            st.flush(self)  # on the current stream context: the reduce-scatter below is ordered after every write
        return _ORIG["bg_start_grad_sync"](self, force_all_reduce)

    BG.start_grad_sync = start_grad_sync


def _report():
    st = _STATE
    _log(f"stats: engaged={st.engaged} disabled={st.disabled_reason} {st.stats}", all_ranks=True)


def install():
    global _INSTALLED
    if _INSTALLED or not ENABLED:
        if not ENABLED:
            _log("disabled by GRAD_BATCH=0")
        return
    _INSTALLED = True
    _patch_ddp()
    _patch_bucket_group()
    atexit.register(_report)
    _log(f"installed (big_numel={BIG_NUMEL})")
