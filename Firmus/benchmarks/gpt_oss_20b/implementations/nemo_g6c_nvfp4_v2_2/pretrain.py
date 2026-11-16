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
import sys

import hydra
import torch
import torch.nn.functional as F
import utils
from callback_warmup import WarmupCallback, create_mock_dataset_config
from callback_logging import DeltaTimingCallback, MLPerfLoggingCallback, mllogger
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    CommOverlapConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    MixedPrecisionConfig,
    OptimizerConfig,
    ProfilingConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.bridge.training.gpt_step import forward_step as _bridge_forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from omegaconf import OmegaConf
from torch.utils.data import Subset

if os.environ.get("NVFP4_TRITON_RMSNORM_BWD", "0") == "1":
    # TE's RMSNorm backward for hidden 2880 is far from bandwidth; both call
    # sites (LayerNormLinear via tex, the op fuser via a module global) must
    # be patched before the CUDA graphs are captured.
    import transformer_engine.pytorch.ops.basic.rmsnorm as _te_rmsnorm_op
    import transformer_engine_torch as _tex
    from transformer_engine.pytorch.ops.basic.nvfp4_triton_rmsnorm_bwd import (
        rmsnorm_bwd as _triton_rmsnorm_bwd,
    )

    _tex.rmsnorm_bwd = _triton_rmsnorm_bwd
    _te_rmsnorm_op.rmsnorm_bwd = _triton_rmsnorm_bwd

import opt_kernels.qknorm_rope_patch as _qknorm_rope  # noqa: E402

if _qknorm_rope.ACTIVE and os.environ.get("NVFP4_QKV_PACKED", "0") == "1":
    # NVIDIA's fused QK norm + RoPE writing Q, K, V into one buffer with the amax, so the
    # FP8 DPA quantizes it in place without concatenating or a separate amax pass.
    from firmus_kernels.qknorm_rope_packed import install as _qknorm_rope_packed_install

    _qknorm_rope_packed_install()

if os.environ.get("NVFP4_TRITON_QKV_ROPE", "0") == "1":
    # TE's fused QKV RoPE runs at ~2.2 TB/s on the GPT-OSS attention shape; the
    # Triton kernels are bit-exact at ~4.5 TB/s. Megatron imports the function by name.
    import megatron.core.transformer.attention as _mc_attention
    from transformer_engine.pytorch.ops.basic.nvfp4_triton_qkv_rope import (
        make_apply_fused_qkv_rotary_pos_emb as _make_triton_qkv_rope,
    )

    _mc_attention.apply_fused_qkv_rotary_pos_emb = _make_triton_qkv_rope(
        _mc_attention.apply_fused_qkv_rotary_pos_emb
    )

    if os.environ.get("NVFP4_QKV_PACKED", "0") == "1":
        # Q, K, V written into one buffer so the FP8 DPA quantizes it without concatenating.
        from transformer_engine.pytorch.ops.fused import grouped_mlp as _te_grouped_mlp_qp

        _te_grouped_mlp_qp._v7_module("nvfp4_qkv_packed").install()

if os.environ.get("NVFP4_FUSED_BIAS_RESIDUAL", "0") == "1":
    # bias_dropout_fusion is rejected with a biased grouped MLP, so the attention
    # output runs `x + bias` and `residual + out` as two kernels per layer.
    import megatron.core.fusions.fused_bias_dropout as _mc_bda
    from transformer_engine.pytorch.ops.basic.nvfp4_triton_bias_residual import (
        bias_residual_add as _bias_residual_add,
        eligible as _bias_residual_eligible,
    )

    _orig_bda_func = _mc_bda._bias_dropout_add_func

    def _fused_bda_func(x_with_bias, residual, prob, training):
        x, bias = x_with_bias
        if _bias_residual_eligible(x, bias, residual, prob, training):
            return _bias_residual_add(x, bias, residual)
        return _orig_bda_func(x_with_bias, residual, prob, training)

    _mc_bda._bias_dropout_add_func = _fused_bda_func

if os.environ.get("NVFP4_RESGRAD", "0") == "1":
    # The residual-stream gradient sum (one BF16 add per norm) is folded into the
    # Triton RMSNorm backward; needs the two patches above.
    from transformer_engine.pytorch.ops.fused import grouped_mlp as _te_grouped_mlp_rg

    _te_grouped_mlp_rg._v7_module("nvfp4_resgrad").install()

    if os.environ.get("FIRMUS_RESGRAD_FWD_NOBIAS", "0") == "1":
        # The MLP residual add folded into the MoE unpermute when the experts have no bias.
        from firmus_kernels.resgrad_fwd_nobias import install as _resgrad_fwd_nobias_install

        _resgrad_fwd_nobias_install()

if os.environ.get("NVFP4_LMHEAD_MXFP8", "0") == "1":
    # The BF16 LM head runs three ~8.5 ms GEMMs per step; Megatron's fp8_output_proj
    # cannot be combined with fp4, so the output layer forward is rerouted to MXFP8.
    from transformer_engine.pytorch.ops.basic.nvfp4_lmhead_mxfp8 import install as _lmhead_install

    _lmhead_install()

if os.environ.get("NVFP4_SWA_TRITON", "0") == "1":
    # cuDNN takes ~1.7 ms fwd+bwd per sliding-window layer, far above the memory floor
    # for a 128-token window; must be installed before the layers are graph-captured.
    from firmus_kernels.swa_attention import install as _swa_install

    _swa_install()

if os.environ.get("NVFP4_EVAL_GRAPH", "0") == "1":
    # MCore replays TE layer graphs only in training mode, so validation runs every layer
    # eagerly (~2x the MXFP8 eval time). With zero dropout, no forced load balancing and no
    # expert bias, the training-mode forward computes the same function, so evaluation
    # replays the captured forward graphs. Params are already synced when eval starts, so
    # the per-layer param all-gather hooks are skipped.
    import megatron.core.transformer.module as _mc_module
    from megatron.core.transformer.cuda_graphs import is_graph_capturing as _is_graph_capturing

    _graphable_cls = next(
        c
        for c in vars(_mc_module).values()
        if isinstance(c, type) and "_should_call_te_cudagraph" in vars(c)
    )
    _orig_graphable_call = _graphable_cls.__call__
    _eval_graph_check = int(os.environ.get("NVFP4_EVAL_GRAPH_CHECK", "0"))
    _eval_graph_checked = [0]

    def _replay_in_eval(self, *args, **kwargs):
        hooks = self.cuda_graph_manual_hooks
        self.cuda_graph_manual_hooks = []
        self.training = True
        try:
            return self._te_cuda_graph_replay(*args, **kwargs)
        finally:
            self.training = False
            self.cuda_graph_manual_hooks = hooks

    def _eval_graph_call(self, *args, **kwargs):
        if (
            not self.training
            and self.cuda_graphs
            and not torch.is_grad_enabled()
            and self.config.cuda_graph_impl == "transformer_engine"
            and not _is_graph_capturing()
        ):
            if _eval_graph_checked[0] >= _eval_graph_check:
                return _replay_in_eval(self, *args, **kwargs)
            # Check mode: compare against the eager eval forward on the same input. The graph
            # result is returned because NVFP4_GRAPH_CHAIN_IO requires each layer graph's
            # input to be the previous layer graph's output buffer.
            _eval_graph_checked[0] += 1
            eager = _orig_graphable_call(self, *args, **kwargs)
            graph = _replay_in_eval(self, *args, **kwargs)
            e = eager[0] if isinstance(eager, tuple) else eager
            g = graph[0] if isinstance(graph, tuple) else graph
            diff = (e.float() - g.float()).abs()
            print(
                f"NVFP4_EVAL_GRAPH_CHECK rank {torch.distributed.get_rank()} "
                f"layer {getattr(self, 'layer_number', -1)} "
                f"bitwise {bool(torch.equal(e, g))} max_abs {diff.max().item():.3e} "
                f"rel_l2 {(diff.norm() / e.float().norm().clamp_min(1e-30)).item():.3e}",
                flush=True,
            )
            return graph
        return _orig_graphable_call(self, *args, **kwargs)

    _graphable_cls.__call__ = _eval_graph_call

if os.environ.get("NVFP4_TAIL_MXFP8", "0") == "1":
    # The four BF16 tail layers cost about twice an NVFP4 layer; MXFP8 matches the
    # precision of the MXFP8 baseline. Must be installed before graph capture.
    from transformer_engine.pytorch.ops.basic.nvfp4_tail_mxfp8 import install as _tail_install

    _tail_install()

if os.environ.get("NVFP4_PAO_BF16_MODEL_PARAMS", "0") == "1":
    # Megatron only routes master weights through TE FusedAdam when fp8_recipe is
    # None/"delayed"; Bridge leaves the default "tensorwise" even though NVFP4 runs
    # have plain BF16 model params. Only valid without fp8/fp4 param gather.
    from megatron.core.optimizer.optimizer_config import OptimizerConfig as _McoreOptCfg

    _orig_opt_post_init = _McoreOptCfg.__post_init__

    def _pao_post_init(self):
        _orig_opt_post_init(self)
        if self.use_precision_aware_optimizer:
            self.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"NVFP4 PAO on BF16 model params (fp8_recipe={self.fp8_recipe})", flush=True)

    _McoreOptCfg.__post_init__ = _pao_post_init

if os.environ.get("NVFP4_DEFER_FC2_BIAS_BF16", "0") == "1":
    # BF16 tail layers run TE's unfused GroupedLinear; hand their scaled FC2 expert bias
    # to the fused-pad unpermute as the NVFP4 fused MLP does. FC1 (wider output) is untouched.
    import transformer_engine.pytorch.ops.basic.grouped_linear as _te_grouped_linear
    from transformer_engine.pytorch.triton.permutation import defer_expert_bias

    _orig_grouped_gemm = _te_grouped_linear.general_grouped_gemm_for_grouped_tensor
    _expert_hidden = int(os.environ.get("NVFP4_PADDED_EXPERT_HIDDEN", "0"))

    def _grouped_gemm_defer_bias(weights, x, out, *args, bias=None, bias_scale=None, **kwargs):
        out_data = getattr(out, "rowwise_data", None)
        bias_data = getattr(bias, "rowwise_data", None)
        if (
            bias_scale is None
            or out_data is None
            or bias_data is None
            or out_data.numel() != bias_scale.numel() * _expert_hidden
        ):
            return _orig_grouped_gemm(
                weights, x, out, *args, bias=bias, bias_scale=bias_scale, **kwargs
            )
        result = _orig_grouped_gemm(weights, x, out, *args, bias=None, bias_scale=None, **kwargs)
        defer_expert_bias(out_data, bias_data.view(-1, _expert_hidden), bias_scale)
        return result

    _te_grouped_linear.general_grouped_gemm_for_grouped_tensor = _grouped_gemm_defer_bias

if os.environ.get("NVFP4_FP4_AG", "0") == "1":
    # Expert weights gathered as NVFP4 (see nvfp4_fp4_ag). Installed before the other
    # start_param_sync wrappers so its local cast runs innermost, right before the gather.
    from transformer_engine.pytorch.ops.fused import grouped_mlp as _te_grouped_mlp_fp4

    _te_grouped_mlp_fp4._v7_module("nvfp4_fp4_ag").install()

if os.environ.get("NVFP4_PREFETCH_PARAM_AG", "0") == "1":
    # Megatron dispatches a bucket group's param all-gather only when the previous group
    # is consumed, so the compute stream stalls wherever one layer's forward is shorter
    # than the next gather. Each gather writes its own slice of the param buffer, so the
    # whole chain can be queued behind the first one.
    import warnings

    from megatron.core.distributed import param_and_grad_buffer as _mc_pgb

    _orig_start_param_sync = _mc_pgb._ParamAndGradBucketGroup.start_param_sync

    def _start_param_sync_prefetch(self, force_sync: bool = False):
        _orig_start_param_sync(self, force_sync)
        if force_sync or not self.ddp_config.overlap_param_gather:
            return
        group = self.next_param_gather_bucket_group
        while group is not None and not group.param_gather_dispatched:
            _orig_start_param_sync(group)
            group = group.next_param_gather_bucket_group

    _mc_pgb._ParamAndGradBucketGroup.start_param_sync = _start_param_sync_prefetch
    warnings.filterwarnings("ignore", message="The next bucket's parameter all-gather")

_early_ag_active = [False]

if os.environ.get("NVFP4_SHARD_WEIGHT_AMAX", "0") == "1":
    # Expert weight amaxes are reduced from the optimizer shards right before the first
    # param all-gather after each optimizer step, so the grouped MLP cast skips its own
    # zero_amax + amax pass over every gathered weight.
    from megatron.core.distributed import param_and_grad_buffer as _mc_pgb_amax
    from megatron.core.optimizer import distrib_optimizer as _mc_distopt
    from transformer_engine.pytorch.ops.fused import nvfp4_shard_weight_amax as _shard_amax

    _amax_bucket_groups = []
    _orig_bucket_group_init = _mc_pgb_amax._ParamAndGradBucketGroup.__init__

    def _bucket_group_init_track(self, *args, **kwargs):
        _orig_bucket_group_init(self, *args, **kwargs)
        _amax_bucket_groups.append(self)

    _mc_pgb_amax._ParamAndGradBucketGroup.__init__ = _bucket_group_init_track

    _orig_start_param_sync_amax = _mc_pgb_amax._ParamAndGradBucketGroup.start_param_sync

    _amax_trace = os.environ.get("NVFP4_SHARD_WEIGHT_AMAX_CHECK", "0") != "0"

    def _start_param_sync_amax(self, force_sync: bool = False):
        before = _shard_amax.counters["refresh"]
        if not _early_ag_active[0]:
            _shard_amax.refresh(_amax_bucket_groups)
        if _amax_trace and _shard_amax.counters["refresh"] != before and before < 8:
            sys.stderr.write(
                f"NVFP4_SHARD_WEIGHT_AMAX refresh rank{torch.distributed.get_rank()} "
                f"{_shard_amax.counters} force_sync={force_sync}\n"
            )
            sys.stderr.flush()
        return _orig_start_param_sync_amax(self, force_sync)

    _mc_pgb_amax._ParamAndGradBucketGroup.start_param_sync = _start_param_sync_amax

    _amax_check_left = [int(os.environ.get("NVFP4_SHARD_WEIGHT_AMAX_CHECK", "0"))]
    _orig_step_with_ready_grads = _mc_distopt.DistributedOptimizer.step_with_ready_grads

    _opt_stack_left = [1 if os.environ.get("NVFP4_TRACE_OPT_STACK", "0") == "1" else 0]

    def _step_with_ready_grads_amax(self, *args, **kwargs):
        if _opt_stack_left[0] > 0 and torch.distributed.get_rank() == 0:
            import traceback

            _opt_stack_left[0] -= 1
            sys.stderr.write("NVFP4_TRACE_OPT_STACK\n" + "".join(traceback.format_stack(limit=12)))
        if _amax_check_left[0] > 0 and _shard_amax._slots:
            _amax_check_left[0] -= 1
            bad = 0
            for p, slot in _shard_amax._slots.items():
                ref = p.detach().abs().amax().float()
                if not torch.equal(ref.view(1), _shard_amax._amax_buf[slot : slot + 1]):
                    bad += 1
            sys.stderr.write(
                f"NVFP4_SHARD_WEIGHT_AMAX_CHECK rank{torch.distributed.get_rank()}: "
                f"{bad}/{len(_shard_amax._slots)} mismatched {_shard_amax.counters}\n"
            )
            sys.stderr.flush()
        result = _orig_step_with_ready_grads(self, *args, **kwargs)
        if not self.__dict__.pop("_nvfp4_amax_refreshed", False):
            _shard_amax.mark_dirty()
        return result

    _mc_distopt.DistributedOptimizer.step_with_ready_grads = _step_with_ready_grads_amax

if int(os.environ.get("NVFP4_PHASED_ADAM_AG", "0")) > 0:
    # With EP1 one DistributedOptimizer owns every bucket, so the first param all-gathers
    # of the next forward (embedding bucket first) can only start after all of Adam. Run
    # Adam in phases along the forward bucket order instead: after the params of each of
    # the first N bucket groups are updated, queue that group's all-gather, then update
    # the rest. Adam is elementwise, so the result is unchanged.
    from megatron.core.distributed import param_and_grad_buffer as _mc_pgb_ph
    from megatron.core.optimizer import distrib_optimizer as _mc_distopt_ph

    _phased_n = int(os.environ["NVFP4_PHASED_ADAM_AG"])
    _base_start_param_sync = globals().get("_orig_start_param_sync") or (
        _mc_pgb_ph._ParamAndGradBucketGroup.start_param_sync
    )
    _phased_amax = os.environ.get("NVFP4_SHARD_WEIGHT_AMAX", "0") == "1"
    if _phased_amax:
        # The explicit refresh below runs while the early gathers are in flight on the DP
        # communicator; its all-reduce needs its own communicator to not queue behind them.
        _shard_amax.use_side_group = True

    def _phase_param_sets(opt, chain, key="_nvfp4_phase_sets"):
        sets = opt.__dict__.get(key)
        if sets is not None:
            return sets
        model_to_shard = {}
        for model_group, shard_group in zip(opt.model_float16_groups, opt.shard_float16_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is not None:
                    model_to_shard[model_param] = shard_param
        for model_group, shard_group in zip(opt.model_fp32_groups, opt.shard_fp32_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                model_to_shard[model_param] = shard_param
        sets = []
        for group in chain:
            ids = set()
            for bucket in group.buckets:
                for param in bucket.params:
                    shard_param = model_to_shard.get(param)
                    if shard_param is not None:
                        ids.add(id(shard_param))
            sets.append(ids)
        opt.__dict__[key] = sets
        sys.stderr.write(
            f"NVFP4_PHASED_ADAM_AG rank{torch.distributed.get_rank()}: {key} phases "
            f"{[len(s) for s in sets]} of {sum(len(g['params']) for g in opt.optimizer.param_groups)} "
            f"optimizer params\n"
        )
        return sets

    # NVFP4_SIDE_ADAM=1: after the first N phases, step every remaining bucket group on a side
    # stream in forward order (dense groups, then expert groups), refreshing the expert weight
    # amaxes of each phase after its update, and record an event per phase. The all-gathers
    # stay lazy: when the forward dispatches one of these groups, it is queued on its own
    # stream behind only that group's event, so the forward waits for the groups it reaches.
    # All the gathers go through one communicator in order, so queueing them at step time
    # would put the next collective on it behind every gather. NVFP4_SIDE_ADAM_CTAS caps the
    # grid of the side-stream Adam kernels so they leave SMs to the forward. The next
    # grad-buffer zeroing waits for the side stream.
    _side_adam = os.environ.get("NVFP4_SIDE_ADAM", "0") == "1"
    _side_phases = max(1, int(os.environ.get("NVFP4_SIDE_ADAM_PHASES", "4")))
    _side_ctas = int(os.environ.get("NVFP4_SIDE_ADAM_CTAS", "0"))
    # NVFP4_ADAM_AMAX=1: the fused clip+Adam kernel max-reduces each updated expert weight
    # shard into its amax slot, so the refresh after the step is only the MAX all-reduce.
    # Any slot whose shard no fused launch covered is recomputed from the shards.
    _adam_amax = os.environ.get("NVFP4_ADAM_AMAX", "0") == "1"
    _side_state = {"stream": None, "ag_stream": None, "event": None, "pending": {}}

    def _forward_chain(first):
        chain = []
        while first is not None:
            chain.append(first)
            first = first.next_param_gather_bucket_group
        return chain

    def _install_phased_adam(opt):
        fused = opt.optimizer
        orig_step = fused.step

        def phased_step(closure=None, grad_scaler=None):
            chunks = opt.model_chunks
            if (
                closure is not None
                or grad_scaler is not None
                or len(chunks) != 1
                or not getattr(chunks[0], "remove_forward_pre_hook_handles", None)
                or not opt.ddp_config.overlap_param_gather
                or getattr(fused, "capturable", False)
                or not chunks[0].bucket_groups
            ):
                return orig_step(closure=closure, grad_scaler=grad_scaler)
            chain = []
            group = chunks[0].bucket_groups[-1]
            while group is not None and len(chain) < _phased_n:
                chain.append(group)
                group = group.next_param_gather_bucket_group
            if any(g.param_gather_dispatched or g.param_gather_handle is not None for g in chain):
                return orig_step()
            sets = _phase_param_sets(opt, chain)
            orig_groups = fused.param_groups
            new_steps = [None] * len(orig_groups)

            def run(keep):
                subs = [dict(g, params=[p for p in g["params"] if keep(p)]) for g in orig_groups]
                fused.param_groups = subs
                try:
                    orig_step()
                finally:
                    fused.param_groups = orig_groups
                for i, sub in enumerate(subs):
                    if sub["params"] and "step" in sub:
                        new_steps[i] = sub["step"]

            side = _side_adam and not torch.cuda.is_current_stream_capturing()
            clip = globals().get("_clip_adam")
            fold = (
                _adam_amax
                and _phased_amax
                and not side
                and clip is not None
                and _shard_amax.begin_fold(_amax_bucket_groups)
            )
            if fold:
                clip.set_amax_target(_shard_amax._amax_buf, _shard_amax.slot_lookup())
            done = set()
            for bucket_group, ids in zip(chain, sets):
                run(lambda p, ids=ids: id(p) in ids)
                done |= ids
                _early_ag_active[0] = True
                try:
                    _base_start_param_sync(bucket_group)
                finally:
                    _early_ag_active[0] = False
            if side:
                if _phased_amax:
                    # The forward quantizes these weights after this point on this stream.
                    _shard_amax.refresh_subset(_amax_bucket_groups, chain)
                _side_rest(opt, chunks[0], chain, run, done)
            else:
                run(lambda p: id(p) not in done)
                if fold:
                    _shard_amax.finish_fold(clip.clear_amax_target())
                    opt.__dict__["_nvfp4_amax_refreshed"] = True
                elif _phased_amax:
                    _shard_amax.mark_dirty()
                    _shard_amax.refresh(_amax_bucket_groups)
                    opt.__dict__["_nvfp4_amax_refreshed"] = True
            for g, step in zip(orig_groups, new_steps):
                if step is not None:
                    g["step"] = step
            return None

        def _side_rest(opt, chunk, chain, run, done):
            rest = [g for g in _forward_chain(chunk.bucket_groups[-1]) if g not in chain]
            if chunk.expert_parallel_bucket_groups:
                rest += _forward_chain(chunk.expert_parallel_bucket_groups[-1])
            sets = _phase_param_sets(opt, rest, key="_nvfp4_side_phase_sets")
            if _side_state["stream"] is None:
                _side_state["stream"] = torch.cuda.Stream()
                _side_state["ag_stream"] = torch.cuda.Stream()
            side = _side_state["stream"]
            side.wait_stream(torch.cuda.current_stream())
            clip = globals().get("_clip_adam")
            scale = clip._pending["scale"] if clip is not None else None
            if isinstance(scale, torch.Tensor):
                scale.record_stream(side)
            # Main grads are views of a few persistent grad buffers; one view per buffer is
            # enough to keep them from being reused while the side stream reads them.
            grad_views = _side_state.get("grad_views")
            if grad_views is None:
                storages = {}
                for group in fused.param_groups:
                    for param in group["params"]:
                        grad = getattr(param, "decoupled_grad", None)
                        if grad is None:
                            grad = param.grad
                        if grad is not None:
                            storages.setdefault(grad.untyped_storage().data_ptr(), grad)
                grad_views = _side_state["grad_views"] = list(storages.values())
            for grad in grad_views:
                grad.record_stream(side)
            # Each FusedAdam call costs host time before the next forward can be queued, so
            # consecutive groups are stepped together in _side_phases calls.
            n = len(rest)
            bounds = [round(i * n / _side_phases) for i in range(_side_phases + 1)]
            pending = _side_state["pending"]
            pending.clear()
            capped = _side_ctas > 0 and clip is not None and clip.set_max_ctas(_side_ctas)
            try:
                with torch.cuda.stream(side):
                    for lo, hi in zip(bounds[:-1], bounds[1:]):
                        if lo == hi:
                            continue
                        ids = set().union(*sets[lo:hi]) - done
                        if ids:
                            run(lambda p, ids=ids: id(p) in ids)
                            done |= ids
                        if _phased_amax:
                            _shard_amax.refresh_subset(_amax_bucket_groups, rest[lo:hi])
                        event = torch.cuda.Event()
                        event.record()
                        for bucket_group in rest[lo:hi]:
                            if not bucket_group.param_gather_dispatched:
                                pending[bucket_group] = event
                    run(lambda p: id(p) not in done)
                    event = torch.cuda.Event()
                    event.record()
            finally:
                if capped:
                    clip.set_max_ctas(0)
            if _phased_amax:
                opt.__dict__["_nvfp4_amax_refreshed"] = True
            _side_state["event"] = event

        fused.step = phased_step

    _orig_step_with_ready_grads_ph = _mc_distopt_ph.DistributedOptimizer.step_with_ready_grads

    def _step_with_ready_grads_phased(self, *args, **kwargs):
        if "_nvfp4_phased" not in self.__dict__:
            self.__dict__["_nvfp4_phased"] = True
            _install_phased_adam(self)
        return _orig_step_with_ready_grads_ph(self, *args, **kwargs)

    _mc_distopt_ph.DistributedOptimizer.step_with_ready_grads = _step_with_ready_grads_phased

    if _side_adam:
        from megatron.core.distributed import distributed_data_parallel as _mc_ddp_side

        _orig_zero_grad_buffer_side = _mc_ddp_side.DistributedDataParallel.zero_grad_buffer

        def _zero_grad_buffer_after_side_adam(self):
            event = _side_state["event"]
            if event is not None:
                async_zero = os.environ.get("NVFP4_ASYNC_GRAD_ZERO", "0") == "1"
                if async_zero and not torch.cuda.is_current_stream_capturing():
                    if getattr(self, "_grad_zero_stream", None) is None:
                        self._grad_zero_stream = torch.cuda.Stream()
                    self._grad_zero_stream.wait_event(event)
                else:
                    torch.cuda.current_stream().wait_event(event)
            return _orig_zero_grad_buffer_side(self)

        _mc_ddp_side.DistributedDataParallel.zero_grad_buffer = _zero_grad_buffer_after_side_adam

        def _gate_param_sync(start):
            def start_param_sync_after_side_adam(self, force_sync: bool = False):
                event = _side_state["pending"].pop(self, None)
                if event is None:
                    return start(self, force_sync)
                current = torch.cuda.current_stream()
                if force_sync or not self.ddp_config.overlap_param_gather:
                    current.wait_event(event)
                    return start(self, force_sync)
                ag_stream = _side_state["ag_stream"]
                ag_stream.wait_stream(current)
                ag_stream.wait_event(event)
                with torch.cuda.stream(ag_stream):
                    return start(self, force_sync)

            return start_param_sync_after_side_adam

        # The prefetch wrapper dispatches the following groups through _orig_start_param_sync
        # itself, so the gate goes under it.
        if "_orig_start_param_sync" in globals():
            _orig_start_param_sync = _gate_param_sync(_orig_start_param_sync)
        else:
            _mc_pgb_ph._ParamAndGradBucketGroup.start_param_sync = _gate_param_sync(
                _mc_pgb_ph._ParamAndGradBucketGroup.start_param_sync
            )

if os.environ.get("NVFP4_CLIP_IN_ADAM", "0") == "1":
    # The grad-norm clip coefficient is applied by the Adam kernel as it loads each
    # gradient, instead of a separate multi_tensor_scale pass over all main grads.
    import nvfp4_clip_adam as _clip_adam
    from megatron.core.optimizer import clip_grads as _mc_clip_ca
    from megatron.core.optimizer import distrib_optimizer as _mc_distopt_ca
    from megatron.core.optimizer import optimizer as _mc_opt_ca

    # TE's multi_tensor_adam is built with --use_fast_math; only the matching build of
    # the extension reproduces its results bit for bit.
    _clip_adam.load_prebuilt(
        os.environ.get("NVFP4_CLIP_ADAM_EXT", "/opt/nvfp4_ext/nvfp4_clip_adam_ext_fm9.so")
    )
    _clip_check = [int(os.environ.get("NVFP4_CLIP_IN_ADAM_CHECK", "0"))]
    _orig_clip_by_norm = _mc_clip_ca.clip_grad_by_total_norm_fp32

    def _clip_by_norm_deferred(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if not isinstance(total_norm, torch.Tensor):
            return _orig_clip_by_norm(parameters, max_norm, total_norm, use_decoupled_grad)
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # The precision-aware optimizer keeps main grads in `.decoupled_grad`, which is
        # also what FusedAdam reads.
        attr = "decoupled_grad" if use_decoupled_grad else "grad"
        grads = [
            getattr(p, attr).detach() for p in parameters if getattr(p, attr, None) is not None
        ]
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        clip_coeff.clamp_max_(1.0)
        if not _clip_adam.defer_clip(grads, clip_coeff):
            return _orig_clip_by_norm(parameters, max_norm, total_norm, use_decoupled_grad)

    _mc_clip_ca.clip_grad_by_total_norm_fp32 = _clip_by_norm_deferred
    _mc_opt_ca.clip_grad_by_total_norm_fp32 = _clip_by_norm_deferred

    _orig_step_with_ready_grads_ca = _mc_distopt_ca.DistributedOptimizer.step_with_ready_grads

    def _step_with_ready_grads_clip(self, *args, **kwargs):
        _clip_adam.install(self.optimizer)
        try:
            return _orig_step_with_ready_grads_ca(self, *args, **kwargs)
        finally:
            pending = len(_clip_adam._pending["grads"])
            _clip_adam.finish()
            if _clip_check[0] > 0:
                _clip_check[0] -= 1
                sys.stderr.write(
                    f"NVFP4_CLIP_IN_ADAM rank{torch.distributed.get_rank()}: "
                    f"leftover_before_finish={pending} {_clip_adam.counters}\n"
                )

    _mc_distopt_ca.DistributedOptimizer.step_with_ready_grads = _step_with_ready_grads_clip

if os.environ.get("NVFP4_FAST_L2NORM", "0") == "1":
    # multi_tensor_l2norm launches once per 320 chunks; the grad norm over all main-grad
    # shards takes ~130 short launches. The extension computes every chunk sum in one
    # launch and adds them up in the same order, so the norm is bitwise the same.
    import nvfp4_clip_adam as _clip_adam_l2
    from megatron.core.optimizer import clip_grads as _mc_clip_l2

    if _clip_adam_l2._ext is None:
        _clip_adam_l2.load_prebuilt(
            os.environ.get("NVFP4_CLIP_ADAM_EXT", "/opt/nvfp4_ext/nvfp4_clip_adam_ext_fm9.so")
        )
    _mc_clip_l2.l2_norm_impl = _clip_adam_l2.wrap_l2norm(_mc_clip_l2.l2_norm_impl)

if os.environ.get("NVFP4_FAST_L2NORM", "0") == "1" and os.environ.get("NVFP4_EARLY_L2NORM", "0") == "1":
    # The last dense bucket group (embedding) finishes its reduce-scatter well after all the
    # others, and the grad norm waits for it. Wait on every other group first and compute
    # the norm's chunk sums inside their grad buffers right away, so only the last group's
    # chunks are left once its reduce-scatter lands.
    from megatron.core.distributed import distributed_data_parallel as _mc_ddp_l2

    _orig_ddp_finish_grad_sync = _mc_ddp_l2.DistributedDataParallel.finish_grad_sync

    def _group_grad_ranges(group):
        ranges = group.__dict__.get("_nvfp4_grad_ranges")
        if ranges is None:
            ranges = [
                (b.grad_data.data_ptr(),
                 b.grad_data.data_ptr() + b.grad_data.numel() * b.grad_data.element_size())
                for b in group.buckets
            ]
            group.__dict__["_nvfp4_grad_ranges"] = ranges
        return ranges

    def _ddp_finish_grad_sync_early_l2(self, force_all_reduce=False):
        groups = self.bucket_groups + self.expert_parallel_bucket_groups
        if (
            force_all_reduce
            or not self.ddp_config.overlap_grad_reduce
            or not self.bucket_groups
            or torch.cuda.is_current_stream_capturing()
        ):
            return _orig_ddp_finish_grad_sync(self, force_all_reduce)
        flush = getattr(self, "_nvfp4_flush_grad_acc", None)
        if flush is not None:
            flush()
        last = self.bucket_groups[-1]
        _clip_adam_l2.l2_prefetch_begin()
        for group in groups:
            if group is not last:
                group.finish_grad_sync(force_all_reduce=force_all_reduce)
                _clip_adam_l2.l2_prefetch(id(group), _group_grad_ranges(group))
        last.finish_grad_sync(force_all_reduce=force_all_reduce)

    _mc_ddp_l2.DistributedDataParallel.finish_grad_sync = _ddp_finish_grad_sync_early_l2
    _clip_adam_l2.l2_check["left"] = int(os.environ.get("NVFP4_EARLY_L2NORM_CHECK", "0"))
    _clip_adam_l2.l2_check["log"] = lambda msg: (
        sys.stderr.write(f"NVFP4_EARLY_L2NORM rank{torch.distributed.get_rank()}: {msg}\n"),
        sys.stderr.flush(),
    )

if os.environ.get("NVFP4_OVERLAP_WEIGHT_QUANT", "0") == "1":
    # Expert weights do not depend on the layer input, so their NVFP4 casts are issued on
    # a side stream at the start of the layer and overlap attention; the grouped MLP
    # joins that stream before its first GEMM.
    from megatron.core.transformer import transformer_layer as _mc_tl_pq
    from transformer_engine.pytorch.ops.fused import grouped_mlp as _te_grouped_mlp

    _orig_forward_attention = _mc_tl_pq.TransformerLayer._forward_attention

    def _find_prequant_fc1(layer):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        fused = getattr(experts, "_fused_ops", None)
        if not fused:
            return None
        for module in fused[0].modules():
            if "_nvfp4_prequant_peer" in module.__dict__:
                return module
        return None

    # The casts read the expert weights before the experts' own forward pre-hook runs, so
    # their bucket groups' param all-gathers are published here first.
    _prequant_wait_ag = os.environ.get("NVFP4_PREQUANT_WAIT_AG", "0")
    _prequant_ag_stats = {"calls": 0, "inflight": 0, "groups": 0}
    _prequant_ag_log_left = [int(os.environ.get("NVFP4_PREQUANT_WAIT_AG_LOG", "0"))]

    def _prequant_ready_callbacks(fc1):
        cached = fc1.__dict__.get("_nvfp4_prequant_ready")
        if cached is not None:
            return cached
        from megatron.core.utils import PARAM_READY_CALLBACK_ATTR

        fc2, num_groups = fc1.__dict__["_nvfp4_prequant_peer"]
        callbacks = {}
        for op in (fc1, fc2):
            for idx in range(num_groups):
                cb = getattr(getattr(op, f"weight{idx}"), PARAM_READY_CALLBACK_ATTR, None)
                if cb is not None:
                    callbacks[id(cb)] = cb
        cached = list(callbacks.values())
        fc1.__dict__["_nvfp4_prequant_ready"] = cached
        return cached

    def _prequant_publish(fc1):
        if torch.cuda.is_current_stream_capturing():
            return
        callbacks = _prequant_ready_callbacks(fc1)
        inflight = 0
        for cb in callbacks:
            group = cb._bucket_group()
            if group is not None and (
                group.param_gather_handle is not None or not group.param_gather_dispatched
            ):
                inflight += 1
        _prequant_ag_stats["calls"] += 1
        _prequant_ag_stats["groups"] += len(callbacks)
        _prequant_ag_stats["inflight"] += inflight
        if _prequant_wait_ag == "1":
            for cb in callbacks:
                cb()
        if _prequant_ag_log_left[0] > 0 and _prequant_ag_stats["calls"] % 24 == 0:
            _prequant_ag_log_left[0] -= 1
            sys.stderr.write(
                f"NVFP4_PREQUANT_WAIT_AG rank{torch.distributed.get_rank()} "
                f"mode={_prequant_wait_ag} {_prequant_ag_stats}\n"
            )

    def _forward_attention_prequant(self, *args, **kwargs):
        if torch.is_grad_enabled():
            fc1 = self.__dict__.get("_nvfp4_prequant_fc1")
            if fc1 is None:
                fc1 = _find_prequant_fc1(self)
                if fc1 is not None:
                    self.__dict__["_nvfp4_prequant_fc1"] = fc1
            if fc1 is not None:
                if _prequant_wait_ag != "0":
                    _prequant_publish(fc1)
                _te_grouped_mlp.launch_expert_weight_prequant(fc1, True)
        return _orig_forward_attention(self, *args, **kwargs)

    _mc_tl_pq.TransformerLayer._forward_attention = _forward_attention_prequant

if os.environ.get("NVFP4_EARLY_DENSE_AG", "0") == "1":
    # The dense optimizer (embedding, attention, norms) steps first and finishes long
    # before the expert optimizer. Queue its param all-gathers right away so they overlap
    # the expert Adam instead of stalling the next forward at the embedding lookup.
    from megatron.core.optimizer import distrib_optimizer as _mc_distopt_ag
    from megatron.core.optimizer.optimizer import ChainedOptimizer as _ChainedOptimizer

    def _own_bucket_groups(opt):
        own = {id(b) for buf in getattr(opt, "buffers", []) for b in buf.buckets}
        groups = []
        for chunk in opt.model_chunks:
            if not getattr(chunk, "remove_forward_pre_hook_handles", None):
                return []
            for group in chunk.bucket_groups + chunk.expert_parallel_bucket_groups:
                if group.buckets and id(group.buckets[0]) in own:
                    groups.append(group)
        return groups

    def _early_ag_diag(idx, opt):
        chunks = getattr(opt, "model_chunks", [])
        own = {id(b) for buf in getattr(opt, "buffers", []) for b in buf.buckets}
        return (
            f"opt{idx} {type(opt).__name__} buffers={len(getattr(opt, 'buffers', []))} "
            f"own_buckets={len(own)} chunks={len(chunks)} "
            f"hooks={[bool(getattr(c, 'remove_forward_pre_hook_handles', None)) for c in chunks]} "
            f"groups={[(len(c.bucket_groups), len(c.expert_parallel_bucket_groups)) for c in chunks]} "
            f"overlap={getattr(getattr(opt, 'ddp_config', None), 'overlap_param_gather', None)}"
        )

    _early_ag_diag_left = [3]

    def _chained_step_early_ag(self):
        success = True
        last = len(self.chained_optimizers) - 1
        if _early_ag_diag_left[0] > 0:
            _early_ag_diag_left[0] -= 1
            sys.stderr.write(
                f"NVFP4_EARLY_DENSE_AG diag rank{torch.distributed.get_rank()}: "
                + " | ".join(_early_ag_diag(i, o) for i, o in enumerate(self.chained_optimizers))
                + "\n"
            )
        for idx, opt in enumerate(self.chained_optimizers):
            success &= opt.step_with_ready_grads()
            if (
                idx == last
                or not success
                or not isinstance(opt, _mc_distopt_ag.DistributedOptimizer)
                or not opt.ddp_config.overlap_param_gather
            ):
                continue
            _early_ag_active[0] = True
            try:
                groups = _own_bucket_groups(opt)
                for group in groups:
                    if not group.param_gather_dispatched:
                        group.start_param_sync()
            finally:
                _early_ag_active[0] = False
            if groups and not _early_ag_logged[0]:
                _early_ag_logged[0] = True
                n_expert = sum(
                    g in c.expert_parallel_bucket_groups for c in opt.model_chunks for g in groups
                )
                sys.stderr.write(
                    f"NVFP4_EARLY_DENSE_AG rank{torch.distributed.get_rank()}: optimizer {idx} "
                    f"dispatched {len(groups)} bucket groups ({n_expert} expert)\n"
                )
        return success

    _early_ag_logged = [False]

    _ChainedOptimizer._step = _chained_step_early_ag
    sys.stderr.write("NVFP4_EARLY_DENSE_AG installed on ChainedOptimizer._step\n")

if os.environ.get("NVFP4_GRAPH_CONST_MASK", "0") == "1":
    # With no padding mask, Megatron builds a fresh all-false [mbs, 1, s, s] mask for every
    # TE graph replay and TE copies it into the layer's static input. Pass one cached mask
    # tagged `_te_graph_const` so the graph input copy can be skipped after the first replay.
    from megatron.core.transformer import transformer_layer as _mc_tl

    _orig_replay_args = _mc_tl.TransformerLayer._get_te_cuda_graph_replay_args
    _const_masks = {}

    def _replay_args_const_mask(self, *args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if "attention_mask" in kwargs and kwargs["attention_mask"] is None and hidden_states is not None:
            cfg = self.config
            slen_per_cp = hidden_states.size(0)
            if cfg.sequence_parallel:
                slen_per_cp *= cfg.tensor_model_parallel_size
            shape = (
                hidden_states.size(1),
                1,
                slen_per_cp,
                slen_per_cp * cfg.context_parallel_size,
            )
            mask = _const_masks.get(shape)
            if mask is None:
                mask = torch.zeros(shape, dtype=torch.bool, device=torch.cuda.current_device())
                mask._te_graph_const = True
                _const_masks[shape] = mask
            kwargs["attention_mask"] = mask
        return _orig_replay_args(self, *args, **kwargs)

    _mc_tl.TransformerLayer._get_te_cuda_graph_replay_args = _replay_args_const_mask

if os.environ.get("NVFP4_TRITON_ADAM", "0") == "1":
    # Single-launch Adam for the BF16 param + int16 remainder path with the grad
    # clip coefficient applied in-kernel (no separate multi_tensor_scale pass).
    from transformer_engine.pytorch.optimizers.nvfp4_triton_adam import (
        install as _install_triton_adam,
    )

    _install_triton_adam()

if os.environ.get("NVFP4_TRACE_COPIES", "0") == "1":
    # Diagnostic: attribute large FP32 Tensor.copy_ calls to Python call sites.
    import collections
    import traceback

    _orig_copy = torch.Tensor.copy_
    _copy_sites = collections.Counter()
    _copy_calls = [0]

    def _traced_copy(self, src, *args, **kwargs):
        if (
            self.numel() >= 1 << 20
            and torch.float32 in (self.dtype, getattr(src, "dtype", None))
            and not torch.cuda.is_current_stream_capturing()
        ):
            frames = traceback.extract_stack(limit=7)[:-1]
            site = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in frames[::-1])
            _copy_sites[(site, self.dtype, getattr(src, "dtype", None), tuple(self.shape))] += 1
            _copy_calls[0] += 1
            if _copy_calls[0] in (2000, 20000):
                if int(os.environ.get("RANK", "0")) == 0:
                    for key, n in _copy_sites.most_common(15):
                        print(f"NVFP4 copy site n={n} {key}", flush=True)
        return _orig_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = _traced_copy

OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
OmegaConf.register_new_resolver("ceil_div", lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)
OmegaConf.register_new_resolver("div", lambda x, y: x / y)
OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("lt", lambda x, y: x < y)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)
OmegaConf.register_new_resolver("neq", lambda x, y: x != y)
OmegaConf.register_new_resolver("or", lambda *args: any(args))
OmegaConf.register_new_resolver("min", lambda x, y: min(x, y))
OmegaConf.register_new_resolver("floor", lambda x: int(x // 1))


def _assert_finite_tree(value, label):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"NVFP4 first non-finite training boundary: {label}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")


def forward_step(state, data_iterator, model, return_schedule_plan=False):
    output, loss_function = _bridge_forward_step(
        state,
        data_iterator,
        model,
        return_schedule_plan=return_schedule_plan,
    )
    if os.getenv("NVFP4_DEBUG_LOSS_FINITE", "0") != "1":
        return output, loss_function
    _assert_finite_tree(output, "model_output")

    def checked_loss_function(*loss_args, **loss_kwargs):
        result = loss_function(*loss_args, **loss_kwargs)
        _assert_finite_tree(result, "masked_next_token_loss")
        return result

    return output, checked_loss_function


def get_data(config):
    if config.model.data.mock_dataset:
        return create_mock_dataset_config(config)

    r = [6]
    train_datasets = [f"/preproc_data/c4-train.en_{idx}_text_document" for idx in r]
    train_datasets_weights = [50] * len(r)
    val_test_path = "/preproc_data/c4-validation-91205-samples.en_text_document"

    data_paths = [(train_datasets, train_datasets_weights), ([val_test_path], None), ([val_test_path], None)]

    return GPTDatasetConfig(
        blend_per_split=data_paths,
        sequence_length=config.model.encoder_seq_length,
        random_seed=config.model.seed,
        dataloader_type="single",
        num_workers=config.model.data.num_workers,
        path_to_cache="/npy_index",
        defer_npy_index_mmap=True,
        fast_cache_load=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=True,
        create_attention_mask=False,
    )


def evaluate_on_fixed_validation_window(eval_samples):
    """Build the validation loader over the first eval_samples samples, sequentially, so every evaluation scores the same samples."""
    from megatron.bridge.data import loaders

    build_pretraining_data_loader = loaders.build_pretraining_data_loader

    def build_fixed_window_validation_loader(dataset, consumed_samples, dataloader_type, *args, **kwargs):
        # Bridge requests the "cyclic" sampler only for the validation loader.
        if dataloader_type != "cyclic":
            return build_pretraining_data_loader(dataset, consumed_samples, dataloader_type, *args, **kwargs)
        return build_pretraining_data_loader(Subset(dataset, range(eval_samples)), 0, "single", *args, **kwargs)

    loaders.build_pretraining_data_loader = build_fixed_window_validation_loader


def get_model(config):
    tp = config.model.tensor_model_parallel_size
    ep = config.model.expert_model_parallel_size
    pp = config.model.pipeline_model_parallel_size
    pp_dtype = torch.bfloat16 if config.model.pipeline_model_parallel_size != 1 else None
    vp = config.model.virtual_pipeline_model_parallel_size
    cp = config.model.context_parallel_size
    sp = config.model.sequence_parallel

    # WAR for Megatron constraint: ETP * EP * PP must equal TP * CP * PP
    # for DP-last mapping compatibility.
    etp = tp * cp // ep
    etp = 1
    ep = int(os.getenv("EXPERT_PARALLEL", 1))

    asym_pp_embed = config.model.account_for_embedding_in_pipeline_split
    asym_pp_loss = config.model.account_for_loss_in_pipeline_split

    if config.model.overwritten_attributes.num_layers is not None:
        num_layers = config.model.overwritten_attributes.num_layers
    else:
        num_layers = 24

    cuda_graph_scope = config.model.overwritten_attributes.cuda_graph_scope
    if OmegaConf.is_list(cuda_graph_scope):
        cuda_graph_scope = list(cuda_graph_scope)

    if config.model.clamped_quick_gelu_activation:
        activation_func, activation_func_clamp_value, glu_linear_offset = quick_gelu, 7.0, 1.0
    else:
        activation_func, activation_func_clamp_value, glu_linear_offset = F.silu, None, 0.0

    model_cfg = GPTModelProvider(
        # GPT-OSS 20B architecture
        num_layers=num_layers,
        num_moe_experts=32,
        # GPT-OSS base defaults
        hidden_size=2880,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=2880,
        kv_channels=64,
        normalization="RMSNorm",
        layernorm_epsilon=config.model.rmsnorm_eps,
        init_method_std=0.008,
        gated_linear_unit=True,
        add_bias_linear=config.model.add_bias_linear,
        share_embeddings_and_output_weights=False,
        vocab_size=128256,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        bias_activation_fusion=True,
        bias_dropout_fusion=os.environ.get("NVFP4_BIAS_DROPOUT_FUSION", "0") == "1",
        activation_func_clamp_value=activation_func_clamp_value,
        glu_linear_offset=glu_linear_offset,
        window_attn_skip_freq=2,
        # position embeddings
        rotary_base=150000,
        # MoE settings
        moe_router_topk=config.model.moe_router_topk,
        moe_token_dispatcher_type=config.model.moe_token_dispatcher_type,
        moe_ffn_hidden_size=2880,
        moe_router_load_balancing_type=config.model.moe_router_load_balancing_type,
        window_size=(config.model.sliding_window_keys - 1, 0),
        softmax_type=config.model.attention_softmax_type,
        qk_layernorm=config.model.qk_layernorm,
        activation_func=activation_func,
        # Parallelism configuration
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        expert_tensor_parallel_size=etp,
        expert_model_parallel_size=ep,
        context_parallel_size=cp,
        sequence_parallel=sp,
        # CP comm
        cp_comm_type=config.model.cp_comm_type,
        # MoE
        moe_grouped_gemm=config.model.moe_grouped_gemm,
        moe_router_pre_softmax=config.model.moe_router_pre_softmax,
        moe_flex_dispatcher_backend=config.model.moe_flex_dispatcher_backend,
        moe_permute_fusion=config.model.moe_permute_fusion,
        moe_hybridep_num_sms=config.model.moe_hybridep_num_sms,
        moe_router_fusion=config.model.moe_router_fusion,
        moe_router_force_load_balancing=config.model.moe_router_force_load_balancing,
        # Pipeline configuration
        pipeline_dtype=pp_dtype,
        account_for_embedding_in_pipeline_split=asym_pp_embed,
        account_for_loss_in_pipeline_split=asym_pp_loss,
        # Model configuration
        seq_length=config.model.encoder_seq_length,
        init_model_with_meta_device=config.model.mcore_fsdp,
        position_embedding_type=config.model.position_embedding_type,
        # Optimization features
        cross_entropy_loss_fusion=config.model.cross_entropy_loss_fusion,
        cross_entropy_fusion_impl=config.model.cross_entropy_fusion_impl,
        cuda_graph_impl=config.model.overwritten_attributes.cuda_graph_implementation,
        cuda_graph_scope=cuda_graph_scope,
        cuda_graph_use_single_mempool=(
            os.getenv("CUDA_GRAPH_USE_SINGLE_MEMPOOL", "1") == "1"
        ),
        fused_single_qkv_rope=config.model.fused_single_qkv_rope
        and (not config.model.qk_layernorm or _qknorm_rope.ACTIVE),
        apply_rope_fusion=config.model.qk_layernorm and not _qknorm_rope.ACTIVE,
        gradient_accumulation_fusion=config.model.gradient_accumulation_fusion,
        tp_only_amax_red=config.model.tp_only_amax_red,
        use_te_rng_tracker=config.model.use_te_rng_tracker,
        use_transformer_engine_op_fuser=config.model.use_transformer_engine_op_fuser,
        cuda_graph_warmup_steps=config.model.custom.cuda_graph_warmup_steps,
        # CPU offloading
        cpu_offloading=config.model.cpu_offloading,
        cpu_offloading_num_layers=config.model.cpu_offloading_num_layers,
        cpu_offloading_weights=False,
    )
    if config.model.use_transformer_engine_op_fuser:
        model_cfg.moe_mlp_glu_interleave_size = 32
    model_cfg.moe_router_padding_for_quantization = True
    if config.model.moe_expert_rank_capacity_factor is not None:
        model_cfg.moe_expert_rank_capacity_factor = config.model.moe_expert_rank_capacity_factor
    model_cfg.moe_paged_stash = config.model.moe_paged_stash
    if config.model.moe_paged_stash_buffer_size_factor_cuda is not None:
        model_cfg.moe_paged_stash_buffer_size_factor_cuda = float(
            config.model.moe_paged_stash_buffer_size_factor_cuda
        )
    model_cfg.moe_single_grouped_weight = config.model.moe_single_grouped_weight_bias
    model_cfg.moe_single_grouped_bias = config.model.moe_single_grouped_weight_bias
    model_cfg.moe_permute_fusion_into_hybridep = config.model.moe_permute_fusion_into_hybridep
    return model_cfg


def get_mixed_precision(config):
    # fp8 knobs:
    fp8_type = None
    if config.model.fp8:
        fp8_type = "hybrid" if config.model.fp8_hybrid else "e4m3"
    fp8_margin = 0
    fp8_amax_history_len = config.model.fp8_amax_history_len
    fp8_amax_compute_algo = config.model.fp8_amax_compute_algo
    fp8_param_gather = config.model.optim.fp8_param_gather
    fp8_recipe = config.model.fp8_recipe
    reuse_grad_buf_for_mxfp8_param_ag = config.model.reuse_grad_buf_for_mxfp8_param_ag
    # fp4 knobs:
    fp4_type = None
    if config.model.fp4:
        fp4_type = "e2m1"
    fp4_recipe = config.model.fp4_recipe

    first_last_layers_bf16 = config.model.first_last_layers_bf16
    num_layers_at_start_in_bf16 = config.model.num_layers_at_start_in_bf16
    num_layers_at_end_in_bf16 = config.model.num_layers_at_end_in_bf16

    if os.environ.get("NVFP4_PAO_BF16_MODEL_PARAMS", "0") == "1" and (
        fp8_param_gather or os.environ.get("NVFP4_FP4_PARAM_GATHER", "0") == "1"
    ):
        raise RuntimeError("NVFP4_PAO_BF16_MODEL_PARAMS requires BF16 params: disable fp8/fp4 param gather")

    if config.model.fp8 or config.model.fp4:
        mixed_precision = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
            # fp8
            fp8=fp8_type,
            fp8_margin=fp8_margin,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_param_gather=fp8_param_gather,
            reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buf_for_mxfp8_param_ag,
            fp8_dot_product_attention=config.model.fp8_dot_product_attention,
            # fp4
            fp4=fp4_type,
            fp4_recipe=fp4_recipe,
            fp4_param_gather=os.environ.get("NVFP4_FP4_PARAM_GATHER", "0") == "1",
            # First/last layers in bf16
            first_last_layers_bf16=first_last_layers_bf16,
            num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
        )
    else:
        mixed_precision = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

    return mixed_precision


def get_ddp_config(config):
    overlap_grad_reduce = config.model.optim.overlap_grad_reduce
    overlap_param_gather = config.model.optim.overlap_param_gather
    align_param_gather = config.model.optim.align_param_gather
    use_distributed_optimizer = config.model.optim.use_distributed_optimizer
    bucket_size = config.model.optim.bucket_size
    num_distributed_optimizer_instances = config.model.optim.num_distributed_optimizer_instances
    fp8_param_gather = config.model.optim.fp8_param_gather
    nccl_ub_dp = config.model.optim.nccl_ub_dp
    outer_dp_sharding_strategy = config.model.optim.outer_dp_sharding_strategy
    average_in_collective = config.model.optim.average_in_collective

    return DistributedDataParallelConfig(
        check_for_nan_in_grad=False,
        grad_reduce_in_fp32=False,
        average_in_collective=average_in_collective,
        bucket_size=bucket_size,
        # Overlap
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        align_param_gather=align_param_gather,
        # Distributed optimizer
        use_distributed_optimizer=use_distributed_optimizer,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
        data_parallel_sharding_strategy="optim_grads_params",
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        # FP8
        fp8_param_gather=fp8_param_gather,
        keep_fp8_transpose_cache=False,
        # FSDP
        fsdp_double_buffer=True,
        use_megatron_fsdp=config.model.mcore_fsdp,
        # NCCL
        nccl_ub=nccl_ub_dp,
    )


def get_dist_config(config):
    nccl_communicator_config_path = config.model.nccl_communicator_config_path
    return DistributedInitConfig(
        nccl_communicator_config_path=nccl_communicator_config_path,
        use_tp_pp_dp_mapping=config.model.use_tp_pp_dp_mapping,
        use_sharp=config.model.sharp,
    )


def get_optimizer_config(config):
    precision_aware = {}
    if os.environ.get("NVFP4_PRECISION_AWARE_OPT", "0") == "1":
        precision_aware["use_precision_aware_optimizer"] = True
        if os.environ.get("NVFP4_PAO_BF16_GRADS", "0") == "1":
            precision_aware["main_grads_dtype"] = torch.bfloat16
    opt_cfg = OptimizerConfig(
        **precision_aware,
        optimizer="adam",
        lr=config.model.optim.lr,
        weight_decay=0.1,
        bf16=config.trainer.precision == "bf16",
        fp16=config.trainer.precision == "fp16",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=config.model.optim.adam_eps,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        min_lr=config.model.optim.sched.min_lr,
    )
    opt_cfg.optimizer_cuda_graph = config.model.optim.optimizer_cuda_graph
    return opt_cfg


def get_scheduler_config(config):
    return SchedulerConfig(
        lr_decay_style="cosine",
        lr_decay_iters=int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) - int(config.model.optim.sched.lr_warmup_iters),
        lr_warmup_iters=config.model.optim.sched.lr_warmup_iters,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
    )


def get_train_config(config):
    return TrainingConfig(
        global_batch_size=config.model.global_batch_size,
        micro_batch_size=config.model.micro_batch_size,
        rampup_batch_size=None,
        train_iters=config.trainer.max_steps,
    )


def get_validation_config(config):
    return ValidationConfig(
        eval_interval=config.trainer.val_check_interval,
        eval_iters=config.trainer.eval_iters,
    )


def get_tokenizer_config(config):
    return TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=config.model.tokenizer.model,
        hf_tokenizer_kwargs={"use_fast": True},
    )


def get_rerun_state_machine_config(config):
    return RerunStateMachineConfig(
        rerun_mode="disabled",
        check_for_nan_in_loss=False,
        check_for_spiky_loss=False,
    )


def get_logger_config(config):
    return LoggerConfig(log_interval=config.trainer.max_steps + 1)


def get_rng_config(config):
    return RNGConfig(seed=config.model.seed, te_rng_tracker=config.model.use_te_rng_tracker)


def get_comm_overlap_config(config):
    tp_comm_overlap = config.model.ub_tp_comm_overlap

    tp_comm_overlap_cfg = None
    if tp_comm_overlap:
        from megatron.bridge.training.comm_overlap import (
            BulkOverlapCfg,
            PipelineOverlapCfg,
            RingExchangeOverlapCfg,
            TransformerLayerTPOverlapCfg,
        )

        buffer_options = config.model.ub_tp_comm_overlap_cfg
        userbuffer_args = {}

        for key in [
            "qkv_dgrad",
            "qkv_wgrad",
            "fc1_dgrad",
            "fc1_wgrad",
            "qkv_fprop",
            "proj_dgrad",
            "fc1_fprop",
            "fc2_dgrad",
            "proj_fprop",
            "fc2_fprop",
        ]:
            if key in buffer_options:
                attributes = buffer_options[key]
                fp8_buf = False
                try:
                    fp8_buf = bool(attributes.fp8_buf)
                except:
                    pass
                if attributes.method == "pipeline":
                    userbuffer_args[key] = PipelineOverlapCfg(
                        num_sm=attributes.num_sm,
                        cga_size=attributes.cga_size,
                        num_splits=attributes.num_splits,
                        set_sm_margin=bool(attributes.set_sm_margin),
                        fp8_buf=fp8_buf,
                    )
                elif attributes.method == "bulk":
                    userbuffer_args[key] = BulkOverlapCfg(
                        num_sm=attributes.num_sm,
                        cga_size=attributes.cga_size,
                        set_sm_margin=bool(attributes.set_sm_margin),
                    )
                elif attributes.method == "ring_exchange":
                    userbuffer_args[key] = RingExchangeOverlapCfg(
                        fp8_buf=fp8_buf,
                    )
                else:
                    assert False, f"method {attributes.method} is not defined."

        tp_comm_overlap_cfg = TransformerLayerTPOverlapCfg(**userbuffer_args)

    return CommOverlapConfig(
        tp_comm_overlap=tp_comm_overlap,
        tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        # PP overlap
        overlap_p2p_comm=config.model.overlap_p2p_comm,
        batch_p2p_comm=config.model.batch_p2p_comm,
        # DP overlap
        overlap_grad_reduce=config.model.optim.overlap_grad_reduce,
        overlap_param_gather=config.model.optim.overlap_param_gather,
        overlap_param_gather_with_optimizer_step=config.model.optim.overlap_param_gather_with_optim_step,
        align_param_gather=config.model.optim.align_param_gather,
        bucket_size=config.model.optim.bucket_size,
        # Pipeline bubble
        defer_embedding_wgrad_compute=config.model.defer_embedding_wgrad_compute,
        wgrad_deferral_limit=config.model.wgrad_deferral_limit,
    )


def get_profiling_config(config):
    if config.misc.memory_profiler.enable:
        return ProfilingConfig(
            use_pytorch_profiler=True,
            record_memory_history=True,
            memory_snapshot_path=f"{config.misc.memory_profiler.file_prefix}.pickle",
            profile_step_start=config.misc.memory_profiler.start_step,
            profile_step_end=config.misc.memory_profiler.end_step,
        )
    return ProfilingConfig(
        use_nsys_profiler=config.model.nsys_profile.enabled,
        profile_step_start=config.model.nsys_profile.start_step,
        profile_step_end=config.model.nsys_profile.end_step,
        profile_ranks=config.model.nsys_profile.ranks,
        record_shapes=config.model.nsys_profile.gen_shape,
        nvtx_ranges=config.model.nsys_profile.nvtx_ranges,
    )


def get_checkpoint_config(config):
    return CheckpointConfig(
        load=None,
        save=None,
        fully_parallel_load=config.model.dist_ckpt_parallel_load,
        dist_ckpt_strictness="log_all",
        load_optim=False,
        load_main_params_from_ckpt=config.model.optim.fp8_param_gather,
    )


def create_config(config):
    return ConfigContainer(
        checkpoint=get_checkpoint_config(config),
        dataset=get_data(config),
        ddp=get_ddp_config(config),
        dist=get_dist_config(config),
        logger=get_logger_config(config),
        mixed_precision=get_mixed_precision(config),
        model=get_model(config),
        optimizer=get_optimizer_config(config),
        rng=get_rng_config(config),
        profiling=get_profiling_config(config),
        rerun_state_machine=get_rerun_state_machine_config(config),
        scheduler=get_scheduler_config(config),
        tokenizer=get_tokenizer_config(config),
        train=get_train_config(config),
        validation=get_validation_config(config),
    )


def log_hyperparams(config):
    mllogger.mlperf_submission_log("gpt_oss_20b")
    if config.trainer.max_steps <= config.model.optim.sched.lr_warmup_iters:
        config.model.optim.sched.lr_warmup_iters = config.trainer.max_steps

    opt_lr_decay_steps = int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) - int(config.model.optim.sched.lr_warmup_iters)

    # Collects configs to be logged
    logging_configs = {
        # seeds
        mllogger.constants.SEED: config.model.seed,
        # HPs
        mllogger.constants.GLOBAL_BATCH_SIZE: config.model.global_batch_size,
        mllogger.constants.GRADIENT_ACCUMULATION_STEPS: (int(os.environ["MINIBS"]) / config.model.micro_batch_size),
        mllogger.constants.MAX_SEQUENCE_LENGTH: config.model.encoder_seq_length,
        mllogger.constants.EVAL_SAMPLES: int(os.environ.get("VAL_SAMPLES", 0)),
        mllogger.constants.TRAIN_SAMPLES: 1574207408,
        mllogger.constants.INIT_CHECKPOINT_STEP: config.model.custom.init_global_step,
        # Optimizers
        mllogger.constants.OPT_NAME: mllogger.constants.ADAMW,
        mllogger.constants.OPT_BASE_LR: config.model.optim.lr,
        mllogger.constants.OPT_ADAMW_BETA_1: 0.9,
        mllogger.constants.OPT_ADAMW_BETA_2: 0.95,
        mllogger.constants.OPT_ADAMW_EPSILON: 1e-5,
        mllogger.constants.OPT_ADAMW_WEIGHT_DECAY: 0.1,
        mllogger.constants.OPT_GRADIENT_CLIP_NORM: 1.0,
        # Schedulers
        mllogger.constants.OPT_END_LR: config.model.optim.sched.min_lr,
        mllogger.constants.OPT_LR_WARMUP_STEPS: config.model.optim.sched.lr_warmup_iters,
        mllogger.constants.OPT_LR_DECAY_STEPS: opt_lr_decay_steps,
        mllogger.constants.MAX_STEPS: int(os.environ.get("MAX_STEPS", 0)),
        mllogger.constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        # extra
        mllogger.constants.TENSOR_PARALLELISM: config.model.tensor_model_parallel_size,
        mllogger.constants.PIPELINE_PARALLELISM: config.model.pipeline_model_parallel_size,
        mllogger.constants.CONTEXT_PARALLELISM: config.model.context_parallel_size,
        mllogger.constants.EXPERT_PARALLELISM: config.model.expert_model_parallel_size,
        mllogger.constants.MICRO_BATCH_SIZE: config.model.micro_batch_size,
        mllogger.constants.CONFIG_FILENAME: f"config_{os.environ.get('DGXSYSTEM', 'UNKNOWN')}.sh",
        mllogger.constants.LOWEST_NUMERICAL_PRECISION_IN_LINEAR: os.environ.get("MLPERF_LINEAR_PRECISION", ""),
        mllogger.constants.LOWEST_NUMERICAL_PRECISION_IN_ATTN: os.environ.get("MLPERF_ATTN_PRECISION", ""),
        mllogger.constants.LOWEST_NUMERICAL_PRECISION_IN_COMM: os.environ.get("MLPERF_COMM_PRECISION", ""),
        # custom
        "target_accuracy": config.custom.target_log_ppl,
    }

    for key, value in logging_configs.items():
        mllogger.event(key=key, value=value)


@hydra.main(config_path="conf", config_name="gpt_oss", version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    # OmegaConf scalar interpolation cannot resolve an environment value
    # directly into a list. Convert bracket notation after interpolation so
    # the official TE partial-graph scopes reach MCore as a real ListConfig.
    scope = cfg.model.overwritten_attributes.cuda_graph_scope
    if isinstance(scope, str) and scope.startswith("[") and scope.endswith("]"):
        cfg.model.overwritten_attributes.cuda_graph_scope = [
            item.strip() for item in scope[1:-1].split(",") if item.strip()
        ]
    config_container = create_config(cfg)
    evaluate_on_fixed_validation_window(
        config_container.validation.eval_iters * config_container.train.global_batch_size
    )
    if utils.rank == 0:
        log_hyperparams(cfg)
    callbacks = [
        WarmupCallback(cfg, forward_step_func=forward_step),
        DeltaTimingCallback(cfg),
        MLPerfLoggingCallback(cfg),
    ]

    # GC Config
    config_container.train.manual_gc = True
    config_container.train.manual_gc_interval = 500
    config_container.train.manual_gc_eval = False

    # Memory management
    config_container.train.empty_unused_memory_level = 0
    config_container.train.train_sync_interval = None

    # Skip numeric checks
    config_container.train.check_optimizer_step_success = False
    config_container.train.skip_sync_grad_norm_across_mp = True

    # Skip logging & timers
    config_container.logger.skip_train_metrics_log = True
    config_container.logger.timing_log_level = -1

    config_container.train.exit_signal_handler = False
    if os.getenv("NVFP4_TORCH_ANOMALY", "0") == "1":
        torch.autograd.set_detect_anomaly(True)

    pretrain(config_container, forward_step_func=forward_step, callbacks=callbacks)


if __name__ == "__main__":
    if utils.rank == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
    main()
