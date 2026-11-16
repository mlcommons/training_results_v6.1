#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Flux.1 FlatModule training with ZeRO-2, ATen loss ops, explicit AdamW, and MLPerf logging."""

import argparse
import gc
import math
import os
import sys
import time

import torch
import torch.distributed as dist

from utils.data_loader import (
    make_img_ids,
    make_txt_ids,
    build_train_loader,
    build_val_loader,
    load_empty_encodings,
)
from utils.mlperf_logger import (
    get_mllogger,
    log_init_start, log_submission_info, log_hyperparams,
    log_numerical_precision,
    log_seed, log_init_stop, log_run_start, log_run_stop,
    log_block_start, log_block_stop,
    log_eval_start, log_eval_stop, log_eval_accuracy,
    log_target_accuracy, log_opt_lr_training_steps,
    log_tracked_stats,
)
from utils.train_utils import (
    HPARAMS, init_flat_weights, all_reduce_scalar,
)
from flat.comm_overlap import BucketManager
from flat.custom_ops import aten_tail_ctm as _aten_tail
from flat.custom_ops import clip_coef_ctm as _clip_coef
from flat.custom_ops import optimizer_ctm as _optimizer_ctm
from flat import jit_guard
from flat.fp8_ops import NVFP4_ENABLED, clear_fp8_weight_cache


def get_lr(step, base_lr, warmup_steps):
    if step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr


def _assert_disclosure_matches_recipe() -> None:
    """Fail the run if the compliance disclosure and the code path disagree."""
    from flat import nvfp4_ops as _nv
    from flat.model import attn_precision as _attn

    for var, selected in (
        ("MLPERF_LINEAR_PRECISION", _nv.forward_format()),
        ("MLPERF_GRAD_PRECISION", _nv.backward_format()),
        ("MLPERF_ATTN_PRECISION", _attn()),
    ):
        disclosed = os.environ.get(var, "").strip().lower()
        if disclosed and disclosed != selected:
            raise RuntimeError(
                f"{var}={disclosed!r} but the code selected {selected!r}; "
                "the disclosure is derived in config_common.sh and must match"
            )


def main():
    jit_guard.install()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/dataset")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--results-dir", default="/results")
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--val-check-interval", type=int, required=True)
    parser.add_argument("--log-every-n-steps", type=int, required=True)
    parser.add_argument(
        "--segment-nodes",
        type=int,
        default=int(os.environ.get("SEGMENT", "0")),
        help="Nodes per NVLink domain (SLURM SEGMENT). 0 => single domain.",
    )
    parser.add_argument(
        "--replicate-inflight",
        type=int,
        default=int(os.environ.get("HSDP_REPLICATE_INFLIGHT", "1")),
        help="Max in-flight inter-domain all-reduces (overlap depth).",
    )
    args = parser.parse_args()

    os.environ["RESULTS_DIR"] = args.results_dir
    os.makedirs(args.results_dir, exist_ok=True)

    # Distributed setup
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "RANK" in os.environ:
        # device_id forces eager NCCL communicator init, which the pool
        # registrations need; the device must be set before init.
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl", device_id=torch.device(f"cuda:{local_rank}"),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    gpus_per_node = min(
        int(os.environ.get("SLURM_NTASKS_PER_NODE", "4")), world_size
    )
    segment_nodes = args.segment_nodes or (world_size // gpus_per_node)
    from flat.hier_topology import build_topology
    topo = build_topology(world_size, rank, segment_nodes, gpus_per_node)

    if dist.is_initialized() and topo.is_hierarchical:
        # Inter-domain (IB) groups are created first to lock the SHARP tree;
        # SHARP/CollNet is then disabled before the shard groups are created.
        _repl_algo = os.environ.get("REPLICATE_NCCL_ALGO")
        _prev_algo = os.environ.get("NCCL_ALGO")
        if _repl_algo:
            os.environ["NCCL_ALGO"] = _repl_algo
        replicate_pgs = [
            dist.new_group(ranks=list(range(g, world_size, topo.shard_size)))
            for g in range(topo.shard_size)
        ]
        replicate_pg = replicate_pgs[topo.intra_pos]
        # bfloat16 to match the dtype of the real replicate all-reduce
        _warm = torch.ones(1024, dtype=torch.bfloat16, device=device)
        for _ in range(20):
            dist.all_reduce(_warm, group=replicate_pg)
        torch.cuda.synchronize()
        if _repl_algo:
            if _prev_algo is None:
                os.environ.pop("NCCL_ALGO", None)
            else:
                os.environ["NCCL_ALGO"] = _prev_algo
        os.environ["NCCL_SHARP_DISABLE"] = "1"
        os.environ["NCCL_COLLNET_ENABLE"] = "0"
        shard_pgs = [
            dist.new_group(
                ranks=list(range(d * topo.shard_size, (d + 1) * topo.shard_size))
            )
            for d in range(topo.n_domains)
        ]
        shard_pg = shard_pgs[topo.domain_id]
        shard_group_name = shard_pg.group_name
        replicate_group_name = replicate_pg.group_name
    else:
        shard_group_name = (
            dist.group.WORLD.group_name if dist.is_initialized() else ""
        )
        replicate_group_name = shard_group_name
        replicate_pg = None
        shard_pg = None
    if rank == 0:
        print(
            f"Topology: world={world_size} G={topo.shard_size} "
            f"D={topo.n_domains} hierarchical={topo.is_hierarchical} "
            f"replicate_groups={topo.n_replicate_groups} "
            f"replicate_size={topo.replicate_group_size} "
            f"sharp_worthwhile={topo.sharp_worthwhile}",
            flush=True,
        )
        if topo.is_hierarchical and not topo.sharp_worthwhile:
            print(
                f"WARNING: D={topo.n_domains} < SHARP min "
                f"({topo.n_replicate_groups} replicate groups of "
                f"{topo.replicate_group_size}); IB SHARP will not engage "
                "and the inter-domain all-reduce will run RING_LL "
                "point-to-point. The all-reduce still overlaps backward "
                "(the dominant win). For in-network SHARP, use a smaller "
                "SEGMENT so D>=4. Verify from steady-state kernel names.",
                flush=True,
            )

    # MLPerf: init phase
    mllogger_inst = get_mllogger()
    log_init_start(mllogger_inst, rank)
    log_submission_info(mllogger_inst, rank, args.num_nodes)
    log_target_accuracy(mllogger_inst, rank, HPARAMS["target_accuracy"])

    # Model seed by intra-domain position: domains are identical replicas.
    torch.manual_seed(args.seed + topo.intra_pos)
    torch.cuda.manual_seed(args.seed + topo.intra_pos)
    log_seed(mllogger_inst, rank, args.seed)

    # Build model; weights come from the original module's init
    from flat.original import FluxModel as OrigFlux
    from flat.model import FluxModel as FlatFlux
    orig = OrigFlux(
        in_channels=HPARAMS["in_channels"], out_channels=HPARAMS["out_channels"],
        hidden_size=HPARAMS["hidden_size"], mlp_ratio=HPARAMS["mlp_ratio"],
        num_heads=HPARAMS["num_heads"], depth=HPARAMS["depth"],
        depth_single_blocks=HPARAMS["depth_single_blocks"],
        axes_dim=HPARAMS["axes_dim"], theta=HPARAMS["theta"],
        qkv_bias=HPARAMS["qkv_bias"], vec_in_dim=HPARAMS["vec_in_dim"],
        context_in_dim=HPARAMS["context_in_dim"],
    )
    orig.init_weights()
    model = FlatFlux(
        in_channels=HPARAMS["in_channels"], out_channels=HPARAMS["out_channels"],
        hidden_size=HPARAMS["hidden_size"], mlp_ratio=HPARAMS["mlp_ratio"],
        num_heads=HPARAMS["num_heads"], depth=HPARAMS["depth"],
        depth_single_blocks=HPARAMS["depth_single_blocks"],
        axes_dim=HPARAMS["axes_dim"], theta=HPARAMS["theta"],
        qkv_bias=HPARAMS["qkv_bias"], vec_in_dim=HPARAMS["vec_in_dim"],
        context_in_dim=HPARAMS["context_in_dim"],
    )
    model.load_state_dict(orig.state_dict())
    del orig
    for block in model.single_blocks:
        block.split_linears()
    model = model.to(device).to(torch.bfloat16)
    for block in model.double_blocks:
        block.alloc_scratch_buffers()
    for block in model.single_blocks:
        block.alloc_scratch_buffers()
    params = list(model.parameters())
    total_numel = sum(p.numel() for p in params)
    if rank == 0:
        print(f"Model parameters: {total_numel:,}", flush=True)

    # Per-bucket ZeRO-2 with communication overlap
    bucket_mgr = BucketManager(
        model,
        topo,
        shard_group_name,
        replicate_group_name,
        device,
        max_bucket_numel=1_000_000_000,
        replicate_inflight_max=args.replicate_inflight,
        replicate_group=replicate_pg,
        shard_group=shard_pg,
    )
    bucket_mgr.assign_grad_views()
    # GEMM-class registration keys on weight identity, so it must run after
    # BucketManager has repointed every param.data at its bucket slice.
    if NVFP4_ENABLED:
        from flat import nvfp4_ops
        nvfp4_ops.register_gemm_classes(model)
        if rank == 0:
            print(
                f"Precision: fprop={nvfp4_ops._FWD_FORMAT} "
                f"bprop={nvfp4_ops._BWD_FORMAT} "
                f"demoted={sorted(nvfp4_ops._DEMOTED) or 'none'} "
                f"registered={len(nvfp4_ops._class_of_weight)} weights "
                f"bf16_bisect={nvfp4_ops._FLAT_NVFP4_BF16}",
                flush=True,
            )
    if rank == 0:
        max_b = max(b.raw_numel for b in bucket_mgr.buckets)
        print(
            f"Buckets: {len(bucket_mgr.buckets)}, "
            f"max_numel={max_b:,} ({max_b * 2 / 1e9:.2f} GB bf16)",
            flush=True,
        )

    # Log hyperparams
    gbs = args.batch_size
    mbs = gbs // world_size
    assert gbs % world_size == 0, f"GBS={gbs} not divisible by world_size={world_size}"
    assert mbs * world_size == gbs, "GBS must equal MBS * world_size (no grad accumulation)"
    # val_check_interval rounded up to a whole step
    val_check_interval = math.ceil(args.val_check_interval / gbs) * gbs
    # Data loaders
    # Two seeds: the per-rank data seed drives noise, timesteps and the CFG
    # mask (every rank must draw differently); the sampler seed must be the
    # same on all ranks so that DistributedSampler's indices[rank::world]
    # partition the dataset instead of overlapping.
    data_seed = args.seed + rank
    # Re-seed the global RNG with the per-rank data seed before building the
    # loaders: DataLoader worker seeds and noise draws come from it.
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed(data_seed)
    empty_t5, empty_clip = load_empty_encodings(args.dataset_root)
    train_loader, _, _ = build_train_loader(
        dataset_root=args.dataset_root,
        micro_batch_size=mbs,
        num_workers=args.num_workers,
        seed=data_seed,
        sampler_seed=args.seed,
        cfg_dropout_prob=HPARAMS["cfg_dropout_prob"],
        vae_scale=HPARAMS["vae_scale"],
        vae_shift=HPARAMS["vae_shift"],
    )
    val_loader = build_val_loader(
        dataset_root=args.dataset_root,
        micro_batch_size=mbs,
        num_workers=args.num_workers,
        vae_scale=HPARAMS["vae_scale"],
        vae_shift=HPARAMS["vae_shift"],
    )
    train_epoch = 0

    log_hyperparams(mllogger_inst, rank, HPARAMS,
                    args.num_nodes, world_size // args.num_nodes, gbs,
                    args.learning_rate, args.warmup_steps, val_check_interval,
                    train_samples=len(train_loader.dataset),
                    eval_samples=len(val_loader.dataset))
    _assert_disclosure_matches_recipe()
    if _optimizer_ctm.BC_MODE == 1 and args.max_steps > _optimizer_ctm.BC_MAX_STEP:
        raise ValueError(
            f"--max-steps {args.max_steps} exceeds the AdamW bias-correction table "
            f"({_optimizer_ctm.BC_MAX_STEP} steps)"
        )
    log_numerical_precision(mllogger_inst, rank)
    log_opt_lr_training_steps(mllogger_inst, rank, args.max_steps)

    # Training loop
    _LOG_GRAD_NORM = os.environ.get("FLAT_LOG_GRAD_NORM", "0") == "1"
    log_every_n_steps = args.log_every_n_steps
    eval_freq_steps = val_check_interval // gbs

    model.train()
    global_step = 0
    train_iter = None
    data_gen = torch.Generator().manual_seed(data_seed)
    block_started = False
    _last_log_time = None  # For window-based throughput
    _last_log_step = 0
    converged = False

    # Static GPU input buffers, written each step before graph replay.
    _bf16 = torch.bfloat16
    _img_buf = torch.empty(mbs, 256, 64, device=device, dtype=_bf16)
    _txt_buf = torch.empty(mbs, 256, 4096, device=device, dtype=_bf16)
    _timesteps_buf = torch.empty(mbs, device=device, dtype=_bf16)
    _y_buf = torch.empty(mbs, 768, device=device, dtype=_bf16)
    _target_buf = torch.empty(mbs, 256, 64, device=device, dtype=_bf16)

    # img_ids/txt_ids are only read once (RoPE is cached)
    _img_ids_buf = torch.empty(mbs, 256, 3, device=device, dtype=_bf16)
    _txt_ids_buf = torch.empty(mbs, 256, 3, device=device, dtype=_bf16)

    if rank == 0:
        print(f"Starting training: GBS={gbs}, MBS={mbs}, "
              f"world_size={world_size}, max_steps={args.max_steps}, "
              f"eval_freq_steps={eval_freq_steps}", flush=True)

    gc.disable()

    # Full-iteration training step (graphable with NCCL).
    # _wgrad_delay_active is enabled only for graph capture and replay.
    _wgrad_delay_active = False

    def _fwd_bwd_clip():
        bucket_mgr.prepare_ag_overlap()
        # No zero_grad: every param.grad is fully overwritten each step.
        pred = model.forward(
            _img_buf, _img_ids_buf, _txt_buf, _txt_ids_buf,
            _timesteps_buf, _y_buf,
            on_block_start=bucket_mgr.wait_ag_for_block,
        )
        bucket_mgr.finish_ag_overlap()
        grad_out = _loss_fwd_bwd(pred)
        # Backward with per-bucket RS overlap
        bucket_mgr.reset_rs_overlap()
        if _wgrad_delay_active:
            # Wgrad delay: dgrad on main stream, wgrad on wgrad stream
            model.backward(
                grad_out,
                on_block_done=(
                    bucket_mgr.fire_rs_for_block_wgrad_graphsafe
                ),
                wgrad_mgr=bucket_mgr,
            )
        else:
            model.backward(
                grad_out,
                on_block_done=(
                    bucket_mgr.fire_rs_for_block_graphsafe
                ),
            )
        # The DSL grad-norm kernel reads the BF16 buffer and scrubs NaN/Inf
        # itself, so neither the nan_to_num pass nor the FP32 upcast is needed.
        bucket_mgr.wait_all_reduce_scatters_graphsafe(
            fuse_upcast=False, scrub=False,
        )
        local_norm_sq = bucket_mgr.get_grad_norm_sq_ctm()
        norm_ar = torch.ops._c10d_functional.all_reduce(
            local_norm_sq, "sum", bucket_mgr.shard_group_name,
        )
        local_norm_sq = torch.ops._c10d_functional.wait_tensor(norm_ar)
        # Gradients are SUM-reduced over world_size ranks: clip against the
        # mean norm and fold 1/world_size into the optimizer's inv_scale.
        if _clip_coef.ENABLED:
            torch.ops.flux.clip_coef_ctm(
                local_norm_sq, bucket_mgr._inv_scale, _gnorm_buf,
                HPARAMS["max_grad_norm"], bucket_mgr.world_size,
            )
        else:
            mean_norm = (
                torch.ops.aten.sqrt.default(local_norm_sq)
                / bucket_mgr.world_size
            )
            clip_coef = torch.ops.aten.clamp.default(
                HPARAMS["max_grad_norm"] / (mean_norm + 1e-6),
                max=1.0,
            )
            if _gnorm_buf is not None:
                _gnorm_buf.copy_(mean_norm)
            bucket_mgr._inv_scale.copy_(clip_coef / bucket_mgr.world_size)

    def _train_step():
        _fwd_bwd_clip()
        bucket_mgr.fused_adamw_step_ctm(
            HPARAMS["beta1"], HPARAMS["beta2"],
            HPARAMS["eps"], HPARAMS["weight_decay"],
        )

    # Overlapped RS step: fires RS during backward; overlaps AG in forward
    # when _ag_overlap_active is set.
    _ag_overlap_active = False

    def _overlapped_step():
        # No zero_grad: every param.grad is fully overwritten each step.
        if _ag_overlap_active:
            pred = model.forward(
                _img_buf, _img_ids_buf, _txt_buf, _txt_ids_buf,
                _timesteps_buf, _y_buf,
                on_block_start=bucket_mgr.wait_ag_for_block,
            )
            bucket_mgr.finish_ag_overlap()
        else:
            pred = model.forward(
                _img_buf, _img_ids_buf, _txt_buf, _txt_ids_buf,
                _timesteps_buf, _y_buf,
            )
        grad_out = _loss_fwd_bwd(pred)
        # Fire RS per bucket during backward
        bucket_mgr.reset_rs_overlap()
        if use_wgrad_delay:
            # 3-stream pipeline: dgrad (main) | wgrad | RS (comm)
            model.backward(
                grad_out,
                on_block_done=(
                    bucket_mgr.fire_rs_for_block_wgrad_delay
                ),
                wgrad_mgr=bucket_mgr,
            )
        else:
            model.backward(
                grad_out,
                on_block_done=bucket_mgr.fire_rs_for_block,
            )
        bucket_mgr.wait_all_reduce_scatters()
        bucket_mgr.upcast_reduced_grads()
        local_norm_sq = bucket_mgr.get_grad_norm_sq()
        norm_ar = torch.ops._c10d_functional.all_reduce(
            local_norm_sq, "sum", bucket_mgr.shard_group_name,
        )
        local_norm_sq = torch.ops._c10d_functional.wait_tensor(norm_ar)
        if _gnorm_buf is not None:
            _gnorm_buf.copy_(
                torch.ops.aten.sqrt.default(local_norm_sq)
                / bucket_mgr.world_size
            )
        bucket_mgr.clip_grads_gpu(
            HPARAMS["max_grad_norm"], local_norm_sq,
        )

    # Static loss buffer; read outside the graph on logging steps.
    _loss_buf = torch.zeros(1, device=device, dtype=torch.float32)
    # Gradient-norm buffer, written inside the captured step.
    _gnorm_buf = (
        torch.zeros(1, device=device, dtype=torch.float32)
        if _LOG_GRAD_NORM else None
    )
    _ones_buf = torch.ones(1, device=device, dtype=torch.float32)

    def _loss_fwd_bwd(pred):
        """MSE loss into _loss_buf and its gradient wrt pred, in one kernel."""
        if _aten_tail.ENABLED:
            return torch.ops.flux.mse_loss_fwd_bwd_ctm(
                pred, _target_buf, _loss_buf,
            )
        pred_f32 = pred.float()
        target_f32 = _target_buf.float()
        _loss_buf.copy_(
            torch.ops.aten.mse_loss.default(pred_f32, target_f32, 1)
        )
        grad_out = torch.ops.aten.mse_loss_backward.default(
            _ones_buf, pred_f32, target_f32, 1,
        )
        return grad_out.to(pred.dtype)
    # Static eval buffers: masked sum of per-sample MSE over the batch and the
    # number of real (non-padding) samples in it. The val sampler pads every
    # rank to the same number of full batches; padded rows carry valid=0.
    _eval_loss_buf = torch.zeros(1, device=device, dtype=torch.float32)
    _eval_count_buf = torch.zeros(1, device=device, dtype=torch.float32)
    _valid_buf = torch.ones(mbs, device=device, dtype=torch.float32)

    # Eval step: forward only, captured into a graph sharing the train pool.
    def _eval_step():
        with torch.no_grad():
            bucket_mgr.prepare_ag_overlap()
            pred = model.forward(
                _img_buf, _img_ids_buf, _txt_buf, _txt_ids_buf,
                _timesteps_buf, _y_buf,
                on_block_start=bucket_mgr.wait_ag_for_block,
            )
            bucket_mgr.finish_ag_overlap()
            pred_f32 = pred.float()
            target_f32 = _target_buf.float()
            diff = pred_f32 - target_f32
            per_sample_loss = (diff * diff).mean(dim=(1, 2))
            _eval_loss_buf.copy_((per_sample_loss * _valid_buf).sum())
            _eval_count_buf.copy_(_valid_buf.sum())

    def _run_validation_graphed(step, samples_count):
        """Validation via captured eval graph replayed per val batch."""
        if mllogger_inst is not None:
            log_eval_start(mllogger_inst, rank, samples_count, step)
        total_loss = 0.0
        total_samples = 0
        for _vp in val_loader:
            _img_buf.copy_(_vp["img"], non_blocking=True)
            _txt_buf.copy_(_vp["txt"], non_blocking=True)
            _timesteps_buf.copy_(_vp["timesteps"], non_blocking=True)
            _y_buf.copy_(_vp["y"], non_blocking=True)
            _target_buf.copy_(_vp["target"], non_blocking=True)
            _valid_buf.copy_(_vp["valid"], non_blocking=True)
            if _eval_graph is not None:
                _eval_graph.replay()
            else:
                # Eager fallback (FLAT_USE_EAGER=1)
                model.eval()
                _eval_step()
                model.train()
            total_loss += _eval_loss_buf.item()
            total_samples += int(round(_eval_count_buf.item()))
        total_loss = all_reduce_scalar(total_loss)
        total_samples = int(all_reduce_scalar(float(total_samples)))
        expected = len(val_loader.dataset)
        assert total_samples == expected, (
            f"validation covered {total_samples} samples, expected {expected}"
        )
        val_loss = total_loss / max(total_samples, 1)
        if mllogger_inst is not None:
            log_eval_accuracy(mllogger_inst, rank, val_loss, samples_count)
            log_eval_stop(mllogger_inst, rank, samples_count, step)
        if rank == 0:
            print(f"  VALIDATION: val_loss={val_loss:.6f} "
                  f"target={HPARAMS['target_accuracy']:.3f}", flush=True)
        return val_loss

    # CUDA graph state
    cuda_graph_warmup_steps = 2
    _cuda_graph = None
    _eval_graph = None
    # FLAT_USE_EAGER=1: eager RS overlap (no CUDA graph); default is CUDA graph.
    use_rs_overlap = os.environ.get("FLAT_USE_EAGER", "0") == "1"
    use_wgrad_delay = (
        use_rs_overlap
        and os.environ.get("WGRAD_DELAY", "0") == "1"
    )
    use_cuda_graph = world_size > 1 and not use_rs_overlap
    # WGRAD_DELAY_GRAPH=1: wgrad GEMMs on a separate stream in graph mode.
    use_wgrad_delay_graph = (
        not use_rs_overlap
        and os.environ.get("WGRAD_DELAY_GRAPH", "0") == "1"
    )
    if use_wgrad_delay_graph:
        bucket_mgr.init_wgrad_delay()
        if rank == 0:
            print("Wgrad delay (graph mode): wgrad GEMMs on separate "
                  "stream", flush=True)
    if use_rs_overlap and rank == 0:
        print(
            f"RS overlap mode: RS fires during backward on comm stream"
            f" (wgrad_delay={use_wgrad_delay})",
            flush=True,
        )

    # Data prefetching
    def _load_next_batch():
        nonlocal train_iter, train_epoch
        try:
            return next(train_iter)
        except StopIteration:
            # New pass over the data: advance the sampler epoch so the
            # permutation changes (it is a function of seed + epoch).
            train_epoch += 1
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(train_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    # NCCL buffer registration (SHARP / NVLS pools, env-gated); must run
    # before CUDA graph capture.
    bucket_mgr.register_sharp_pool()
    bucket_mgr.register_nvls_pool()

    def _prebuild_kernels():
        """Run one synthetic train step and eval forward so every JIT kernel is built.

        Weights, optimizer state and RNG streams are restored afterwards.
        """
        t0 = time.time()
        cuda_rng = torch.cuda.get_rng_state(device)
        cpu_rng = torch.get_rng_state()
        gen = torch.Generator(device=device).manual_seed(0)
        for buf in (_img_buf, _txt_buf, _y_buf, _target_buf):
            buf.normal_(generator=gen)
        _timesteps_buf.uniform_(generator=gen)
        _img_ids_buf.copy_(make_img_ids(mbs))
        _txt_ids_buf.copy_(make_txt_ids(mbs))
        torch.cuda.empty_cache()
        if use_cuda_graph:
            _fwd_bwd_clip()
        elif use_rs_overlap:
            _overlapped_step()
        else:
            bucket_mgr.zero_grad()
            pred = model.forward(
                _img_buf, _img_ids_buf, _txt_buf, _txt_ids_buf,
                _timesteps_buf, _y_buf,
            )
            grad_out = torch.ops.aten.mse_loss_backward.default(
                _ones_buf, pred.float(), _target_buf.float(), 1,
            )
            model.backward(grad_out.to(pred.dtype))
        model.eval()
        clear_fp8_weight_cache()
        _eval_step()
        model.train()
        torch.cuda.synchronize()
        clear_fp8_weight_cache()
        model._cached_pe = None
        # The synthetic step above never reaches the optimizer apply, so the AdamW shard kernel
        # would otherwise compile inside the timing region. Compiles against fake tensors: no
        # parameter, moment or step state is read or written.
        from flat.custom_ops import optimizer_ctm as _opt_ctm
        _opt_ctm.prebuild(device.index)
        torch.cuda.set_rng_state(cuda_rng, device)
        torch.set_rng_state(cpu_rng)
        jit_guard.freeze()
        if rank == 0:
            print(f"Prebuilt {jit_guard.compile_count()} JIT kernels "
                  f"({jit_guard.summary()}) in {time.time() - t0:.1f} s "
                  "before run_start", flush=True)

    if os.environ.get("FLAT_PREBUILD", "1") == "1":
        _prebuild_kernels()

    # Tensor capture at the fp8_ops seam (FLAT_CAPTURE_DIR; needs FLAT_USE_EAGER=1).
    from flat import quant_capture
    _capture_on = quant_capture.install_capture(model)

    # MLPerf: init_stop, run_start
    log_init_stop(mllogger_inst, rank)
    log_run_start(mllogger_inst, rank)
    train_iter = iter(train_loader)
    _next_processed = _load_next_batch()

    _nsys_start = int(os.environ.get("NSYS_START_STEP", "0"))
    _nsys_end = int(os.environ.get("NSYS_END_STEP", "0"))

    while global_step < args.max_steps and not converged:
        step_start = time.time()

        if not block_started:
            log_block_start(mllogger_inst, rank, val_check_interval,
                            global_step)
            block_started = True

        data_t0 = time.time()
        processed = _next_processed
        _img_buf.copy_(processed["img"], non_blocking=True)
        _txt_buf.copy_(processed["txt"], non_blocking=True)
        _timesteps_buf.copy_(processed["timesteps"], non_blocking=True)
        _y_buf.copy_(processed["y"], non_blocking=True)
        _target_buf.copy_(processed["target"], non_blocking=True)
        if model._cached_pe is None:
            _img_ids_buf.copy_(processed["img_ids"], non_blocking=True)
            _txt_ids_buf.copy_(processed["txt_ids"], non_blocking=True)
        del processed
        data_time = time.time() - data_t0

        global_step += 1
        current_lr = get_lr(
            global_step, args.learning_rate, args.warmup_steps,
        )

        if _capture_on:
            quant_capture.begin_step(global_step)

        if _nsys_start and global_step == _nsys_start:
            torch.cuda.cudart().cudaProfilerStart()

        # Execute or replay the training step
        # PROFILE_STEPS: dump a kernel trace on rank 0 at that step
        _profile_step = (
            os.environ.get("PROFILE_STEPS")
            and global_step == int(os.environ["PROFILE_STEPS"])
            and rank == 0
        )
        if world_size > 1:
            if _cuda_graph is not None:
                # Graph replay: update per-step scalars, then replay.
                bucket_mgr._lr_tensor.fill_(current_lr)
                bucket_mgr._step_tensor.fill_(global_step)
                if _profile_step:
                    from torch.profiler import profile, ProfilerActivity
                    with profile(
                        activities=[ProfilerActivity.CUDA],
                        record_shapes=True,
                    ) as prof:
                        _cuda_graph.replay()
                        torch.cuda.synchronize()
                    prof_dir = os.environ.get(
                        "PROFILE_DIR", "/workspace/flat-models/profiles",
                    )
                    prof.export_chrome_trace(
                        f"{prof_dir}/flat_torch_profile.json",
                    )
                    print(prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=40,
                    ), flush=True)
                else:
                    _cuda_graph.replay()
            elif use_cuda_graph and global_step == cuda_graph_warmup_steps:
                bucket_mgr._lr_tensor.fill_(current_lr)
                bucket_mgr._step_tensor.fill_(global_step)
                # Enable wgrad delay for graph capture and all replays
                if use_wgrad_delay_graph:
                    _wgrad_delay_active = True
                if rank == 0:
                    print(f"Capturing CUDA graph"
                          f" (RS overlap={use_rs_overlap})...",
                          flush=True)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                _cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    _cuda_graph,
                    capture_error_mode="thread_local",
                ):
                    _train_step()
                torch.cuda.synchronize()
                if rank == 0:
                    print("CUDA graph captured.", flush=True)

                # Eval graph shares the train graph's memory pool.
                model.eval()
                clear_fp8_weight_cache()
                if rank == 0:
                    print("Capturing eval CUDA graph...", flush=True)
                _eval_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    _eval_graph,
                    pool=_cuda_graph.pool(),
                    capture_error_mode="thread_local",
                ):
                    _eval_step()
                torch.cuda.synchronize()
                model.train()
                if rank == 0:
                    print("Eval CUDA graph captured.", flush=True)
            else:
                # Warmup: eager run of the step to be captured
                if use_cuda_graph:
                    torch.cuda.empty_cache()
                    bucket_mgr._lr_tensor.fill_(current_lr)
                    bucket_mgr._step_tensor.fill_(global_step)
                    if _profile_step:
                        from torch.profiler import (
                            profile, ProfilerActivity,
                        )
                        with profile(
                            activities=[ProfilerActivity.CUDA],
                            record_shapes=True,
                        ) as prof:
                            _train_step()
                            torch.cuda.synchronize()
                        prof_dir = os.environ.get(
                            "PROFILE_DIR",
                            "/workspace/flat-models/profiles",
                        )
                        prof.export_chrome_trace(
                            f"{prof_dir}/flat_torch_profile.json",
                        )
                        print(prof.key_averages().table(
                            sort_by="cuda_time_total",
                            row_limit=60,
                        ), flush=True)
                    else:
                        _train_step()
                else:
                    # RS overlap eager path
                    if _profile_step:
                        from torch.profiler import (
                            profile, ProfilerActivity,
                        )
                        with profile(
                            activities=[ProfilerActivity.CUDA],
                            record_shapes=True,
                        ) as prof:
                            _overlapped_step()
                            bucket_mgr.fused_adamw_step(
                                current_lr,
                                HPARAMS["beta1"],
                                HPARAMS["beta2"],
                                HPARAMS["eps"],
                                HPARAMS["weight_decay"],
                                global_step,
                            )
                            bucket_mgr.prepare_ag_overlap()
                            _ag_overlap_active = True
                            torch.cuda.synchronize()
                        prof_dir = os.environ.get(
                            "PROFILE_DIR",
                            "/workspace/flat-models/profiles",
                        )
                        prof.export_chrome_trace(
                            f"{prof_dir}/flat_torch_profile.json",
                        )
                        print(prof.key_averages().table(
                            sort_by="cuda_time_total",
                            row_limit=60,
                        ), flush=True)
                    else:
                        _overlapped_step()
                        bucket_mgr.fused_adamw_step(
                            current_lr,
                            HPARAMS["beta1"], HPARAMS["beta2"],
                            HPARAMS["eps"],
                            HPARAMS["weight_decay"],
                            global_step,
                        )
                        bucket_mgr.prepare_ag_overlap()
                        _ag_overlap_active = True

        else:
            bucket_mgr.zero_grad()
            pred = model.forward(
                _img_buf, _img_ids_buf, _txt_buf, _txt_ids_buf,
                _timesteps_buf, _y_buf,
            )
            grad_out = _loss_fwd_bwd(pred)
            model.backward(grad_out)
            bucket_mgr.copy_grads_to_reduced()
            bucket_mgr.upcast_reduced_grads()
            bucket_mgr.clip_grads_gpu(
                HPARAMS["max_grad_norm"],
                bucket_mgr.get_grad_norm_sq(),
            )
            bucket_mgr.fused_adamw_step(
                current_lr,
                HPARAMS["beta1"], HPARAMS["beta2"],
                HPARAMS["eps"], HPARAMS["weight_decay"],
                global_step,
            )
            bucket_mgr.copy_master_to_params()
        clear_fp8_weight_cache()

        if _nsys_end and global_step == _nsys_end:
            # Teardown order matters: sync, stop the profiler, reset graphs
            # (they hold NCCL comm refs), barrier, destroy NCCL, then exit.
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            if _eval_graph is not None:
                _eval_graph.reset()
                _eval_graph = None
            if _cuda_graph is not None:
                _cuda_graph.reset()
                _cuda_graph = None
            gc.collect()
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(0)

        # Capture finishes in place; breaking out of the loop would hang the collectives.
        if _capture_on and quant_capture.capture_done():
            print(f"CAPTURE manifest: {quant_capture.finish_capture()}",
                  flush=True)

        prefetch_t0 = time.time()
        _next_processed = _load_next_batch()
        prefetch_wait = time.time() - prefetch_t0

        # Logging (GPU sync only on logging steps)
        if global_step % log_every_n_steps == 0:
            torch.cuda.synchronize()
            now = time.time()
            loss_value = _loss_buf.item()
            if _last_log_time is not None:
                window = now - _last_log_time
                avg_step = window / (global_step - _last_log_step)
            else:
                avg_step = now - step_start
            if rank == 0:
                throughput = gbs / avg_step
                gnorm = (
                    f"gnorm={_gnorm_buf.item():.6f} | "
                    if _gnorm_buf is not None else ""
                )
                print(f"step={global_step:5d} | loss={loss_value:.6f} | "
                      f"{gnorm}"
                      f"lr={current_lr:.6e} | "
                      f"samples={global_step * gbs} | "
                      f"throughput={throughput:.1f} samples/s | "
                      f"avg_step={avg_step:.3f}s "
                      f"prefetch={prefetch_wait:.3f}s",
                      flush=True)
            _last_log_time = now
            _last_log_step = global_step
            log_tracked_stats(mllogger_inst, rank, loss_value,
                              current_lr, global_step,
                              global_step * gbs, avg_step)

        # Online validation + convergence check
        if global_step % eval_freq_steps == 0:

            log_block_stop(mllogger_inst, rank, val_check_interval,
                           global_step)
            block_started = False

            val_loss = _run_validation_graphed(global_step, global_step * gbs)
            if val_loss <= HPARAMS["target_accuracy"]:
                if rank == 0:
                    print(f"CONVERGED at step {global_step}, "
                          f"val_loss={val_loss:.6f}", flush=True)
                converged = True
            else:
                log_block_start(mllogger_inst, rank, val_check_interval,
                                global_step)
                block_started = True

            # Re-stamp the log timer and step so eval time does not land in
            # the next logging window's avg_step.
            if _last_log_time is not None:
                _last_log_time = time.time()
                _last_log_step = global_step

    # MLPerf: run_stop
    if not converged and block_started:
        log_block_stop(mllogger_inst, rank, val_check_interval, global_step)

    status = "success" if converged else "aborted"
    log_run_stop(mllogger_inst, rank, status, global_step,
                 global_step * gbs)

    if rank == 0:
        print(f"Training complete. Steps: {global_step}, "
              f"Samples: {global_step * gbs}, Status: {status}", flush=True)

    # Reset CUDA graphs before destroying the process group: they hold NCCL comm refs.
    if _eval_graph is not None:
        _eval_graph.reset()
        _eval_graph = None
    if _cuda_graph is not None:
        _cuda_graph.reset()
        _cuda_graph = None
    gc.enable()
    gc.collect()
    torch.cuda.synchronize()

    bucket_mgr.deregister_sharp_pool()
    bucket_mgr.deregister_nvls_pool()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        rank = int(os.environ.get("RANK", -1))
        with open(f"/workspace/logs/error_rank{rank}.txt", "w") as ef:
            traceback.print_exc(file=ef)
        sys.exit(1)
