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

"""Shared training utilities for Flux.1 FlatModule and reference training.

Contains: LR scheduler, loss computation, validation loop, weight init for
flat model, distributed helpers, logging utilities.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


# ---------------------------------------------------------------------------
# Hyperparameters (MLPerf reference values for GBS=1024 from rcps_flux1.json)
# ---------------------------------------------------------------------------

HPARAMS = {
    "beta1": 0.9,             # opt_adamw_beta_1
    "beta2": 0.95,            # opt_adamw_beta_2
    "eps": 1e-8,              # opt_adamw_epsilon
    "weight_decay": 0.1,      # opt_adamw_weight_decay
    "max_grad_norm": 1.0,     # opt_gradient_clip_norm
    "vae_scale": 0.3611,
    "vae_shift": 0.1159,
    "cfg_dropout_prob": 0.1,
    "target_accuracy": 0.586,
    "val_dataset_size": 29696,
    "train_dataset_size": 1099776,
    # Model config
    "hidden_size": 3072,
    "num_heads": 24,
    "depth": 19,
    "depth_single_blocks": 38,
    "in_channels": 64,
    "out_channels": 64,
    "context_in_dim": 4096,
    "vec_in_dim": 768,
    "mlp_ratio": 4.0,
    "axes_dim": (16, 56, 56),
    "theta": 10_000,
    "qkv_bias": True,
}


# ---------------------------------------------------------------------------
# LR Scheduler: linear warmup then constant hold
# ---------------------------------------------------------------------------

class WarmupConstantLR:
    """Linear warmup followed by constant learning rate."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            lr = self.base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self) -> float:
        if self.step_count <= self.warmup_steps:
            return self.base_lr * (self.step_count / self.warmup_steps)
        return self.base_lr


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_train_loss(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """MSE loss with mean reduction over local batch elements.

    Matches TorchTitan mlperf-training-flux.1 pinned branch:
    F.mse_loss(pred.float(), target.float().detach(), reduction='mean').

    Args:
        prediction: Model output [B, S, D].
        target: Ground truth [B, S, D].

    Returns:
        Scalar loss tensor.
    """
    return torch.nn.functional.mse_loss(
        prediction.float(), target.float().detach(), reduction="mean",
    )


def compute_val_loss_components(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, int]:
    """Compute per-sample MSE loss for validation.

    Returns (loss_sum, sample_count) for accumulation.
    """
    diff = prediction - target
    per_sample_loss = (diff * diff).mean(dim=(1, 2))
    loss_sum = per_sample_loss.sum()
    sample_count = prediction.shape[0]
    return loss_sum, sample_count


# ---------------------------------------------------------------------------
# Gradient clipping for flat model
# ---------------------------------------------------------------------------

def clip_grad_norm_flat(model, max_norm: float) -> float:
    """Clip gradient norm for flat model (no autograd).

    Returns the total gradient norm before clipping.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.float().norm().item() ** 2
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return total_norm


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training. Returns (rank, local_rank, world_size)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_scalar(value: float, op=dist.ReduceOp.SUM) -> float:
    """All-reduce a scalar value across ranks."""
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=op)
    return t.item()


# ---------------------------------------------------------------------------
# Weight initialization for flat model (nn.Module container names)
# ---------------------------------------------------------------------------

def init_flat_weights(model) -> None:
    """Initialize weights for the flat FluxModel, matching original init_weights().

    Uses nn.Module container names (Rule 2 compliant).
    """
    import torch.nn as nn

    # img_in, txt_in
    nn.init.xavier_uniform_(model.img_in.weight)
    nn.init.constant_(model.img_in.bias, 0)
    nn.init.xavier_uniform_(model.txt_in.weight)
    nn.init.constant_(model.txt_in.bias, 0)

    # time_in, vector_in (MLPEmbedder)
    for mlp in (model.time_in, model.vector_in):
        nn.init.normal_(mlp.in_layer.weight, std=0.02)
        nn.init.constant_(mlp.in_layer.bias, 0)
        nn.init.normal_(mlp.out_layer.weight, std=0.02)
        nn.init.constant_(mlp.out_layer.bias, 0)

    # Double blocks
    for block in model.double_blocks:
        # MLP weights (via register_module "0" and "2")
        for sub in ("img_mlp", "txt_mlp"):
            mlp = getattr(block, sub)
            nn.init.xavier_uniform_(mlp._modules["0"].weight)
            nn.init.constant_(mlp._modules["0"].bias, 0)
            nn.init.xavier_uniform_(mlp._modules["2"].weight)
            nn.init.constant_(mlp._modules["2"].bias, 0)
        # Attention weights
        for sub in ("img_attn", "txt_attn"):
            attn = getattr(block, sub)
            nn.init.xavier_uniform_(attn.qkv.weight)
            nn.init.constant_(attn.qkv.bias, 0)
            nn.init.xavier_uniform_(attn.proj.weight)
            nn.init.constant_(attn.proj.bias, 0)
        # Modulation: zero init
        # QKNorm weights: initialize to 1.0 (matching nn.RMSNorm default)
        for sub in ("img_attn", "txt_attn"):
            attn = getattr(block, sub)
            nn.init.constant_(attn.norm.query_norm.weight, 1.0)
            nn.init.constant_(attn.norm.key_norm.weight, 1.0)
        block.img_mod.lin.weight.data.zero_()
        block.img_mod.lin.bias.data.zero_()
        block.txt_mod.lin.weight.data.zero_()
        block.txt_mod.lin.bias.data.zero_()

    # Single blocks
    for block in model.single_blocks:
        nn.init.xavier_uniform_(block.linear1.weight)
        nn.init.constant_(block.linear1.bias, 0)
        nn.init.xavier_uniform_(block.linear2.weight)
        nn.init.constant_(block.linear2.bias, 0)
        nn.init.constant_(block.norm.query_norm.weight, 1.0)
        nn.init.constant_(block.norm.key_norm.weight, 1.0)
        block.modulation.lin.weight.data.zero_()
        block.modulation.lin.bias.data.zero_()

    # Final layer: zero init
    model.final_layer.adaLN_modulation._modules["1"].weight.data.zero_()
    model.final_layer.adaLN_modulation._modules["1"].bias.data.zero_()
    model.final_layer.linear.weight.data.zero_()
    model.final_layer.linear.bias.data.zero_()
