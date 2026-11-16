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
#
# This file is derived from the FLUX.1 transformer implementation in
# pytorch/torchtitan; see that project's LICENSE for the upstream terms.

"""Simplified FluxModel for FlatModule tracing."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    row0 = torch.stack([cos_out, -sin_out], dim=-1)
    row1 = torch.stack([sin_out, cos_out], dim=-1)
    rot = torch.stack([row0, row1], dim=-2)
    return rot.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    t = time_factor * t
    half = dim // 2
    with torch.device(t.device):
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def _make_attn_container(dim: int, num_heads: int, qkv_bias: bool) -> nn.Module:
    """Create an nn.Module container with qkv, norm, proj matching SelfAttention state_dict keys."""
    head_dim = dim // num_heads
    attn = nn.Module()
    attn.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    norm = nn.Module()
    norm.query_norm = nn.RMSNorm(head_dim)
    norm.key_norm = nn.RMSNorm(head_dim)
    attn.norm = norm
    attn.proj = nn.Linear(dim, dim, bias=True)
    return attn


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: tuple):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        out = self.lin(F.silu(vec))[:, None, :]
        return out.chunk(self.multiplier, dim=-1)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(dim=hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = _make_attn_container(hidden_size, num_heads, qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(dim=hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = _make_attn_container(hidden_size, num_heads, qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_mod_out = self.img_mod(vec)
        img_shift1, img_scale1, img_gate1 = img_mod_out[0], img_mod_out[1], img_mod_out[2]
        img_shift2, img_scale2, img_gate2 = img_mod_out[3], img_mod_out[4], img_mod_out[5]
        txt_mod_out = self.txt_mod(vec)
        txt_shift1, txt_scale1, txt_gate1 = txt_mod_out[0], txt_mod_out[1], txt_mod_out[2]
        txt_shift2, txt_scale2, txt_gate2 = txt_mod_out[3], txt_mod_out[4], txt_mod_out[5]

        D = self.hidden_size
        Dh = D // self.num_heads

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_scale1) * img_modulated + img_shift1
        img_qkv = self.img_attn.qkv(img_modulated)
        B, L_img = img_qkv.shape[0], img_qkv.shape[1]
        img_qkv = img_qkv.reshape(B, L_img, 3, self.num_heads, Dh).permute(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]
        img_q = self.img_attn.norm.query_norm(img_q).to(img_v)
        img_k = self.img_attn.norm.key_norm(img_k).to(img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_scale1) * txt_modulated + txt_shift1
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        L_txt = txt.shape[1]
        txt_qkv = txt_qkv.reshape(B, L_txt, 3, self.num_heads, Dh).permute(2, 0, 3, 1, 4)
        txt_q, txt_k, txt_v = txt_qkv[0], txt_qkv[1], txt_qkv[2]
        txt_q = self.txt_attn.norm.query_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn.norm.key_norm(txt_k).to(txt_v)

        # run actual attention (joint over txt+img)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        q, k = apply_rope(q, k, pe)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, -1, D)

        txt_attn, img_attn = attn[:, :L_txt], attn[:, L_txt:]

        # calculate the img blocks
        img = img + img_gate1 * self.img_attn.proj(img_attn)
        img = img + img_gate2 * self.img_mlp(
            (1 + img_scale2) * self.img_norm2(img) + img_shift2
        )

        # calculate the txt blocks
        txt = txt + txt_gate1 * self.txt_attn.proj(txt_attn)
        txt = txt + txt_gate2 * self.txt_mlp(
            (1 + txt_scale2) * self.txt_norm2(txt) + txt_shift2
        )
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)

        # QKNorm inlined as a plain nn.Module container (same state_dict keys)
        self.norm = nn.Module()
        self.norm.query_norm = nn.RMSNorm(head_dim)
        self.norm.key_norm = nn.RMSNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(dim=hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod_out = self.modulation(vec)
        mod_shift, mod_scale, mod_gate = mod_out[0], mod_out[1], mod_out[2]
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        B, L = qkv.shape[0], qkv.shape[1]
        Dh = self.hidden_size // self.num_heads
        qkv = qkv.reshape(B, L, 3, self.num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.norm.query_norm(q).to(v)
        k = self.norm.key_norm(k).to(v)

        # compute attention
        q, k = apply_rope(q, k, pe)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, L, self.hidden_size)

        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class FluxModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        hidden_size: int = 3072,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth: int = 2,
        depth_single_blocks: int = 2,
        axes_dim: tuple = (16, 56, 56),
        theta: int = 10_000,
        qkv_bias: bool = True,
        vec_in_dim: int = 768,
        context_in_dim: int = 4096,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        pe_dim = hidden_size // num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.vector_in = MLPEmbedder(in_dim=vec_in_dim, hidden_dim=hidden_size)
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=True)

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias) for _ in range(depth)]
        )

        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth_single_blocks)]
        )

        self.final_layer = LastLayer(hidden_size, patch_size=1, out_channels=out_channels)

    def init_weights(self):
        nn.init.xavier_uniform_(self.img_in.weight)
        nn.init.constant_(self.img_in.bias, 0)
        nn.init.xavier_uniform_(self.txt_in.weight)
        nn.init.constant_(self.txt_in.bias, 0)
        for mlp in (self.time_in, self.vector_in):
            nn.init.normal_(mlp.in_layer.weight, std=0.02)
            nn.init.constant_(mlp.in_layer.bias, 0)
            nn.init.normal_(mlp.out_layer.weight, std=0.02)
            nn.init.constant_(mlp.out_layer.bias, 0)
        for block in self.double_blocks:
            for layer in (block.img_mlp[0], block.img_mlp[2], block.txt_mlp[0], block.txt_mlp[2]):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            for attn in (block.img_attn, block.txt_attn):
                nn.init.xavier_uniform_(attn.qkv.weight)
                nn.init.constant_(attn.qkv.bias, 0)
                nn.init.xavier_uniform_(attn.proj.weight)
                nn.init.constant_(attn.proj.bias, 0)
            for mod in (block.img_mod, block.txt_mod):
                nn.init.constant_(mod.lin.weight, 0)
                nn.init.constant_(mod.lin.bias, 0)
        for block in self.single_blocks:
            nn.init.xavier_uniform_(block.linear1.weight)
            nn.init.constant_(block.linear1.bias, 0)
            nn.init.xavier_uniform_(block.linear2.weight)
            nn.init.constant_(block.linear2.bias, 0)
            nn.init.constant_(block.modulation.lin.weight, 0)
            nn.init.constant_(block.modulation.lin.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tensor:
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)
        return img


def build_model() -> FluxModel:
    """Build a simplified FluxModel with reduced depth for tracing."""
    model = FluxModel(
        in_channels=64,
        out_channels=64,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        vec_in_dim=768,
        context_in_dim=4096,
    )
    model.init_weights()
    return model


_BATCH = 2
_IMG_SEQ = 256
_TXT_SEQ = 256

example_inputs = (
    torch.randn(_BATCH, _IMG_SEQ, 64),
    torch.zeros(_BATCH, _IMG_SEQ, 3),
    torch.randn(_BATCH, _TXT_SEQ, 4096),
    torch.zeros(_BATCH, _TXT_SEQ, 3),
    torch.rand(_BATCH),
    torch.randn(_BATCH, 768),
)
