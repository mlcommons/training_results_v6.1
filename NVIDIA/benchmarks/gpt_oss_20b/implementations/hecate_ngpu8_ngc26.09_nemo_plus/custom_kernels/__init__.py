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

"""Custom kernels and TransformerEngine/Megatron patches for gpt_oss_20b (CuTe DSL, Triton, CUDA C++).

Attention : fused_attn_bwd_fp8.py, fused FP8 (Float8CurrentScaling) SDPA backward for Blackwell/Rubin tcgen05 (CUTLASS 4.8
            fmha_helpers.py vendored unmodified as fmha_helpers.py); attn_bwd_hook.py routes TE's fused_attn_bwd to it.
RMSNorm   : rmsnorm_bwd_fused.py (Triton, two launches) and rmsnorm_bwd_cuda* (CUDA C++, one launch, NORM_BWD_CUDA):
            fused backward, dx + residual accumulate + MXFP8 row/col output + dgamma/dbias in one sweep of dy/x;
            norm_bwd_hook.py routes TE's rmsnorm_bwd and Megatron's pre-MLP norm / attention BDA to it.
MoE       : moe_qperm.py (Triton) / moe_qperm_cuda* (CUDA C++, MOE_QPERM_CUDA): HybridEP dense-layout permute fused with
            the MXFP8 quantize (fwd), probability-scaled gradient gather-quantize + dprob + fc2 dbias (bwd); moe_routing_meta.py
            (EP=1 routing metadata, replaces HybridEP's planner chain); unpermute_add.py (unpermute + fused row add); hooks
            moe_qperm_hook.py, moe_dglu_strip_hook.py, residual_bias_hook.py.
Optimizer : flat_adam_cuda* (FLAT_ADAM): flat per-bucket capturable master Adam, bit for bit TE's AdamCapturableMasterFunctor,
            25 launches per step instead of 150; opt_side_stream_hook.py runs the distributed optimizer per bucket
            (RS -> grad-norm partial -> Adam -> AG) on a side stream inside the full-iteration CUDA graph, step-boundary host syncs removed.
"""
