# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

             

from typing import List

import torch

                         
import triton

                         
import triton.language as tl
from generative_recommenders.common import triton_autotune
from generative_recommenders.ops.utils import is_sm100_plus

TMA_AVAILABLE = False
try:
                             
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    pass

HAS_TLX = False
try:
                             
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    pass


def is_blackwell_triton_swiglu_supported() -> bool:
    return is_sm100_plus() and TMA_AVAILABLE and HAS_TLX


def _swiglu_tma_set_block_size_hook(nargs) -> None:
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)

    nargs["x_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["w_gate_desc"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["w_up_desc"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["out_desc"].block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]


def get_swiglu_configs(pre_hook) -> List[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        ),
    ]


@triton.jit
def _compute_pid_swiglu(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_autotune(
    configs=get_swiglu_configs(pre_hook=_swiglu_tma_set_block_size_hook),
    key=["M_BLOCK", "N", "K"],
)
@triton.jit
def _swiglu_fwd_tma_ws_persistent(
    x_desc,
    w_gate_desc,
    w_up_desc,
    out_desc,
    M,
    N,
    K,
    M_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
                           
    x_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), x_desc.dtype, NUM_SMEM_BUFFERS)

                                               
    w_gate_buffers = tlx.local_alloc(
        (BLOCK_N, BLOCK_K), w_gate_desc.dtype, NUM_SMEM_BUFFERS
    )
    w_up_buffers = tlx.local_alloc(
        (BLOCK_N, BLOCK_K), w_up_desc.dtype, NUM_SMEM_BUFFERS
    )

                                    
    tmem_gate_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )
    tmem_up_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )

                                                   
    smem_full_bars_x_gate = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS,
        arrive_count=1,                  
    )
    smem_full_bars_up = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS,
        arrive_count=1,                  
    )
                                                                                   
    smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS,
        arrive_count=2,                  
    )

                                                   
                    
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
                    
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
                                                                                  
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

                                        
            processed_tiles = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid_swiglu(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )
                offs_m = pid_m * BLOCK_M
                offs_n = pid_n * BLOCK_N

                cur_tmem_buf = processed_tiles % int(NUM_TMEM_BUFFERS)
                tmem_read_phase = (processed_tiles // int(NUM_TMEM_BUFFERS)) & 1

                                                        
                                 
                tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)

                                                    
                                 
                gate_tmem = tmem_gate_buffers[cur_tmem_buf]
                up_tmem = tmem_up_buffers[cur_tmem_buf]

                if EPILOGUE_SUBTILE > 1:
                                              
                    slice_size: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE
                    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                        gate_subslice = tlx.local_slice(
                            gate_tmem,
                            [0, slice_id * slice_size],
                                            
                            [BLOCK_M, slice_size],
                        )
                        up_subslice = tlx.local_slice(
                            up_tmem,
                            [0, slice_id * slice_size],
                                            
                            [BLOCK_M, slice_size],
                        )

                        gate = tlx.local_load(gate_subslice).to(out_desc.dtype)
                        up = tlx.local_load(up_subslice).to(out_desc.dtype)

                        gate_fp32 = gate.to(tl.float32)
                        silu_gate = (gate_fp32 * tl.sigmoid(gate_fp32)).to(
                            out_desc.dtype
                        )
                        result = silu_gate * up

                        out_desc.store([offs_m, offs_n + slice_id * slice_size], result)
                else:
                                       
                    gate = tlx.local_load(gate_tmem).to(out_desc.dtype)
                    up = tlx.local_load(up_tmem).to(out_desc.dtype)

                    gate_fp32 = gate.to(tl.float32)
                    silu_gate = (gate_fp32 * tl.sigmoid(gate_fp32)).to(out_desc.dtype)
                    result = silu_gate * up

                    out_desc.store([offs_m, offs_n], result)

                                                     
                                
                tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)

                processed_tiles += 1

                                                                             
        with tlx.async_task(num_warps=4, num_regs=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

            processed_k_iters = 0
            processed_tiles = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid_swiglu(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )

                cur_tmem_buf = processed_tiles % int(NUM_TMEM_BUFFERS)
                tmem_write_phase = (processed_tiles // int(NUM_TMEM_BUFFERS)) & 1

                                             
                tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase ^ 1)

                                                              
                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)

                    total_iters = processed_k_iters + k
                    dot_phase = (total_iters // int(NUM_SMEM_BUFFERS)) & 1

                                                                          
                    tlx.barrier_wait(smem_full_bars_x_gate[buf], dot_phase)

                                                     
                    w_gate_trans = tlx.local_trans(w_gate_buffers[buf])

                                                 
                    tlx.async_dot(
                        x_buffers[buf],
                        w_gate_trans,
                        tmem_gate_buffers[cur_tmem_buf],
                                        
                        use_acc=(k > 0),
                        mBarriers=[smem_empty_bars[buf]],
                        out_dtype=tl.float32,
                    )

                                                                      
                    tlx.barrier_wait(smem_full_bars_up[buf], dot_phase)

                    w_up_trans = tlx.local_trans(w_up_buffers[buf])

                                             
                    tlx.async_dot(
                        x_buffers[buf],
                        w_up_trans,
                        tmem_up_buffers[cur_tmem_buf],
                                        
                        use_acc=(k > 0),
                        mBarriers=[smem_empty_bars[buf]],
                        out_dtype=tl.float32,
                    )

                                               
                last_buf = (processed_k_iters + k_tiles - 1) % int(NUM_SMEM_BUFFERS)
                last_total_iters = processed_k_iters + k_tiles - 1
                last_dot_phase = (last_total_iters // int(NUM_SMEM_BUFFERS)) & 1
                tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

                                                        
                                
                tlx.barrier_arrive(tmem_full_bars[cur_tmem_buf], 1)

                processed_tiles += 1
                processed_k_iters += k_tiles

                                                 
        with tlx.async_task(num_warps=1, num_regs=24):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

                                       
            processed_k_iters = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid_swiglu(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )
                offs_m = pid_m * BLOCK_M
                offs_n = pid_n * BLOCK_N

                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)

                    total_iters = processed_k_iters + k
                    load_phase = (total_iters // int(NUM_SMEM_BUFFERS)) & 1

                                                
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)

                    offs_k = k * BLOCK_K

                                                             
                    tlx.barrier_expect_bytes(
                        smem_full_bars_x_gate[buf],
                                        
                        2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N),
                    )

                                                         
                    tlx.barrier_expect_bytes(
                        smem_full_bars_up[buf],
                                        
                        2 * (BLOCK_K * BLOCK_N),
                    )

                                                                           
                    tlx.async_descriptor_load(
                        x_desc,
                        x_buffers[buf],
                        [offs_m, offs_k],
                        smem_full_bars_x_gate[buf],
                    )

                                                                              
                    tlx.async_descriptor_load(
                        w_gate_desc,
                        w_gate_buffers[buf],
                        [offs_n, offs_k],
                        smem_full_bars_x_gate[buf],
                    )

                                                                    
                    tlx.async_descriptor_load(
                        w_up_desc,
                        w_up_buffers[buf],
                        [offs_n, offs_k],
                        smem_full_bars_up[buf],
                    )

                processed_k_iters += k_tiles


@torch.fx.wrap
def triton_swiglu_fwd_tma_ws_persistent_tlx(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    N, K_gate = w_gate.shape
    N_up, K_up = w_up.shape

                                            
    supported_dtypes = (torch.bfloat16, torch.float16)
    assert x.dtype in supported_dtypes, (
        f"x.dtype must be bfloat16 or float16, got {x.dtype}"
    )
    assert w_gate.dtype in supported_dtypes, (
        f"w_gate.dtype must be bfloat16 or float16, got {w_gate.dtype}"
    )
    assert w_up.dtype in supported_dtypes, (
        f"w_up.dtype must be bfloat16 or float16, got {w_up.dtype}"
    )

    assert K == K_gate, f"Incompatible dimensions: x.K={K}, w_gate.K={K_gate}"
    assert K == K_up, f"Incompatible dimensions: x.K={K}, w_up.K={K_up}"
    assert N == N_up, f"Incompatible dimensions: w_gate.N={N}, w_up.N={N_up}"

                     
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return out

    M_BLOCK = triton.next_power_of_2(M)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

                                                              
    dummy_block = [1, 1]

                    
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
                    
    w_gate_desc = TensorDescriptor(w_gate, w_gate.shape, w_gate.stride(), dummy_block)
                    
    w_up_desc = TensorDescriptor(w_up, w_up.shape, w_up.stride(), dummy_block)
                    
    out_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ),
        )

    _swiglu_fwd_tma_ws_persistent[grid](
        x_desc,
        w_gate_desc,
        w_up_desc,
        out_desc,
        M,
        N,
        K,
        M_BLOCK,
        NUM_SMS=NUM_SMS,
        NUM_SMEM_BUFFERS=4,
        NUM_TMEM_BUFFERS=2,
        EPILOGUE_SUBTILE=2,
    )
    return out


                                                                               
                                                            
 
                                                                      
                                                                   
 
                                                                        
                                                                             
 
                                                          
                                                           
                                                                               


def _get_swiglu_fwd_configs() -> List[triton.Config]:
    '' 
                                                                    

                                                                           
                                              
       
    configs = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]
    if torch.version.hip:
        hip_num_stages = 2 if triton.__version__ >= "3.2.0" else 0
        configs.extend(
            [
                triton.Config(
                    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
            ]
        )
    return configs


@triton_autotune(
    configs=_get_swiglu_fwd_configs(),
    key=["M_BLOCK", "N", "K"],
)
@triton.jit
def _swiglu_fwd_kernel(
                                      
    x_ptr,                           
    w_gate_ptr,                                                       
    w_up_ptr,                                                     
    out_ptr,                                                 
                       
    M,                                    
    N,                                 
    K,                                         
    M_BLOCK,                                            
             
    stride_xm,
    stride_xk,
    stride_wgk,
    stride_wgn,
    stride_wuk,
    stride_wun,
    stride_om,
    stride_on,
                            
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    '' 
                                                              

                                                                  
                                                                      
       
                                                                             
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M

    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

                                                             
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
                        
    x_ptrs = (
        x_ptr
        + (pid_m.to(tl.int64) * BLOCK_M + offs_m)[:, None] * stride_xm
        + offs_k[None, :] * stride_xk
    )
                        
    wg_ptrs = (
        w_gate_ptr
        + offs_k[:, None] * stride_wgk
        + (pid_n.to(tl.int64) * BLOCK_N + offs_n)[None, :] * stride_wgn
    )

                        
    wu_ptrs = (
        w_up_ptr
        + offs_k[:, None] * stride_wuk
        + (pid_n.to(tl.int64) * BLOCK_N + offs_n)[None, :] * stride_wun
    )

                                                              
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_m & mask_k, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        wg = tl.load(wg_ptrs, mask=mask_k & mask_n, other=0.0)
        wu = tl.load(wu_ptrs, mask=mask_k & mask_n, other=0.0)

        acc_gate += tl.dot(x, wg, allow_tf32=ALLOW_TF32)
        acc_up += tl.dot(x, wu, allow_tf32=ALLOW_TF32)

        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wgk
        wu_ptrs += BLOCK_K * stride_wuk

                                                                            
    gate_activated = acc_gate * tl.sigmoid(acc_gate)        
    result = (gate_activated * acc_up).to(out_ptr.dtype.element_ty)

                                
    offs_m = pid_m * BLOCK_M + offs_m
    offs_n = pid_n * BLOCK_N + offs_n
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask_m & mask_n)


def triton_swiglu_fwd(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    '' 
                                                                           

                                               

         
                              
                                                     
                                                 

            
                            
       
    M, K = x.shape
    N, K_gate = w_gate.shape
    N_up, K_up = w_up.shape
    assert K == K_gate, f"x.K={K} != w_gate.K={K_gate}"
    assert K == K_up, f"x.K={K} != w_up.K={K_up}"
    assert N == N_up, f"w_gate.N={N} != w_up.N={N_up}"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return out

    M_BLOCK = triton.next_power_of_2(M)

                                                                 
    w_gate_t = w_gate.t().contiguous()
    w_up_t = w_up.t().contiguous()

    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _swiglu_fwd_kernel[grid](
        x,
        w_gate_t,
        w_up_t,
        out,
        M,
        N,
        K,
        M_BLOCK,
        x.stride(0),
        x.stride(1),
        w_gate_t.stride(0),
        w_gate_t.stride(1),
        w_up_t.stride(0),
        w_up_t.stride(1),
        out.stride(0),
        out.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )
    return out


def triton_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    if is_sm100_plus() and TMA_AVAILABLE and HAS_TLX:
                                                                
        _, K = x.shape
        N, _ = w_gate.shape
        assert K % 16 == 0 and N % 16 == 0, (
            f"K ({K}) and N ({N}) must be divisible by 16 for TMA alignment"
        )
        return triton_swiglu_fwd_tma_ws_persistent_tlx(x, w_gate, w_up)
    else:
                                                  
        return triton_swiglu_fwd(x, w_gate, w_up)
