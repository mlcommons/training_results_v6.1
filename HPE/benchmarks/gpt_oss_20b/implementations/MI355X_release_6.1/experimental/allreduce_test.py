#!/usr/bin/env python3
"""8-GPU all-reduce through torch.distributed/RCCL. Run with torchrun."""
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(64 << 20, device="cuda", dtype=torch.bfloat16)  # 128 MB
for _ in range(3):
    dist.all_reduce(x)
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t) / 20
if r == 0:
    gb = x.numel() * 2 / 1e9
    print(f"ALLREDUCE OK cumem={os.environ.get('NCCL_CUMEM_ENABLE')} 128MB bf16: {dt*1e3:.2f} ms, busbw {2*7/8*gb/dt:.1f} GB/s", flush=True)
dist.destroy_process_group()
