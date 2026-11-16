''                                                                            
from datetime import timedelta
import os
import time
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl', rank=rank, world_size=world, timeout=timedelta(seconds=60),
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT'])+1}")

def measure(name, operation):
    for _ in range(2):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.monotonic()
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    elapsed = torch.tensor([(time.monotonic()-start)/5], device='cuda', dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f'NCCL_PREFLIGHT {name} max_rank_ms={elapsed.item()*1000:.3f}', flush=True)

x = torch.empty(10*1024*1024, device='cuda', dtype=torch.float32)
def allreduce():
    x.fill_(1)
    dist.all_reduce(x)
measure('allreduce_40MiB', allreduce)
assert bool((x == world).all()), 'All-reduce validation failed'
del x

send = torch.full((world*16384,), float(rank), device='cuda', dtype=torch.float32)
receive = torch.empty_like(send)
def alltoall():
    dist.all_to_all_single(receive, send)
measure('alltoall_4MiB', alltoall)
expected = torch.arange(world, device='cuda', dtype=torch.float32).view(world, 1)
assert bool((receive.view(world, -1) == expected).all()), 'All-to-all validation failed'
dist.barrier()
if rank == 0:
    print(f'NCCL_PREFLIGHT_PASS world_size={world} allreduce=PASS alltoall=PASS', flush=True)
dist.destroy_process_group()
