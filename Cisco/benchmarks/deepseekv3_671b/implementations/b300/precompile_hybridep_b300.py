''                                                               

                                                                            
                                                                            
              
   

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist


HIDDEN_SIZE = int(os.environ.get("HYBRID_EP_HIDDEN_SIZE", "7168"))
TOKENS_PER_RANK = int(os.environ.get("HYBRID_EP_TOKENS_PER_RANK", "2048"))
NUM_EXPERTS = int(os.environ.get("HYBRID_EP_NUM_EXPERTS", "256"))
NUM_LOCAL_EXPERTS = int(os.environ.get("HYBRID_EP_NUM_LOCAL_EXPERTS", "32"))
TOPK = int(os.environ.get("HYBRID_EP_TOPK", "8"))
PAD_MULTIPLE = int(os.environ.get("HYBRID_EP_PAD_MULTIPLE", "128"))
NUM_SMS = int(os.environ.get("HYBRID_EP_NUM_SMS", "32"))


def _init_dist() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), local_rank


def _balanced_routing(local_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
                                                                             
                                                                             
                                                                        
    token_ids = torch.arange(TOKENS_PER_RANK, device="cuda", dtype=torch.int64)
    offsets = torch.arange(TOPK, device="cuda", dtype=torch.int64)
    topk_idx = (token_ids[:, None] * TOPK + offsets[None, :] + local_rank) % NUM_EXPERTS
    routing_map = torch.zeros(
        (TOKENS_PER_RANK, NUM_EXPERTS), device="cuda", dtype=torch.bool
    )
    routing_map.scatter_(1, topk_idx, True)
    probs = routing_map.to(torch.float32) / TOPK
    return routing_map, probs


def _publish_cache(local_rank: int) -> int:
    runtime_root = Path(os.environ["HYBRID_EP_CACHE_DIR"])
    process_dir = (
        runtime_root / ".deepep" / "hybrid_ep" / "jit" / f"proc-{os.getpid()}"
    )
    libraries = sorted(process_dir.glob("*.so"))
    if not libraries:
        raise RuntimeError(f"HybridEP produced no JIT libraries in {process_dir}")

    output_root = Path(os.environ["HYBRID_EP_PREBUILT_OUTPUT"])
    rank_dir = output_root / f"rank-{local_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    for library in libraries:
        shutil.copy2(library, rank_dir / library.name)
    return len(libraries)


def main() -> None:
    rank, local_rank = _init_dist()

                                                                          
    if os.environ.get("HYBRID_EP_PREBUILT_CACHE"):
        from hybridep_jit_cache import install_deep_ep_buffer_cache

        install_deep_ep_buffer_cache()

    from megatron.core.transformer.moe.fused_a2a import (
        hybrid_ep_combine,
        hybrid_ep_dispatch,
    )

    routing_map, probs = _balanced_routing(local_rank)
    hidden = torch.randn(
        (TOKENS_PER_RANK, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    dist.barrier()
    start = time.monotonic()
    dispatched, _, _, tokens_per_expert, handle = hybrid_ep_dispatch(
        x=hidden,
        routing_map=routing_map,
        probs=probs,
        group=dist.group.WORLD,
        num_local_experts=NUM_LOCAL_EXPERTS,
        num_sms_dispatch_api=NUM_SMS,
        num_sms_combine_api=NUM_SMS,
        fused=False,
        num_permuted_tokens=None,
        pad_multiple=PAD_MULTIPLE,
        num_sms_preprocessing_api=NUM_SMS,
    )
    num_permuted_tokens = tokens_per_expert.to(torch.int64).sum()
    combined = hybrid_ep_combine(
        dispatched,
        handle,
        num_permuted_tokens,
        PAD_MULTIPLE,
        fused=False,
    )
    combined.float().sum().backward()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.monotonic() - start

    if os.environ.get("HYBRID_EP_PREBUILT_OUTPUT"):
        count = _publish_cache(local_rank)
        print(
            f"rank={rank} local_rank={local_rank} published={count} "
            f"elapsed_s={elapsed:.3f}",
            flush=True,
        )
    else:
        print(
            f"rank={rank} local_rank={local_rank} cache_verify=PASS "
            f"elapsed_s={elapsed:.3f}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

