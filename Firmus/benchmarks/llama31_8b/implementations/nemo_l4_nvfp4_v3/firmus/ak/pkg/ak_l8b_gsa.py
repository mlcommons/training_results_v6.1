"""auto_kernel GSA (Llama 3.1 8B campaign, 2026-10-02): the gradient's 1/DP scaling done by the collective.

Stock (the recipe passes average_in_collective=False, pretrain.py:285): before each bucket's reduce-scatter Megatron
multiplies the whole bucket by 1/8 (param_and_grad_buffer.py:712-713, `bucket.grad_data *= gradient_scaling_factor`;
10 launches x 470 us per step on the compute stream, PROF-L0) and reduce-scatters with SUM.
With AK_GSA=1: average_in_collective=True, so DistributedDataParallel sets gradient_scaling_factor = 1.0 (no multiply)
and the collective is ReduceOp.AVG, which NCCL applies as a pre-multiply by 1/world of each rank's contribution.
rs_avg_test.py measured the two byte-identical at world 4 on every pattern tried (subnormals, +/-0, inf/NaN).

The first eager gradient sync (warm-up iteration 0) also checks it on the real data-parallel group: the first bucket
reduce-scattered both ways (a clone pre-scaled by 1/world + SUM, and a clone with AVG), every rank's shard compared
as bytes, the mismatch count all-reduced; rank 0 prints the proof line and a mismatch raises.

Loaded by `import firmus_ak` (pretrain.py's first import); armed by AK_GSA=1 alone.
"""
import os
import sys

_ARMED = os.environ.get("AK_GSA", "0") == "1"
_STATE = {"forced": 0, "checked": False}

if _ARMED:
    import importlib.abc
    import importlib.util

    def _patch_ddp(mod):
        # pretrain.py builds Megatron-Bridge's DistributedDataParallelConfig (a dataclass subclass with its own generated
        # __init__), so the flag is set where it is consumed: on the config DistributedDataParallel.__init__ receives,
        # before it derives gradient_scaling_factor from it (distributed_data_parallel.py:281-286).
        cls = mod.DistributedDataParallel
        orig_init = cls.__init__

        def __init__(self, *a, **k):
            ddp_config = k.get("ddp_config") if "ddp_config" in k else (a[1] if len(a) > 1 else None)
            if ddp_config is not None and not ddp_config.average_in_collective:
                ddp_config.average_in_collective = True
                _STATE["forced"] += 1
            orig_init(self, *a, **k)
        __init__.__wrapped__ = orig_init
        cls.__init__ = __init__

    def _check(group_obj):
        import torch
        import torch.distributed as dist
        g = group_obj.intra_distributed_optimizer_instance_group
        world = dist.get_world_size(g)
        b = group_obj.buckets[0]
        x = b.grad_data
        n = x.numel() - (x.numel() % world)
        x = x[:n]
        a = x.clone()
        a *= 1.0 / world                                   # the stock path's pre-scale (gradient_scaling_factor 1/world)
        out_a = torch.empty(n // world, dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out_a, a, op=dist.ReduceOp.SUM, group=g)
        c = x.clone()
        out_b = torch.empty_like(out_a)
        dist.reduce_scatter_tensor(out_b, c, op=dist.ReduceOp.AVG, group=g)
        diff = (out_a.view(torch.int16) != out_b.view(torch.int16)).sum().to(torch.int64)
        nz = (out_a != 0).sum().to(torch.int64)
        stats = torch.stack([diff, nz])
        dist.all_reduce(stats, group=g)
        ok = int(stats[0]) == 0
        if dist.get_rank() == 0:
            print("[auto_kernel gsa] gradient 1/DP scaling in the collective ARMED: average_in_collective=True ({} config(s) "
                  "forced), gradient_scaling_factor 1.0, ReduceOp.AVG; first bucket ({} elements, {} bf16) pre-scale 1/{} + "
                  "SUM v AVG over {} ranks: bytes differ in {} of {} shard elements ({} non-zero): {}".format(
                      _STATE["forced"], n, x.dtype, world, world, int(stats[0]), n, int(stats[1]), ok), flush=True)
        del a, c, out_a, out_b
        if not ok:
            raise RuntimeError("[auto_kernel gsa] E0 self-check FAILED: %d bytes differ" % int(stats[0]))

    def _patch_buffer(mod):
        cls = mod._ParamAndGradBucketGroup
        orig = cls.start_grad_sync

        def start_grad_sync(self, *a, **k):
            if not _STATE["checked"]:
                import torch
                if not torch.cuda.is_current_stream_capturing():
                    _STATE["checked"] = True
                    if not self.ddp_config.average_in_collective:
                        raise RuntimeError("[auto_kernel gsa] armed but average_in_collective is False at the first sync")
                    _check(self)
            return orig(self, *a, **k)
        start_grad_sync.__wrapped__ = orig
        cls.start_grad_sync = start_grad_sync

    _HOOKS = {"megatron.core.distributed.distributed_data_parallel": _patch_ddp,
              "megatron.core.distributed.param_and_grad_buffer": _patch_buffer}

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            fn = _HOOKS.get(fullname)
            if fn is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec, _fn=fn):
                _orig(module)
                _fn(module)
            spec.loader.exec_module = exec_module
            return spec

    for _name, _fn in list(_HOOKS.items()):
        if _name in sys.modules:
            _fn(sys.modules[_name])
            _HOOKS.pop(_name)
    if _HOOKS:
        sys.meta_path.insert(0, _Finder())
