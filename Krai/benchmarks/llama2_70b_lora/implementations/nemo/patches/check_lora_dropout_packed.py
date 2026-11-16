"""Bit-exactness and speed of lora_dropout_packed's kernels against torch's own dropout, on one GPU.

Run inside the training image (one GPU is enough):
    cd /workspace/trainloom_patches && python check_lora_dropout_packed.py
It imports only the kernels and the rebuild hook, not NeMo's adapter, so no model-parallel setup is needed.
"""

import sys
import types

import torch

# The module patches NeMo on import; the check needs only its kernels, so give it stand-ins for NeMo.
stub = types.ModuleType("parallel_adapters")
stub.ParallelLinearAdapter = type("ParallelLinearAdapter", (), {"forward": lambda self, x: x})
stub.LoraDenseAttentionAdapter = type("LoraDenseAttentionAdapter", (stub.ParallelLinearAdapter,), {})
sys.modules["nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters"] = stub
import lora_dropout_packed as ldp  # noqa: E402

P, SCALE, SHAPE = 0.1, 1.0 / 0.9, (8192, 1, 8192)


def same(a, b, what):
    """Bit equality (NaN payloads included), not numeric closeness."""
    ok = torch.equal(a.view(torch.int16), b.view(torch.int16))
    diff = "" if ok else f", max abs diff {(a.float() - b.float()).abs().max().item()}"
    print(f"{what}: {'bit-identical' if ok else 'DIFFERENT'}{diff}")
    return ok


def timed(fn, n=20):
    fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def main():
    torch.manual_seed(0)
    x = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16) * 4
    grad = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)
    out, mask = torch.native_dropout(x, P, True)
    packed = ldp.pack_mask(mask)
    ok = same(ldp.masked_scale(x, packed, SCALE), out, "dropout output rebuilt from packed mask")
    ok &= same(
        ldp.masked_scale(grad, packed, SCALE),
        torch.ops.aten.native_dropout_backward(grad, mask, SCALE),
        "dropout backward from packed mask",
    )

    weight = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xs = [x.clone().requires_grad_(), x.clone().requires_grad_()]
    torch.manual_seed(1)
    dropped = torch.nn.functional.dropout(xs[0], P, True)
    torch.nn.functional.linear(dropped, weight).float().sum().backward()
    stock = (xs[0].grad.clone(), weight.grad.clone())
    weight.grad = None
    torch.manual_seed(1)
    holder = {}
    dropped = ldp._PackedDropout.apply(xs[1], P, holder)

    def pack(t):
        if t.data_ptr() == dropped.data_ptr():
            return ldp._Rebuild(xs[1], holder["packed"], SCALE, t.shape)
        return t

    def unpack(saved):
        return ldp.masked_scale(saved.x, saved.packed, saved.scale).view(saved.shape) if isinstance(saved, ldp._Rebuild) else saved

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = torch.nn.functional.linear(dropped, weight)
    y.float().sum().backward()
    ok &= same(xs[1].grad, stock[0], "input gradient, dense path (rebuilt dropped input)")
    ok &= same(weight.grad, stock[1], "LoRA A weight gradient, dense path")

    print(f"native_dropout fwd {timed(lambda: torch.native_dropout(x, P, True)):.3f} ms")
    print(f"pack_mask          {timed(lambda: ldp.pack_mask(mask)):.3f} ms")
    print(f"masked_scale       {timed(lambda: ldp.masked_scale(grad, packed, SCALE)):.3f} ms")
    print(f"native bwd (stock) {timed(lambda: torch.ops.aten.native_dropout_backward(grad, mask, SCALE)):.3f} ms")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
