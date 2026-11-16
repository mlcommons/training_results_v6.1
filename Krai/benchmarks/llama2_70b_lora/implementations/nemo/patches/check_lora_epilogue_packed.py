"""lora_dropout_packed on top of hs_lora_epilogue (fix #2): bit-exactness against fix #2 alone, on one GPU.

Run inside the training image (one GPU is enough):
    cd /workspace/trainloom_patches && python check_lora_epilogue_packed.py
Compares hs_lora_epilogue's own _LoraDown and dgrad fold with the packed versions this repo swaps in: the LoRA
down-projection output, the LoRA A weight gradient, the input gradient (plain path), and the dgrad fold, for the
qkv adapter (dropped input kept) and the dense adapter (dropped input rebuilt from its input).
"""

import sys

import torch

import hs_lora_epilogue as hle

stock_down, stock_acc = hle._LoraDown, hle.lora_dgrad_acc_
import lora_dropout_packed as ldp  # noqa: E402  (swaps hle._LoraDown and hle.lora_dgrad_acc_)

P, M, N, K = 0.1, 8192, 8192, 16


def same(a, b, what):
    """Bit equality (NaN payloads included), not numeric closeness."""
    ok = torch.equal(a.view(torch.int16), b.view(torch.int16))
    diff = "" if ok else f", max abs diff {(a.float() - b.float()).abs().max().item()}"
    print(f"{what}: {'bit-identical' if ok else 'DIFFERENT'}{diff}")
    return ok


def down(cls, x, a_w, seed, dense):
    ldp._tls.dense = dense
    torch.manual_seed(seed)
    x = x.clone().requires_grad_()
    w = a_w.clone().requires_grad_()
    a = cls.apply(x, w, P, True, None)
    ga = torch.randn_like(a)
    a.backward(ga)
    ldp._tls.dense = False
    return a.detach(), x.grad, w.grad


def main():
    assert hle._LoraDown is not stock_down and hle.lora_dgrad_acc_ is not stock_acc, "integration not installed"
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 4
    a_w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.05
    ok = True
    for dense in (False, True):
        kind = "dense (rebuilt input)" if dense else "qkv (kept input)"
        ref, new = down(stock_down, x, a_w, 1, dense), down(hle._LoraDown, x, a_w, 1, dense)
        ok &= same(new[0], ref[0], f"{kind}: LoRA down output")
        ok &= same(new[1], ref[1], f"{kind}: input gradient")
        ok &= same(new[2], ref[2], f"{kind}: LoRA A weight gradient")

    torch.manual_seed(2)
    _, mask = torch.native_dropout(x, P, True)
    ga = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    dgrad = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref = stock_acc(dgrad.clone(), ga, a_w, mask, 1.0 / (1.0 - P))
    new = hle.lora_dgrad_acc_(dgrad.clone(), ga, a_w, ldp.pack_mask(mask), 1.0 / (1.0 - P))
    ok &= same(new, ref, "dgrad fold with packed mask")

    packed = ldp.pack_mask(mask)
    for name, fn in (("dgrad fold, fix 2 (bool mask)", lambda: stock_acc(dgrad.clone(), ga, a_w, mask, 1.0 / (1.0 - P))),
                     ("dgrad fold, packed mask", lambda: hle.lora_dgrad_acc_(dgrad.clone(), ga, a_w, packed, 1.0 / (1.0 - P))),
                     ("clone alone (subtract from both)", lambda: dgrad.clone())):
        print(f"{name}: {timed(fn):.3f} ms")
    sys.exit(0 if ok else 1)


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


if __name__ == "__main__":
    main()
