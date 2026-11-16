#!/usr/bin/env python3
"""test_lmh.py (Llama 3.1 8B campaign, 2026-10-08, helper line lmh29): the LM head's pinned cuBLASLt GEMMs.

    python3 test_lmh.py enum <ext dir> [<pin json out>]   the heuristic's list per problem, each entry's kernel (profiler),
                                                          bytes v torch and time; writes the pin (the target kernels' bytes)
    python3 test_lmh.py test <ext dir>                    with <ext dir>/lmh_pin.json: byte equality v torch, eager and under
                                                          CUDA-graph replay, determinism, the patch end to end through
                                                          Megatron's linear (stock v patched), the negative control
The recipe's problems (one GPU's micro-batch: MBS 2 x seq 8192 = 16384 tokens): forward x[16384, 4096] @ W[128256, 4096]^T
and dgrad dy[16384, 128256] @ W.  Prints "=== lmh FAILURES n" last."""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

V, H, T = 128256, 4096, 16384
TARGET = {0: "nvjet_sm103_tst_256x256_64x4_2x2f_2cta_h_bz_TNT", 1: "nvjet_sm103_tst_128x512_64x4_2x1_2cta_v_bz_NNT"}
NAMES = ("forward", "dgrad")
FAIL = []


def load_ext(d):
    import importlib.util
    path = os.path.join(d, "ak_l8b_lmh.so")
    spec = importlib.util.spec_from_file_location("ak_l8b_lmh", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print("[lmh] module %s sha256 %s" % (path, hashlib.sha256(open(path, "rb").read()).hexdigest()), flush=True)
    return m


def same(a, b):
    return bool(torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)))


def gen(shape, seed, kind, dev, scale=1.0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(*shape, generator=g, device=dev) * scale
    if kind == "wide":
        e = torch.randint(-20, 13, (shape[0], 1), generator=g, device=dev).float()
        x = x * torch.pow(2.0, e)
        x[:: max(1, shape[0] // 7)] = 0.0
    elif kind == "special":
        x = x.clone()
        x[1] = float("nan")
        x[2] = float("inf")
        x[3] = float("-inf")
        x[4] = -0.0
        x[5] = x[5] * 1e-39                            # bf16 subnormals
        x[6, ::3] = float("nan")
    return x.to(torch.bfloat16)


def recipe_like(dev, seed):
    """x like the final RMSNorm's output (unit rows times a gamma near 1), W the recipe's init (std 0.02), dy the
    cross-entropy gradient's shape: (softmax(logits) - one_hot) / tokens"""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(V, H, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(T, H, generator=g, device=dev)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * (1.0 + 0.1 * torch.randn(H, generator=g, device=dev))
    x = x.to(torch.bfloat16)
    logits = torch.matmul(x, w.t()).float()
    p = torch.softmax(logits, -1)
    del logits
    tgt = torch.randint(0, V, (T,), generator=g, device=dev)
    p[torch.arange(T, device=dev), tgt] -= 1.0
    dy = (p / T).to(torch.bfloat16)
    del p
    return x, w, dy


def ref(kind, a, w):
    return torch.matmul(a, w.t()) if kind == 0 else a.matmul(w)


def kernels_of(fn):
    from torch.profiler import profile, ProfilerActivity
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if getattr(e, "device_type", None) is not None
                   and str(e.device_type).endswith("CUDA")})


def time_graph(fn, n=5, reps=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / n)
    return sorted(ts)[len(ts) // 2]


def problem(kind, dev, seed=1):
    w = gen((V, H), seed + 7, "gauss", dev, 0.02)
    a = gen((T, H) if kind == 0 else (T, V), seed, "gauss", dev, 1.0 if kind == 0 else 0.01)
    out = torch.empty(T, V if kind == 0 else H, dtype=torch.bfloat16, device=dev)
    return w, a, out


def cublaslt_version():
    import ctypes
    for n in ("libcublasLt.so", "libcublasLt.so.13", "libcublasLt.so.12"):
        try:
            return int(ctypes.CDLL(n).cublasLtGetVersion())
        except OSError:
            continue
    return None


def enum(d, pin_out):
    dev = torch.device("cuda", 0)
    ext = load_ext(d)
    ws = torch.empty(32 * 1024 * 1024 + 1024, dtype=torch.uint8, device=dev)
    pin = {"cublasLt": cublaslt_version(), "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "T": T,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    import megatron.core.tensor_parallel.layers as L
    import inspect
    src = inspect.getsource(inspect.unwrap(L.LinearWithGradAccumulationAndAsyncCommunication.backward))
    pin["bwd_src_sha256"] = hashlib.sha256(src.encode()).hexdigest()
    pin["bwd_src_dgrad_lines"] = src.count("grad_input = grad_output.matmul(weight)")
    print("[lmh] stock backward source sha256 %s, dgrad lines %d" % (pin["bwd_src_sha256"], pin["bwd_src_dgrad_lines"]))
    for kind in (0, 1):
        w, a, out = problem(kind, dev)
        lst = ext.heuristics(kind, w, a, out, ws, 64)
        lst2 = ext.heuristics(kind, w, a, out, ws, 64)
        stable = [e[0] for e in lst] == [e[0] for e in lst2]
        r = ref(kind, a, w)
        t_torch = time_graph(lambda: ref(kind, a, w))
        k_torch = kernels_of(lambda: ref(kind, a, w))
        print("[lmh] %s: %d algorithms (list stable across calls: %s); torch: %s %.1f us" % (
            NAMES[kind], len(lst), stable, k_torch, t_torch), flush=True)
        if not stable:
            FAIL.append("%s list unstable" % NAMES[kind])
        rows = []
        for i, (blob, wsz, waves, st) in enumerate(lst):
            ext.gemm(kind, w, a, out, ws, blob)
            torch.cuda.synchronize()
            eq = same(out, r)
            ks = kernels_of(lambda: ext.gemm(kind, w, a, out, ws, blob))
            t = time_graph(lambda: ext.gemm(kind, w, a, out, ws, blob))
            rows.append({"index": i, "algo": blob.hex(), "algo_sha": hashlib.sha256(blob).hexdigest(), "kernels": ks,
                         "equal_torch": eq, "time_us": t, "workspace": wsz, "waves": waves})
            print("[lmh]   #%d %s ws %d waves %.3f: %s %.1f us, byte-equal to torch %s" % (
                i, rows[-1]["algo_sha"][:16], wsz, waves, ks, t, eq), flush=True)
        hit = [x for x in rows if TARGET[kind] in x["kernels"]]
        if not hit:
            FAIL.append("%s: no algorithm launches %s" % (NAMES[kind], TARGET[kind]))
            continue
        h = hit[0]
        pin["%s_T%d" % (NAMES[kind], T)] = {"algo": h["algo"], "algo_sha": h["algo_sha"], "kernel": TARGET[kind],
                                           "index": h["index"], "count": len(lst), "time_us": h["time_us"],
                                           "time_torch_us": t_torch, "torch_kernels": k_torch,
                                           "equal_torch": h["equal_torch"], "list": rows}
        print("[lmh] %s pin: #%d %s (%s) %.1f us v torch %.1f us" % (
            NAMES[kind], h["index"], h["algo_sha"][:16], TARGET[kind], h["time_us"], t_torch), flush=True)
        del w, a, out, r
        torch.cuda.empty_cache()
    if pin_out:
        json.dump(pin, open(pin_out, "w"), indent=1)
        print("[lmh] wrote %s" % pin_out)


def test(d):
    dev = torch.device("cuda", 0)
    ext = load_ext(d)
    pin = json.load(open(os.path.join(d, "lmh_pin.json")))
    ws = torch.empty(32 * 1024 * 1024 + 1024, dtype=torch.uint8, device=dev)
    algo = {}
    for kind in (0, 1):
        p = pin["%s_T%d" % (NAMES[kind], T)]
        w, a, out = problem(kind, dev, 11)
        lst = ext.heuristics(kind, w, a, out, ws, 64)
        idx = [i for i, e in enumerate(lst) if e[0].hex() == p["algo"]]
        ks = kernels_of(lambda: ext.gemm(kind, w, a, out, ws, bytes.fromhex(p["algo"])))
        ok = bool(idx) and p["kernel"] in ks
        print("[lmh] %s: pinned algorithm %s in this process's list at %s of %d; kernel %s: %s" % (
            NAMES[kind], p["algo_sha"][:16], idx, len(lst), ks, ok), flush=True)
        if not ok:
            FAIL.append("%s: pin not found or kernel differs" % NAMES[kind])
        algo[kind] = bytes.fromhex(p["algo"])
        del w, a, out
    # byte equality v torch on every input kind, eager
    cases = [("gauss", s) for s in (1, 2, 3)] + [("wide", 4), ("special", 5)]
    for kind in (0, 1):
        for typ, seed in cases:
            w = gen((V, H), seed + 7, "wide" if typ == "wide" else "gauss", dev, 0.02)
            a = gen((T, H) if kind == 0 else (T, V), seed, typ, dev, 1.0 if kind == 0 else 0.01)
            out = torch.empty(T, V if kind == 0 else H, dtype=torch.bfloat16, device=dev)
            ext.gemm(kind, w, a, out, ws, algo[kind])
            o1 = out.clone()
            ext.gemm(kind, w, a, out, ws, algo[kind])
            det = same(o1, out)
            r = ref(kind, a, w)
            eq = same(out, r)
            nd = int((out.view(torch.int16) != r.view(torch.int16)).sum())
            print("[lmh] %s %-8s seed %d: byte-equal to torch %s (%d differing elements), deterministic %s" % (
                NAMES[kind], typ, seed, eq, nd, det), flush=True)
            if not (eq and det):
                FAIL.append("%s %s %d" % (NAMES[kind], typ, seed))
            del w, a, out, o1, r
            torch.cuda.empty_cache()
    # the recipe-like data (two seeds): forward on the norm output, dgrad on the cross-entropy gradient
    for seed in (21, 22):
        x, w, dy = recipe_like(dev, seed)
        for kind, a in ((0, x), (1, dy)):
            out = torch.empty(T, V if kind == 0 else H, dtype=torch.bfloat16, device=dev)
            ext.gemm(kind, w, a, out, ws, algo[kind])
            eq = same(out, ref(kind, a, w))
            print("[lmh] %s recipe-like seed %d: byte-equal to torch %s" % (NAMES[kind], seed, eq), flush=True)
            if not eq:
                FAIL.append("%s recipe-like %d" % (NAMES[kind], seed))
            del out
        del x, w, dy
        torch.cuda.empty_cache()
    # under CUDA-graph replay: one capture per problem on static buffers, three new inputs copied in
    for kind in (0, 1):
        w = gen((V, H), 31, "gauss", dev, 0.02)
        a = gen((T, H) if kind == 0 else (T, V), 32, "gauss", dev, 1.0 if kind == 0 else 0.01)
        out = torch.empty(T, V if kind == 0 else H, dtype=torch.bfloat16, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            ext.gemm(kind, w, a, out, ws, algo[kind])
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ext.gemm(kind, w, a, out, ws, algo[kind])
        for j, typ in enumerate(("gauss", "wide", "gauss")):
            w.copy_(gen((V, H), 40 + j, typ, dev, 0.02))
            a.copy_(gen((T, H) if kind == 0 else (T, V), 50 + j, typ, dev, 1.0 if kind == 0 else 0.01))
            g.replay()
            torch.cuda.synchronize()
            eq = same(out, ref(kind, a, w))
            print("[lmh] %s graph replay %d (%s): byte-equal to torch %s" % (NAMES[kind], j, typ, eq), flush=True)
            if not eq:
                FAIL.append("%s graph replay %d" % (NAMES[kind], j))
        del g, w, a, out
        torch.cuda.empty_cache()
    # time per call, ours v torch (CUDA graphs of 5 calls, median of 3)
    for kind in (0, 1):
        w, a, out = problem(kind, dev, 61)
        t_o = time_graph(lambda: ext.gemm(kind, w, a, out, ws, algo[kind]))
        t_t = time_graph(lambda: ref(kind, a, w))
        print("[lmh] %s time per call: ours %.1f us, torch %.1f us (%.2f %%)" % (NAMES[kind], t_o, t_t, 100.0 * (t_o / t_t - 1)))
        del w, a, out
        torch.cuda.empty_cache()
    patch_test(d)


def patch_test(d):
    """the patch end to end: Megatron's linear_with_grad_accumulation_and_async_allreduce (the LM head's call, gradient
    accumulation fusion onto a bf16 main_grad) stock, then patched, from the same inputs; then the negative control"""
    dev = torch.device("cuda", 0)
    import megatron.core.tensor_parallel.layers as L
    stock_bwd = L.LinearWithGradAccumulationAndAsyncCommunication.backward

    def run(seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        w = torch.nn.Parameter((torch.randn(V, H, generator=g, device=dev) * 0.02).to(torch.bfloat16))
        w.main_grad = torch.zeros(V, H, dtype=torch.bfloat16, device=dev)
        x = torch.randn(8192, 2, H, generator=g, device=dev).to(torch.bfloat16).requires_grad_(True)
        dy = (torch.randn(8192, 2, V, generator=g, device=dev) * 1e-4).to(torch.bfloat16)
        y = L.linear_with_grad_accumulation_and_async_allreduce(x, w, None, True, False, False)
        y.backward(dy)
        torch.cuda.synchronize()
        return y.detach().clone(), x.grad.clone(), w.main_grad.clone()

    stock = [run(s) for s in (71, 72)]
    os.environ.update({"AK_LMH": "1", "AK_LMH_DIR": d})
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ak_l8b_lmh_patch as P
    print("[lmh] patch: armed %s, dgrad installed %s %s; backward replaced %s" % (
        P._ARMED, P._S["dgrad_installed"], P._S["why"],
        L.LinearWithGradAccumulationAndAsyncCommunication.backward is not stock_bwd), flush=True)
    if not P._S["dgrad_installed"]:
        FAIL.append("patch: dgrad not installed")
    for i, s in enumerate((71, 72)):
        got = run(s)
        eqs = [same(a, b) for a, b in zip(stock[i], got)]
        print("[lmh] patch seed %d: output %s, input gradient %s, main_grad %s (stock v patched, byte for byte)" % (
            s, *eqs), flush=True)
        if not all(eqs):
            FAIL.append("patch seed %d %s" % (s, eqs))
    print("[lmh] patch calls: forward %d, dgrad %d; checked %s" % (P._S["calls"][0], P._S["calls"][1], sorted(P._S["checked"])))
    if P._S["calls"] != [2, 2]:
        FAIL.append("patch calls %s" % P._S["calls"])
    # the negative control: one corrupted output byte in a fresh first call must raise, and leave the problem unchecked
    real = P._S["ext"]

    class _Corrupt:
        def __getattr__(self, k):
            return getattr(real, k)

        def gemm(self, kind, w, a, out, ws, algo):
            real.gemm(kind, w, a, out, ws, algo)
            out.view(-1)[:1].view(torch.int16).bitwise_xor_(0x0001)

    P._S["ext"] = _Corrupt()
    P._S["checked"].clear()
    raised = False
    try:
        run(73)
    except RuntimeError as e:
        raised = "E0 self-check FAILED for the LM-head" in str(e)
    finally:
        P._S["ext"] = real
    neg = raised and not P._S["checked"]
    print("[lmh] negative control: a corrupted output raises and leaves the problem unchecked: %s" % neg, flush=True)
    if not neg:
        FAIL.append("negative control")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _ap.add_argument("mode", help="enum or test")
    _ap.add_argument("d", help="the directory holding ak_l8b_lmh.so (and, for test, lmh_pin.json)")
    _ap.add_argument("pin_out", nargs="?", default=None, help="enum: the pin json to write")
    _args = _ap.parse_args()
    mode, d = _args.mode, _args.d
    if mode == "enum":
        enum(d, _args.pin_out)
    else:
        test(d)
    print("=== lmh FAILURES %d %s" % (len(FAIL), FAIL), flush=True)
    sys.exit(1 if FAIL else 0)
