"""auto_kernel LMH (Llama 3.1 8B campaign, 2026-10-08, helper line lmh29): the LM head's BF16 forward and dgrad GEMMs
through cuBLASLt with a pinned algorithm -- the same bytes as torch's matmul, less energy under the power cap.

The recipe's LM head is Megatron's output layer (tensor_parallel.ColumnParallelLinear, no logit dtype, TP 1):
  forward  _linear_forward(x, W, None, None)       -> torch.matmul(x, W.t())   [8192, 2, 4096] x [128256, 4096]^T
  dgrad    LinearWithGradAccumulationAndAsyncCommunication.backward: grad_input = grad_output.matmul(weight)
  wgrad    APEX wgrad_gemm_accum_fp16 onto main_grad (unchanged here).
ge28 (EXPERIMENT_LOG sec L16.5) measured, with TE's bf16 cuBLASLt descriptors for the same problems, a heuristic entry that
is byte-equal to torch's output and cheaper under the cap: the forward's nvjet_sm103_tst_256x256_64x4_2x2f_2cta_h_bz_TNT
(-3.75 % J per call) and the dgrad's nvjet_sm103_tst_128x512_64x4_2x1_2cta_v_bz_NNT (-1.25 %).  ak_l8b_lmh.so builds TE's
descriptors and runs one cublasLtMatmul with a given algorithm.  The algorithms are opaque (no config attribute reads),
so each is PINNED BY ITS 64 BYTES, recorded in lmh_pin.json at build time together with the kernel it launched (the unit
test resolves the kernel by name with the profiler and records the bytes).  At a problem's first call the patch asks the
heuristic for the problem's list and takes the entry whose bytes are the pinned ones; if none is, the patch raises
(L4: an armed switch never falls back silently; a T without a pin runs Megatron's code).

Only the recipe's case takes it: bf16 contiguous CUDA tensors, W [V, H] with V = 128256 and H = 4096, T = numel / H rows
(the pin is per (kind, T)), no bias, no output dtype; anything else runs Megatron's code.  The first call of each problem
(eager -- the training warm-up's iteration 0, and the validation warm-up's) compares the output with torch's matmul byte for
byte and raises on a mismatch; a pinned problem first met under CUDA-graph capture raises (its check never ran).  The
dgrad is reached through a copy of the stock backward with that one line changed, compiled from the stock source only
when its sha256 is the one recorded here (the image's Megatron); else the patch raises (AK_LMH_DGRAD=0 keeps it stock).

Proof lines: "[auto_kernel lmh] LM-head forward through cuBLASLt, the pinned algorithm ARMED: ... True" and
"[auto_kernel lmh] LM-head dgrad through cuBLASLt, the pinned algorithm ARMED: ... True".  Loaded by `import firmus_ak`;
armed by AK_LMH=1 alone; AK_LMH_DIR holds ak_l8b_lmh.so and lmh_pin.json; AK_LMH_DGRAD=0 keeps the
dgrad stock.
"""
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import os
import sys
import textwrap

_ARMED = os.environ.get("AK_LMH", "0") == "1"
_DGRAD_ON = _ARMED and os.environ.get("AK_LMH_DGRAD", "1") == "1"
_DIR = os.environ.get("AK_LMH_DIR", "/opt/firmus/ak/lmh")
V_LM, H_LM = 128256, 4096
# the dgrad copy is made only from the stock backward whose source sha256 (inspect.getsource of
# LinearWithGradAccumulationAndAsyncCommunication.backward in the image's Megatron, /workspace/Megatron-Bridge/3rdparty/
# Megatron-LM, llama31_8b-arm:20260929) is lmh_pin.json's "bwd_src_sha256", recorded in that image when the pin was made
DGRAD_LINE = "grad_input = grad_output.matmul(weight)"

_S = {"ext": None, "ext_sha": None, "pin": None, "ws": {}, "algo": {}, "checked": set(), "stock": set(), "calls": [0, 0],
      "dgrad_installed": False, "why": ""}
_NAMES = ("forward", "dgrad")


def _rank():
    try:
        import torch
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get("RANK", "0"))
    except Exception:  # noqa: BLE001
        return 0


def _say(msg):
    print(msg, flush=True)


def _ext():
    if _S["ext"] is None:
        path = os.path.join(_DIR, "ak_l8b_lmh.so")
        spec = importlib.util.spec_from_file_location("ak_l8b_lmh", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _S["ext"] = mod
        _S["ext_sha"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        _S["pin"] = json.load(open(os.path.join(_DIR, "lmh_pin.json")))
    return _S["ext"]


def _workspace(dev):
    import torch
    ws = _S["ws"].get(dev)
    if ws is None:
        # TE's cuBLAS workspace size on sm >= 9: 32 MiB + 1 KiB (cpp_extensions/gemm.py)
        ws = torch.empty(32 * 1024 * 1024 + 1024, dtype=torch.uint8, device=dev)
        _S["ws"][dev] = ws
    return ws


def _bytes(t):
    import torch
    return t.detach().reshape(-1).view(torch.uint8)


def _case(kind, a, w):
    """(T, a2d) when this call is the recipe's LM-head case, else None"""
    import torch
    if not (a.is_cuda and w.is_cuda and a.dtype == torch.bfloat16 and w.dtype == torch.bfloat16):
        return None
    if w.dim() != 2 or tuple(w.shape) != (V_LM, H_LM) or not w.is_contiguous() or not a.is_contiguous():
        return None
    last = H_LM if kind == 0 else V_LM
    if a.dim() < 2 or a.shape[-1] != last or a.numel() % last:
        return None
    return a.numel() // last, a.reshape(-1, last)


def _run(kind, a2d, w, out2d):
    """our GEMM into out2d when (kind, T) has its pinned algorithm armed; False when the problem stays stock"""
    import torch
    T = a2d.shape[0]
    key = (kind, T, w.device.index)
    if key in _S["stock"]:
        return False
    ext = _ext()
    ws = _workspace(w.device)
    algo = _S["algo"].get(key)
    if algo is None:
        pin = _S["pin"].get("%s_T%d" % (_NAMES[kind], T))
        if pin is None:                       # no pin for this T: Megatron's code (shape routing)
            _S["stock"].add(key)
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[auto_kernel lmh] LM-head %s (T %d) first called under graph capture: its check "
                               "never ran" % (_NAMES[kind], T))
        lst = ext.heuristics(kind, w, a2d, out2d, ws, 64)
        want = bytes.fromhex(pin["algo"])
        idx = [i for i, e in enumerate(lst) if e[0] == want]
        if not idx:
            raise RuntimeError("[auto_kernel lmh] LM-head %s through cuBLASLt NOT ARMED (T %d): the pinned algorithm %s "
                               "is not in the heuristic's list of %d" % (_NAMES[kind], T, pin["algo_sha"][:16], len(lst)))
        algo = want
        _S["algo"][key] = algo
        _S["pin_index"] = idx[0]
        _S.setdefault("list_len", {})[key] = len(lst)
    ext.gemm(kind, w, a2d, out2d, ws, algo)
    _S["calls"][kind] += 1
    return True


def _first_check(kind, key, ours, ref_fn, shape_txt):
    import torch
    if key in _S["checked"]:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("ak_l8b_lmh: first call of %s under graph capture" % (key,))
    ref = ref_fn()
    ok = bool(torch.equal(_bytes(ours), _bytes(ref)))
    del ref
    pin = _S["pin"]["%s_T%d" % (_NAMES[kind], key[1])]
    if _rank() == 0 or not ok:
        _say("[auto_kernel lmh] LM-head %s through cuBLASLt, the pinned algorithm ARMED: %s; heuristic entry %d of %d, "
             "algorithm %s (pinned: %s); module ak_l8b_lmh.so sha256=%s; byte-exact v torch's matmul: %s" % (
                 _NAMES[kind], shape_txt, _S["pin_index"], _S["list_len"][key], pin["algo_sha"][:16], pin["kernel"],
                 _S["ext_sha"][:16], ok))
    if not ok:
        raise RuntimeError("E0 self-check FAILED for the LM-head %s (ak_l8b_lmh): output differs from torch's matmul"
                           % _NAMES[kind])
    _S["checked"].add(key)


def _lm_forward(x, w):
    """torch.matmul(x, w.t()) through the pinned algorithm, or None (stock)"""
    import torch
    c = _case(0, x, w)
    if c is None:
        return None
    T, x2d = c
    out = torch.empty(x.shape[:-1] + (V_LM,), dtype=x.dtype, device=x.device)
    if not _run(0, x2d, w, out.view(T, V_LM)):
        return None
    key = (0, T, w.device.index)
    if key not in _S["checked"]:
        _first_check(0, key, out, lambda: torch.matmul(x, w.t()),
                     "x %s x W %s^T -> %s" % (list(x.shape), list(w.shape), list(out.shape)))
    return out


def _lm_dgrad(g, w):
    """g.matmul(w) through the pinned algorithm (the dgrad line of the stock backward)"""
    import torch
    c = _case(1, g, w)
    if c is not None:
        T, g2d = c
        out = torch.empty(g.shape[:-1] + (H_LM,), dtype=g.dtype, device=g.device)
        if _run(1, g2d, w, out.view(T, H_LM)):
            key = (1, T, w.device.index)
            if key not in _S["checked"]:
                _first_check(1, key, out, lambda: g.matmul(w),
                             "dy %s x W %s -> %s" % (list(g.shape), list(w.shape), list(out.shape)))
            return out
    return g.matmul(w)


def _patch_layers(L):
    orig_fwd = L._linear_forward

    def _linear_forward(input, weight, bias, output_dtype):
        if bias is None and (output_dtype is None or output_dtype == input.dtype):
            out = _lm_forward(input, weight)
            if out is not None:
                return out
        return orig_fwd(input, weight, bias, output_dtype)

    _linear_forward.__wrapped__ = orig_fwd
    L._linear_forward = _linear_forward
    if not _DGRAD_ON:
        return
    _install_dgrad(L)                          # L4 (C5): a failure raises; AK_LMH_DGRAD=0 keeps the dgrad stock


def _install_dgrad(L):
    cls = L.LinearWithGradAccumulationAndAsyncCommunication
    raw = inspect.unwrap(cls.backward)
    src = inspect.getsource(raw)
    sha = hashlib.sha256(src.encode()).hexdigest()
    want = json.load(open(os.path.join(_DIR, "lmh_pin.json"))).get("bwd_src_sha256")
    if sha != want or src.count(DGRAD_LINE) != 1:
        raise RuntimeError("[auto_kernel lmh] LM-head dgrad NOT ARMED: the stock backward's source is not the "
                           "recorded one (sha256 %s, lmh_pin.json %s, dgrad lines %d)"
                           % (sha[:16], str(want)[:16], src.count(DGRAD_LINE)))
    lines = src.splitlines()
    body = textwrap.dedent("\n".join(l for l in lines if not l.strip().startswith("@")))
    body = body.replace(DGRAD_LINE, "grad_input = _ak_lmh_dgrad(grad_output, weight)")
    L.__dict__["_ak_lmh_dgrad"] = _lm_dgrad
    code = compile(body, "<ak_l8b_lmh: %s.backward, dgrad line changed>" % cls.__qualname__, "exec")
    ns = {}
    exec(code, L.__dict__, ns)
    new = L.custom_bwd(ns["backward"])
    cls.backward = staticmethod(new)
    _S["dgrad_installed"] = True


def _install(name, patch):
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                patch(module)
            spec.loader.exec_module = exec_module
            return spec

    if name in sys.modules:
        patch(sys.modules[name])
    else:
        sys.meta_path.insert(0, _Finder())


if _ARMED:
    _install("megatron.core.tensor_parallel.layers", _patch_layers)
