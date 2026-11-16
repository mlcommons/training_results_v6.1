"""auto_kernel PROJCOL (Llama 3.1 8B campaign, 2026-10-02): the attention output projection keeps the NVFP4 column-wise
copy of its input from the forward, instead of re-quantizing the saved BF16 input in the backward.

Stock: megatron-core sets save_original_input on linear_proj for fp4 training (attention.py:432-448, to avoid holding the
quantized copy of the attention output next to the BF16 one attention saves).  TE's Linear then quantizes the attention
output rowwise only in the forward (linear.py:518-532) and again, column-wise with RHT, in the backward
(linear.py:1297-1311): a second HadamardAmax pass and RHT cast of the same [8192, 2, 4096] input per layer (PROF-L0:
28 + 41 us).  With AK_PROJCOL=1 the flag is cleared right after Attention.__init__, so the forward quantizes rowwise +
column-wise in one call and the backward uses the stored column-wise copy (+34 MB of NVFP4 data per layer held from the
forward to the backward).  The input quantizer rounds to nearest (no stochastic rounding, no RNG), so the column-wise
bytes are a function of the input alone, which the stock backward re-reads unchanged.

First eager call check (warm-up iteration 0): the first rowwise + column-wise quantize inside a cleared linear_proj's
forward is compared, every buffer as bytes, with a row-only quantize (the stock forward's call) and a column-only
quantize (the stock backward's call) of the same input.  Rank 0 prints the proof line; a mismatch raises.

Loaded by `import firmus_ak` (pretrain.py's first import); armed by AK_PROJCOL=1 alone.
"""
import os
import sys

_ARMED = os.environ.get("AK_PROJCOL", "0") == "1"
_STATE = {"cleared": 0, "checked": False, "in_proj": 0}

if _ARMED:
    import importlib.abc
    import importlib.util

    _BUFS = ("_rowwise_data", "_rowwise_scale_inv", "_amax_rowwise", "_columnwise_data", "_columnwise_scale_inv",
             "_amax_columnwise")

    def _same(a, b):
        import torch
        if a is None or b is None:
            return a is None and b is None
        return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and bool(
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def _check(q, x, out):
        import torch
        import torch.distributed as dist
        import transformer_engine_torch as tex
        qr = q.copy()
        qr.set_usage(rowwise=True, columnwise=False)
        qc = q.copy()
        qc.set_usage(rowwise=False, columnwise=True)
        ref_r = tex.quantize(x, qr)
        ref_c = tex.quantize(x, qc)
        res = {}
        for b in _BUFS:
            ref = ref_r if b.startswith("_row") or b == "_amax_rowwise" else ref_c
            res[b.lstrip("_")] = _same(getattr(out, b, None), getattr(ref, b, None)) and getattr(out, b, None) is not None
        res["swizzled_flag"] = (getattr(out, "_with_gemm_swizzled_scales", None)
                                == getattr(ref_r, "_with_gemm_swizzled_scales", None)
                                == getattr(ref_c, "_with_gemm_swizzled_scales", None))
        ok = all(res.values())
        bad = torch.tensor([0 if ok else 1], device=x.device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(bad)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("[auto_kernel projcol] linear_proj keeps its NVFP4 column-wise input from the forward ARMED: "
                  "save_original_input cleared on {} module(s); first forward quantize {} {} (row {} col {}, RHT {}, SR {}) "
                  "v the row-only (stock forward) and column-only (stock backward) quantizes of the same input: {}; "
                  "ranks failing {}: {}".format(_STATE["cleared"], list(x.shape), x.dtype, q.rowwise_usage,
                                                q.columnwise_usage, q.with_rht, q.stochastic_rounding, res, int(bad[0]),
                                                ok and int(bad[0]) == 0), flush=True)
        del ref_r, ref_c
        if not ok or int(bad[0]) != 0:
            raise RuntimeError("[auto_kernel projcol] E0 self-check FAILED: %s" % res)

    def _wrap_proj_forward(mod):
        orig = mod.forward

        def forward(*a, **k):
            if _STATE["checked"]:
                return orig(*a, **k)
            _STATE["in_proj"] += 1
            try:
                return orig(*a, **k)
            finally:
                _STATE["in_proj"] -= 1
        forward.__wrapped__ = orig
        mod.forward = forward

    def _patch_attention(mod):
        cls = mod.Attention
        orig_init = cls.__init__

        def __init__(self, *a, **k):
            orig_init(self, *a, **k)
            lp = getattr(self, "linear_proj", None)
            if lp is not None and getattr(lp, "save_original_input", False):
                lp.save_original_input = False
                _STATE["cleared"] += 1
                _wrap_proj_forward(lp)
        __init__.__wrapped__ = orig_init
        cls.__init__ = __init__

    def _patch_nvfp4(mod):
        cls = mod.NVFP4Quantizer
        orig = cls.quantize_impl

        def quantize_impl(self, tensor):
            out = orig(self, tensor)
            if _STATE["in_proj"] and not _STATE["checked"] and self.rowwise_usage and self.columnwise_usage:
                import torch
                if not torch.cuda.is_current_stream_capturing():
                    _STATE["checked"] = True
                    _check(self, tensor, out)
            return out
        quantize_impl.__wrapped__ = orig
        cls.quantize_impl = quantize_impl

    _HOOKS = {"megatron.core.transformer.attention": _patch_attention,
              "transformer_engine.pytorch.tensor.nvfp4_tensor": _patch_nvfp4}

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
