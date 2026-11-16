"""auto_kernel OPSWZ (Llama 3.1 8B campaign, 2026-10-02): the op-fuser fc1 / fc2 weights quantized with GEMM-swizzled
scales, as TE's own modules already quantize theirs.

Stock: TE's Linear / LayerNormLinear / LayerNormMLP set optimize_for_gemm on an NVFP4 2D weight quantizer whose weight
dims are multiples of 128 (module/base.py:1277-1308 `_enable_weight_preswizzle`; linear.py:2429-2432), so the 2D
quantize kernel writes the scales in the GEMM layout.  The op fuser's BasicLinear (the recipe's fc1 / fc2 under
USE_TE_OPS) never sets it, so its weights are quantized with compact scales and every GEMM that uses them swizzles them
first: PROF-L0, `swizzle_row_scaling_kernel` 128 x 6.9 us per step (fc1 and fc2, forward row-wise and backward
column-wise).  With AK_OPSWZ=1 BasicLinear.reset_recipe_state sets the flag under the same rule, so the quantize writes
the swizzled layout and the GEMMs skip the swizzle.  The data bytes are unchanged and the scale bytes are the same
permutation of the same values (swizzle::gemm_swizzled_scale_idx), so every GEMM gets identical operands.

First eager call check, once per weight shape: the first quantize by a flagged quantizer is compared with the stock
quantize (flag off) of the same weight: data, amaxes, and the swizzled scales v the compact ones permuted in torch.  Rank 0 prints the proof line;
a mismatch raises.  Loaded by `import firmus_ak` (pretrain.py's first import); armed by AK_OPSWZ=1 alone.
"""
import os
import sys

_ARMED = os.environ.get("AK_OPSWZ", "0") == "1"
_STATE = {"set": 0, "checked": set()}

if _ARMED:
    import importlib.abc
    import importlib.util

    def _eligible(q, w):
        if q is None or type(q).__name__ != "NVFP4Quantizer" or w is None:
            return False
        g = lambda k, d=None: getattr(q, k, d)  # noqa: E731
        if not g("with_2d_quantization", False) or g("with_rht", False) or g("row_scaled_nvfp4", False) \
                or g("nvfp4_use_4over6", False):
            return False
        try:
            rows, cols = w.numel() // w.shape[-1], w.shape[-1]
        except Exception:  # noqa: BLE001
            return False
        return rows % 128 == 0 and cols % 128 == 0 and type(w).__name__ in ("Parameter", "Tensor")

    def _patch_basic_linear(mod):
        cls = mod.BasicLinear
        orig = cls.reset_recipe_state

        def reset_recipe_state(self, *a, **k):
            r = orig(self, *a, **k)
            try:
                q = self.get_quantizer("forward", 1)
            except Exception:  # noqa: BLE001
                q = None
            if _eligible(q, getattr(self, "weight", None)) and not getattr(q, "optimize_for_gemm", False):
                q.optimize_for_gemm = True
                q._ak_opswz = True
                _STATE["set"] += 1
            return r
        reset_recipe_state.__wrapped__ = orig
        cls.reset_recipe_state = reset_recipe_state

    def _swizzle(compact, cols16):
        """swizzle::gemm_swizzled_scale_idx applied in torch: compact [R, C] -> the swizzled buffer (flat)"""
        import torch
        R, C = compact.shape
        r = torch.arange(R, device=compact.device)[:, None]
        c = torch.arange(C, device=compact.device)[None, :]
        cbg_cnt = (cols16 + 3) // 4
        idx = (((r // 128) * cbg_cnt + c // 4) * 32 + r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
        out = torch.zeros(R * C, dtype=torch.uint8, device=compact.device)
        out[idx.reshape(-1)] = compact.contiguous().view(torch.uint8).reshape(-1)
        return out

    def _check(q, w, new):
        import torch
        import torch.distributed as dist
        import transformer_engine_torch as tex
        qr = q.copy()
        qr.optimize_for_gemm = False
        ref = tex.quantize(w, qr)
        u8 = lambda t: None if t is None else t.contiguous().view(torch.uint8).reshape(-1)  # noqa: E731
        rows, cols = w.numel() // w.shape[-1], w.shape[-1]
        res = {}
        for b in ("_rowwise_data", "_columnwise_data", "_amax_rowwise", "_amax_columnwise"):
            a, n = u8(getattr(ref, b, None)), u8(getattr(new, b, None))
            res[b.lstrip("_")] = (a is None and n is None) or (a is not None and n is not None and torch.equal(a, n))
        rs, cs = getattr(ref, "_rowwise_scale_inv", None), getattr(ref, "_columnwise_scale_inv", None)
        if rs is not None:
            res["row_scales_swizzled"] = torch.equal(_swizzle(rs, cols // 16), u8(new._rowwise_scale_inv))
        if cs is not None:
            res["col_scales_swizzled"] = torch.equal(_swizzle(cs, rows // 16), u8(new._columnwise_scale_inv))
        res["flags"] = (getattr(ref, "_with_gemm_swizzled_scales", None) is False
                        and getattr(new, "_with_gemm_swizzled_scales", None) is True)
        ok = all(res.values())
        bad = torch.tensor([0 if ok else 1], device=w.device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(bad)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("[auto_kernel opswz] op-fuser weight quantizers write GEMM-swizzled scales ARMED: optimize_for_gemm set "
                  "on {} quantizer(s); first weight {} (row {} col {}) v the stock compact-scale quantize: {}; ranks "
                  "failing {}: {}".format(_STATE["set"], list(w.shape), q.rowwise_usage, q.columnwise_usage, res,
                                          int(bad[0]), ok and int(bad[0]) == 0), flush=True)
        if not ok or int(bad[0]) != 0:
            raise RuntimeError("[auto_kernel opswz] E0 self-check FAILED: %s" % res)

    def _patch_nvfp4(mod):
        cls = mod.NVFP4Quantizer
        orig = cls.quantize_impl

        def quantize_impl(self, tensor):
            out = orig(self, tensor)
            if getattr(self, "_ak_opswz", False) and tuple(tensor.shape) not in _STATE["checked"]:
                import torch
                if not torch.cuda.is_current_stream_capturing():
                    _STATE["checked"].add(tuple(tensor.shape))
                    _check(self, tensor, out)
            return out
        quantize_impl.__wrapped__ = orig
        cls.quantize_impl = quantize_impl

    _HOOKS = {"transformer_engine.pytorch.ops.basic.basic_linear": _patch_basic_linear,
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
