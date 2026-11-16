"""auto_kernel FP8CS version 2 (Llama 3.1 8B campaign, 2026-10-07): the FP8 current-scaling quantize of FP8 attention's
inputs by our kernels (ak_l8b_fp8q), TE's arithmetic byte for byte -- as version 1 (fp8cs, ak_l8b_fp8cs), with less
traffic and fewer launches.

Stock (TE 2.20, Float8CurrentScalingQuantizer; PROF-L2k-1x4 per layer): dO [8192, 2, 32, 128] in E5M2 costs a zero-amax
kernel 1.2 + amax_kernel 27.8 + compute_scale_from_amax 1.2 + cast_fp8_1D 31.3 us; the packed q|k|v buffer [100663296]
in E4M3 (version 1: the amax from the X12 RoPE forward's partial maxima) a scale kernel 2.5 + our cast 48.8 us.
Here, for a row-wise, unreduced quantizer on a contiguous bf16 tensor:
  * the q|k|v buffer the X12 RoPE forward just wrote (x12pack2 registers its pointer, size and partial maxima): the
    1-CTA scale kernel, then a persistent cast with AK_FP8CS_CAST = "u,rev,blocks_per_sm" (16-byte vectors per
    thread, the walk back to front when rev is 1: the RoPE kernel leaves the buffer's tail in L2);
  * anything else (dO), with AK_FP8CS_AMAX=1: ak_l8b_fp8q.amax_cast -- a partial-amax pass (one maximum per block, no
    atomics, no zeroing kernel) and a cast whose blocks each reduce those partials and compute the scale themselves
    (block 0 stores scale_inv): two launches instead of four; AK_FP8CS_AMAX_CFG = "u_amax,rev_amax,amax_bps,u_cast,
    rev_cast,cast_bps".
With the x12pack3 overlay's QKV direct (AK_QKV_DIRECT=1) the q|k|v path has a third form: under CUDA-graph capture
the RoPE forward wrote only its partial maxima, and the cast here is fused_qkv_rope_fp8._qkv_rope_fp8_kernel -- the
same bf16 values recomputed from the RoPE's registered inputs, times the scale, TE's satfinite cast -- writing the FP8
bytes without the bf16 buffer ever existing.  Its proof is eager: at the first eager quantize of a registered buffer
(written, as every eager forward writes it) the direct path runs into scratch buffers and is compared with tex.quantize
of the written buffer; only then is it allowed (the registry's direct_ok), and rank 0 prints one ARMED line.
Version 3 (2026-10-07): when the RoPE overlay (x12pack4, AK_QKV_DIRECT_IMPL=cuda) registers its QKV direct passes as
callables ("p1": the partial maxima, "p2": the FP8 cast from the RoPE's inputs; ak_l8b_qkvx), the direct path and its
eager check call those -- the check proves what runs under capture; it compares the FP8 bytes and scale_inv with TE's
quantize of the written buffer and the amax with the written path's (the partials themselves are reported).
The output is what tex.quantize allocates (tex.create_empty_quantized_tensor) with _data and _scale_inv filled.  The
first eager call of each (path, shape, format) is checked against TE's tex.quantize on the same tensor (FP8 data,
scale_inv, transpose, type) as bytes; rank 0 prints the proof line, a mismatch raises.  Loaded by `import firmus_ak`;
armed by AK_FP8CS=1 alone.  AK_FP8CS_EXT_DIR names the directory holding ak_l8b_fp8q*.so.
"""
import os
import sys

_ARMED = os.environ.get("AK_FP8CS", "0") == "1"
_STATE = {"checked": set(), "ext": None, "calls": {"partials": 0, "amax": 0}}
# the amax path (dO): ak_l8b_fp8q.amax_cast; the q|k|v cast's and the amax path's launch shapes (dev f1 / f2)
_AMAX_PATH = os.environ.get("AK_FP8CS_AMAX", "0") == "1"
_CAST_CFG = [int(v) for v in os.environ.get("AK_FP8CS_CAST", "4,1,4").split(",")]
_AMAX_CFG = [int(v) for v in os.environ.get("AK_FP8CS_AMAX_CFG", "4,1,4,4,0,8").split(",")]
assert len(_CAST_CFG) == 3 and len(_AMAX_CFG) == 6, "AK_FP8CS_CAST: u,rev,bps; AK_FP8CS_AMAX_CFG: 6 integers"
# the QKV direct cast's walk (1: back to front) and FP8 conversion (1: cvt.rn.satfinite inline; 0: Triton's own)
_DIRECT_REV = os.environ.get("AK_QKV_DIRECT_REV", "1") == "1"
_DIRECT_ASM = os.environ.get("AK_QKV_DIRECT_ASM", "0") == "1"
# the partial-maxima pass as x12pack3 launches it under capture (kernel version, warps), and the cast's warps
_DIRECT_P1 = os.environ.get("AK_QKV_DIRECT_P1", "1")
_DIRECT_W1 = int(os.environ.get("AK_QKV_DIRECT_W1", "2"))
_DIRECT_W2 = int(os.environ.get("AK_QKV_DIRECT_W2", "2"))


def _ext():
    if _STATE["ext"] is None:
        d = os.environ.get("AK_FP8CS_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_fp8q
        _STATE["ext"] = ak_l8b_fp8q
    return _STATE["ext"]


def _rank():
    try:
        import torch
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def _qkva_reg():
    rope = sys.modules.get("transformer_engine.pytorch.attention.rope")
    return getattr(rope, "_AK_QKVA_REG", None) if rope is not None else None


def _qkva_partials(tensor):
    """(partials, deferred, direct) registered by the X12 RoPE forward when `tensor` is exactly its packed q|k|v
    buffer, else None: the partial maxima; the RoPE's inputs when it left the buffer unwritten (QKV direct); the RoPE's
    inputs for the direct path's eager check"""
    reg = _qkva_reg()
    if not reg or reg.get("part") is None:
        return None
    if tensor.data_ptr() != reg["ptr"] or tensor.numel() != reg["numel"]:
        return None
    part = reg["part"]
    reg["part"] = None          # single use: the next forward registers its own buffer
    return part, reg.get("deferred"), reg.get("direct")


def _direct_cast(args, ws, data):
    """the registered RoPE inputs straight to the FP8 bytes (scale ws[0]): the RoPE overlay's own pass 2 when it
    registered one (args["p2"]), else fused_qkv_rope_fp8._qkv_rope_fp8_kernel"""
    if "p2" in args:
        args["p2"](ws, data)
        return
    f8 = sys.modules.get("fused_qkv_rope_fp8")
    a = args
    f8._qkv_rope_fp8_kernel[(a["n_prog"],)](a["qkv"], a["tab"], ws, data, a["b"], a["n_prog"], a["nqe"], a["nke"],
                                            D=a["D"], NQ=a["NQ"], H=a["H"], TOTAL=a["TOTAL"], SAME=a["SAME"],
                                            REV=_DIRECT_REV, ASM=_DIRECT_ASM, num_warps=_DIRECT_W2)


def _direct_desc(args, path_k):
    import hashlib
    if "p1" in args:
        m = sys.modules.get("ak_l8b_qkvx")
        path = getattr(m, "__file__", "?")
        return "module {} sha256={} ({})".format(os.path.basename(path),
                                                 hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], args["impl"])
    return "module {} sha256={} (rev {} asm {} p1 v{} w{} w{})".format(
        os.path.basename(path_k), hashlib.sha256(open(path_k, "rb").read()).hexdigest()[:16], int(_DIRECT_REV),
        int(_DIRECT_ASM), _DIRECT_P1, _DIRECT_W1, _DIRECT_W2)


def _direct_partials(args, part):
    if "p1" in args:
        args["p1"](part)
        return
    f8 = sys.modules.get("fused_qkv_rope_fp8")
    a = args
    k1 = f8._qkv_rope_amax_only_kernel2 if _DIRECT_P1 == "2" else f8._qkv_rope_amax_only_kernel
    k1[(a["n_prog"],)](a["qkv"], a["tab"], part, a["b"], D=a["D"], NQ=a["NQ"], H=a["H"], TOTAL=a["TOTAL"],
                       SAME=a["SAME"], num_warps=_DIRECT_W1)


if _ARMED:
    import importlib.abc
    import importlib.util

    def _patch(mod):
        import torch
        import transformer_engine_torch as tex
        cls = mod.Float8CurrentScalingQuantizer
        orig = cls.quantize_impl
        e4m3, e5m2 = tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2

        def _eligible(q, t):
            return (type(t) is torch.Tensor and t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous()
                    and t.numel() > 0 and t.numel() % 8 == 0 and t.data_ptr() % 16 == 0
                    and bool(getattr(q, "rowwise_usage", False)) and not getattr(q, "columnwise_usage", False)
                    and not getattr(q, "with_amax_reduction", False) and q.dtype in (e4m3, e5m2))

        def _ours(q, t, part, deferred=None):
            # what tex.quantize(t, q) allocates: the quantizer's C++ create_tensor (cast.cpp)
            out = tex.create_empty_quantized_tensor(q, list(t.shape), t.dtype, t.device, False)
            fp8_max = 448.0 if q.dtype == e4m3 else 57344.0
            args = (out._data.view(-1), out._scale_inv.view(-1), fp8_max, q.dtype == e5m2,
                    bool(getattr(q, "force_pow_2_scales", False)), float(getattr(q, "amax_epsilon", 0.0)))
            if part is not None:
                ws = torch.empty(3, dtype=torch.float32, device=t.device)
                _ext().scale_from_partials(part, ws, args[1], args[2], args[4], args[5])
                if deferred is not None:            # the buffer was never written: from the RoPE's inputs
                    _direct_cast(deferred, ws, args[0])
                else:
                    u, rev, bps = _CAST_CFG
                    _ext().cast_from_ws(t.view(-1), ws, args[0], args[3], u, bool(rev), bps)
            else:
                ua, ra, ab, uc, rc, cb = _AMAX_CFG
                _ext().amax_cast(t.view(-1), *args, ua, bool(ra), ab, uc, bool(rc), cb)
            return out

        def same(a, b):
            if a is None or b is None:
                return a is None and b is None
            return (tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype
                    and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))))

        def _check_direct(q, t, direct, part, ref):
            """the QKV direct path into scratch buffers against tex.quantize of the written buffer (eager)"""
            _STATE["direct_checked"] = True
            part2 = torch.empty(direct["n_prog"], dtype=torch.float32, device=t.device)
            _direct_partials(direct, part2)
            out2 = tex.create_empty_quantized_tensor(q, list(t.shape), t.dtype, t.device, False)
            ws = torch.empty(3, dtype=torch.float32, device=t.device)
            _ext().scale_from_partials(part2, ws, out2._scale_inv.view(-1), 448.0 if q.dtype == e4m3 else 57344.0,
                                       bool(getattr(q, "force_pow_2_scales", False)),
                                       float(getattr(q, "amax_epsilon", 0.0)))
            _direct_cast(direct, ws, out2._data.view(-1))
            amax_w = torch.nan_to_num(part, nan=0.0, posinf=float("inf")).max().view(1)
            ok = {"amax": same(ws[2:3], amax_w), "data": same(ref._data, out2._data),
                  "scale_inv": same(ref._scale_inv, out2._scale_inv), "e4m3": q.dtype == e4m3}
            good = all(ok.values())
            ok["partials"] = same(part2, part)          # reported, not required (an all-NaN position may differ)
            if _rank() == 0:
                import hashlib
                f8 = sys.modules.get("fused_qkv_rope_fp8")
                path_k = f8.__file__
                print("[auto_kernel fp8cs] q|k|v FP8 straight from the RoPE inputs ARMED ({} -> E4M3, the bf16 buffer "
                      "unwritten under graph capture): {}; byte-exact v tex.quantize of the written buffer {}: {}".format(
                          list(t.shape), _direct_desc(direct, path_k), ok, good), flush=True)
            if not good:
                raise RuntimeError("[auto_kernel fp8cs] E0 self-check FAILED (QKV direct): {}".format(ok))
            reg = _qkva_reg()
            if reg is not None:
                reg["direct_ok"] = True

        def quantize_impl(self, tensor):
            if not _eligible(self, tensor):
                return orig(self, tensor)
            r = _qkva_partials(tensor)
            part, deferred, direct = r if r is not None else (None, None, None)
            if part is None and not _AMAX_PATH:
                return orig(self, tensor)
            path = "partials" if part is not None else "amax"
            _STATE["calls"][path] += 1
            capturing = torch.cuda.is_current_stream_capturing()
            if deferred is not None:
                if not capturing:
                    raise RuntimeError("[auto_kernel fp8cs] an unwritten q|k|v buffer outside CUDA-graph capture")
                _STATE["calls"]["direct"] = _STATE["calls"].get("direct", 0) + 1
                return _ours(self, tensor, part, deferred)
            key = (path, tuple(tensor.shape), int(self.dtype))
            first = key not in _STATE["checked"] and not capturing
            check_direct = direct is not None and not capturing and not _STATE.get("direct_checked")
            if not first and not check_direct:
                return _ours(self, tensor, part)
            ref = orig(self, tensor)
            new = _ours(self, tensor, part)
            if first:
                _STATE["checked"].add(key)
                ok = {f.lstrip("_"): same(getattr(ref, f, None), getattr(new, f, None))
                      for f in ("_data", "_scale_inv", "_transpose")}
                ok["type"] = type(ref) is type(new)
                good = all(ok.values())
                if _rank() == 0:
                    import hashlib
                    path_so = _ext().__file__
                    print("[auto_kernel fp8cs] FP8 current-scaling quantize ARMED ({} path, {} {} -> {}): module {} "
                          "sha256={} v2 ({}); byte-exact v tex.quantize {}: {}".format(
                              path, list(tensor.shape), tensor.dtype, "E4M3" if self.dtype == e4m3 else "E5M2",
                              os.path.basename(path_so), hashlib.sha256(open(path_so, "rb").read()).hexdigest()[:16],
                              "cast " + ",".join(map(str, _CAST_CFG)) if path == "partials"
                              else "amax+cast " + ",".join(map(str, _AMAX_CFG)), ok, good), flush=True)
                if not good:
                    raise RuntimeError("[auto_kernel fp8cs] E0 self-check FAILED ({}): {}".format(path, ok))
            if check_direct:
                _check_direct(self, tensor, direct, part, ref)
            return new
        quantize_impl.__wrapped__ = orig
        cls.quantize_impl = quantize_impl

    _NAME = "transformer_engine.pytorch.tensor.float8_tensor"

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _NAME:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch(module)
            spec.loader.exec_module = exec_module
            return spec

    if _NAME in sys.modules:
        _patch(sys.modules[_NAME])
    else:
        sys.meta_path.insert(0, _Finder())
