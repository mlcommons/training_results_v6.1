# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA C++ single-sweep fused RMSNorm backward with the dgamma/dbias column reduction finished in-kernel.
nvcc builds rmsnorm_bwd_cuda.cu + rmsnorm_bwd_cuda_binding.cpp into <build dir>/rmsnorm_bwd_cuda.so
(one build per source hash; ranks sharing the directory serialize on a lock file and reuse the .so via its stamp) and exposes
one torch.library op, torch.ops.gpt_oss.rmsnorm_bwd_fused_cuda: dy, x, rstd, gamma, dres and the MXFP8 outputs as in the
Triton op of rmsnorm_bwd_fused.py, but no partials; instead dgamma_out [N] bf16 (a parameter's main_grad or a fresh
tensor) and, when out_mxfp8, dbias_out [N] bf16 (column sums of the emitted bf16 gradient = the proj bias grad) are written
in place (+= when accumulate is set, as DDP accumulates main_grad).  Versus the Triton pair (rmsnorm_bwd_fused 7.645 ms +
colsum 1.241 ms per step) it produces the same outputs, but dy and x are read from DRAM once (the slab stays in shared
memory across the two sweeps), the fp32 partials never leave L2 (per-CTA last-arrives tickets in a 3-level tree: 16 row
blocks -> group -> 8 groups -> final, each level a fixed-order sum by the last-arriving CTA alone, hence deterministic) and
dgamma/dbias land in bf16 in the caller's targets: no partials round trip, no colsum launch.  The tickets live in a
per-device int32 ring allocated at load time and re-armed by the kernel itself, so the op is capturable in a full-iteration
CUDA graph.  Knobs: NORM_BWD_CUDA 1 (default) / 0 = stay on the Triton pair; NORM_BWD_CUDA_ROWS 8 (default: 4-CTA cluster per
32-row block, 2 CTAs/SM at N=2880) or 4 (8-CTA cluster); NORM_BWD_CUDA_BUILD_DIR / _ARCH / _STD / _BUILD_TIMEOUT (build).
"""
import hashlib
import importlib.machinery
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from typing import List, Optional, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAME = "rmsnorm_bwd_cuda"
_SOURCES = ("rmsnorm_bwd_cuda.cu", "rmsnorm_bwd_cuda_binding.cpp", "rmsnorm_bwd_cuda.h")

ENABLED = os.getenv("NORM_BWD_CUDA", "1") == "1"
ROWS = int(os.getenv("NORM_BWD_CUDA_ROWS", "8"))
BUILD_TIMEOUT_S = int(os.getenv("NORM_BWD_CUDA_BUILD_TIMEOUT", "900"))
DBG = int(os.getenv("NORM_BWD_CUDA_DBG", "0"))   # development only: phase-skipping mask for timing breakdowns
MX_BLOCK = 32
GROUP_MIN = 16          # row blocks (of 32 rows) per level-1 reduction group
FAN2 = 8                # level-1 groups per level-2 group
TICKET_SLOT = 1 + 64 + 8  # int32 per slot: [final ticket, 64 level-1 tickets, 8 level-2 tickets] (== pbn::kTicketSlot)
TICKET_RING = 64        # slots per device; consecutive calls rotate (same-stream calls never overlap anyway)
TORCH_FP8 = {7: torch.float8_e4m3fn, 8: torch.float8_e5m2}

_MOD = None
_STATE = {"tried": False, "ok": False, "why": "", "build_s": 0.0, "mode": "", "calls": 0}
_TICKETS = {}           # device index -> int32 [TICKET_RING, TICKET_SLOT] zeros (re-armed by the kernel)
_RING = {"next": 0}


def _say(msg: str) -> None:
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"NORM_BWD_CUDA rank={rank} {msg}\n")
    sys.__stdout__.flush()


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


# ------------------------------------------------------------------------------------------------ build
def _nvcc() -> Optional[str]:
    cands = []
    for root in (os.getenv("CUDA_HOME"), os.getenv("CUDA_PATH")):
        if root:
            cands.append(os.path.join(root, "bin", "nvcc"))
    try:
        from torch.utils.cpp_extension import CUDA_HOME as _TCH
        if _TCH:
            cands.append(os.path.join(_TCH, "bin", "nvcc"))
    except Exception:  # noqa
        pass
    cands += ["/usr/local/cuda/bin/nvcc", shutil.which("nvcc") or ""]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _arch_candidates(nvcc: str) -> List[List[str]]:
    forced = os.getenv("NORM_BWD_CUDA_ARCH", "").strip()
    out: List[List[str]] = []
    if forced:
        num = re.sub(r"\D", "", forced)
        out.append([f"-gencode=arch=compute_{num},code=sm_{num}"])
    major, minor = torch.cuda.get_device_capability()
    num = f"{major}{minor}"
    try:
        listed = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True, timeout=60).stdout.split()
    except Exception:  # noqa
        listed = []
    known = {re.sub(r"\D", "", a) for a in listed}
    if not known or num in known:
        out.append([f"-gencode=arch=compute_{num},code=sm_{num}"])
    out.append(["-arch=native"])
    if known:
        same_major = sorted((int(k) for k in known if k[: len(str(major))] == str(major) and int(k) <= int(num)), reverse=True)
        for k in same_major[:2]:
            out.append([f"-gencode=arch=compute_{k},code=compute_{k}"])
    seen = set()
    uniq = []
    for flags in out:
        key = tuple(flags)
        if key not in seen:
            seen.add(key)
            uniq.append(flags)
    return uniq


def _source_hash() -> str:
    h = hashlib.sha256()
    for s in _SOURCES:
        with open(os.path.join(_HERE, s), "rb") as f:
            h.update(f.read())
    h.update(torch.__version__.encode())
    h.update(str(torch.version.cuda).encode())
    return h.hexdigest()[:16]


def build_dir() -> str:
    d = os.getenv("NORM_BWD_CUDA_BUILD_DIR", "").strip()
    if d:
        return d
    major, minor = torch.cuda.get_device_capability()
    tag = re.sub(r"[^0-9A-Za-z.+_-]", "_", torch.__version__)
    return os.path.join(_HERE, "_build", f"{tag}-sm{major}{minor}")


def _so_path(d: str) -> str:
    return os.path.join(d, f"{_NAME}.so")


def _stamp_path(d: str) -> str:
    return os.path.join(d, f"{_NAME}.stamp")


def _stamp_ok(d: str, want: str) -> bool:
    try:
        with open(_stamp_path(d)) as f:
            return f.read().strip().split()[0] == want and os.path.isfile(_so_path(d))
    except Exception:  # noqa
        return False


def _torch_flags() -> Tuple[List[str], List[str], List[str]]:
    from torch.utils import cpp_extension as ce
    try:
        incs = ce.include_paths(device_type="cuda")
    except TypeError:
        incs = ce.include_paths(cuda=True)
    incs = list(incs) + [sysconfig.get_paths()["include"]]
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    links = [f"-L{lib_dir}", "-lc10", "-ltorch", "-ltorch_cpu", "-lc10_cuda", "-ltorch_cuda", "-ltorch_python", "-lcudart"]
    defs = [f"-DTORCH_EXTENSION_NAME={_NAME}", "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    try:
        defs += list(ce._get_pybind11_abi_build_flags())
    except Exception:  # noqa
        for name in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
            v = getattr(torch._C, f"_PYBIND11_{name}", None)
            if v is not None:
                defs.append(f'-DPYBIND11_{name}=\\"{v}\\"')
    return incs, links, defs


def _cxx_std() -> str:
    forced = os.getenv("NORM_BWD_CUDA_STD", "").strip()
    if forced:
        return forced
    try:
        import inspect
        from torch.utils import cpp_extension as ce
        stds = [int(x) for x in re.findall(r"[-/]std[=:]c\+\+(\d\d)", inspect.getsource(ce))]
        if stds:
            return f"c++{max(stds)}"
    except Exception:  # noqa
        pass
    return "c++20"


def _nvcc_common(nvcc: str) -> Tuple[List[str], List[str]]:
    incs, links, defs = _torch_flags()
    common = [nvcc, "-O3", f"-std={_cxx_std()}", "-Xcompiler", "-fPIC", "-Xcompiler", "-O3", "--expt-relaxed-constexpr",
              "-lineinfo", f"-I{_HERE}"] + [f"-I{i}" for i in incs] + defs
    return common, links


def _run(cmd: List[str], label: str, log, t0: float) -> bool:
    log.write("CMD " + " ".join(cmd) + "\n")
    log.flush()
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = pr.communicate(timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pr.kill()
        out, err = pr.communicate()
        err += "\n(timeout)"
    log.write(f"RC {pr.returncode} in {time.time() - t0:.1f}s for {label}\n{out}\n{err}\n")
    log.flush()
    return pr.returncode == 0


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _build(d: str, want: str) -> str:
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found (CUDA_HOME, torch CUDA_HOME, /usr/local/cuda, PATH)")
    os.makedirs(d, exist_ok=True)
    common, links = _nvcc_common(nvcc)
    bobj = os.path.join(d, "binding.o")
    kobj = os.path.join(d, "kernel.o")
    with open(os.path.join(d, "build.log"), "a") as log:
        log.write(f"==== build {time.strftime('%Y-%m-%d %H:%M:%S')} hash={want} torch={torch.__version__} cuda={torch.version.cuda} "
                  f"nvcc={nvcc} std={_cxx_std()}\n")
        t0 = time.time()
        if not _run(common + ["-c", os.path.join(_HERE, _SOURCES[1]), "-o", bobj], "binding", log, t0):
            raise RuntimeError(f"nvcc could not compile the binding; see {os.path.join(d, 'build.log')}")
        errors = []
        try:
            for flags in _arch_candidates(nvcc):
                t1 = time.time()
                ok = _run(common + flags + ["-Xptxas", "-v", "-c", os.path.join(_HERE, _SOURCES[0]), "-o", kobj],
                          f"kernel {' '.join(flags)}", log, t1)
                if ok:
                    ok = _run([nvcc, "--shared", "-Xcompiler", "-fPIC"] + flags + [kobj, bobj, "-o", _so_path(d) + ".tmp"] + links,
                              f"link {' '.join(flags)}", log, t1)
                if ok:
                    os.replace(_so_path(d) + ".tmp", _so_path(d))
                    with open(_stamp_path(d), "w") as f:
                        f.write(f"{want} {' '.join(flags)} {time.time():.0f}\n")
                    return " ".join(flags)
                errors.append(" ".join(flags))
                _rm(_so_path(d) + ".tmp")
        finally:
            _rm(bobj)
            _rm(kobj)
    raise RuntimeError(f"nvcc build failed for every target ({errors}); see {os.path.join(d, 'build.log')}")


def _import_so(d: str):
    path = _so_path(d)
    loader = importlib.machinery.ExtensionFileLoader(_NAME, path)
    spec = importlib.util.spec_from_file_location(_NAME, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _acquire_build(d: str, want: str) -> str:
    """Build once per source hash across the ranks that share d; everyone else waits for the stamp."""
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, ".build.lock")
    t0 = time.time()
    while True:
        if _stamp_ok(d, want):
            return "prebuilt"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {os.uname().nodename} {time.time():.0f}\n".encode())
            os.close(fd)
        except FileExistsError:
            try:
                age = time.time() - os.stat(lock).st_mtime
            except OSError:
                age = 0.0
            if age > BUILD_TIMEOUT_S:
                _rm(lock)
                continue
            if time.time() - t0 > BUILD_TIMEOUT_S:
                raise RuntimeError(f"timed out waiting for another process's build in {d}")
            time.sleep(2.0)
            continue
        try:
            if _stamp_ok(d, want):
                return "prebuilt"
            tb = time.time()
            flags = _build(d, want)
            _STATE["build_s"] = time.time() - tb
            return f"built ({flags}, {_STATE['build_s']:.0f}s)"
        finally:
            _rm(lock)


def _tickets(dev_index: int) -> torch.Tensor:
    t = _TICKETS.get(dev_index)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ticket ring must be allocated eagerly (first use inside a CUDA-graph capture)")
        t = torch.zeros(TICKET_RING, TICKET_SLOT, dtype=torch.int32, device=torch.device("cuda", dev_index))
        torch.cuda.synchronize(dev_index)
        _TICKETS[dev_index] = t
    return t


def ensure_loaded() -> bool:
    """Load (building if needed) the extension; False (reason logged once) leaves the Triton pair in charge."""
    global _MOD
    if _STATE["tried"]:
        return _STATE["ok"]
    _STATE["tried"] = True
    if not ENABLED:
        _STATE["why"] = "disabled by NORM_BWD_CUDA=0 (control path: Triton rmsnorm_bwd_fused + colsum)"
        _say(_STATE["why"])
        return False
    if not torch.cuda.is_available():
        _STATE["why"] = "CUDA not available"
        _say(_STATE["why"])
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("first use inside a CUDA graph capture; build/load must happen eagerly")
        if ROWS not in (4, 8):
            raise RuntimeError(f"NORM_BWD_CUDA_ROWS must be 4 or 8 (got {ROWS})")
        d = build_dir()
        want = _source_hash()
        mode = _acquire_build(d, want)
        _MOD = _import_so(d)
        info = _MOD.device_info()
        if int(info["cc_major"]) < 9:
            raise RuntimeError(f"needs sm_90+ (thread-block clusters, cp.async.bulk); device is {info['cc_major']}.{info['cc_minor']}")
        need = max(int(_MOD.smem_bytes(ROWS, 2880, False)), int(_MOD.smem_bytes(ROWS, 2880, True)))
        if need > int(info["smem_optin"]):
            raise RuntimeError(f"shared memory {need} B exceeds the device opt-in {info['smem_optin']} B")
        _tickets(torch.cuda.current_device())
        _STATE["ok"] = True
        _STATE["mode"] = mode
        attrs = {}
        for key, (hr, mx) in {"plain": (False, False), "res": (True, False), "full": (True, True)}.items():
            try:
                attrs[key] = dict(_MOD.kernel_attrs(ROWS, hr, mx, False, 2880))
            except Exception as ex:  # noqa
                attrs[key] = f"{type(ex).__name__}: {ex}"
        _STATE["attrs"] = attrs
        _say(f"loaded {mode}: {d}; device sms={info['sms']} cc={info['cc_major']}.{info['cc_minor']} smem_optin={info['smem_optin']}; "
             f"rows={ROWS} cluster={32 // ROWS} smem={need}; attrs {attrs}")
        return True
    except Exception as ex:  # noqa
        _STATE["why"] = f"not loaded: {type(ex).__name__}: {str(ex)[:600]}"
        _say(_STATE["why"] + " -- the Triton pair stays in charge")
        _STATE["ok"] = False
        return False


def state() -> dict:
    return dict(_STATE, enabled=ENABLED, rows=ROWS)


def kernel_attrs(rows: int, has_res: bool, mxfp8: bool, e5m2: bool = False, N: int = 2880) -> dict:
    assert _MOD is not None
    return dict(_MOD.kernel_attrs(int(rows), bool(has_res), bool(mxfp8), bool(e5m2), int(N)))


# ------------------------------------------------------------------------------------------------ the op
def _plan(M: int) -> Tuple[int, int]:
    nb32 = M // MX_BLOCK
    group = max(GROUP_MIN, -(-nb32 // 64))
    ngroups = -(-nb32 // group)
    return group, ngroups


@torch.library.custom_op("gpt_oss::rmsnorm_bwd_fused_cuda", mutates_args=("dgamma_out", "dbias_out"))
def rmsnorm_bwd_fused_cuda(
    dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor, dres: Optional[torch.Tensor],
    dgamma_out: torch.Tensor, dbias_out: Optional[torch.Tensor],
    zero_centered_gamma: bool, out_mxfp8: bool, fp8_code: int, accumulate: bool, rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _MOD is not None, "rmsnorm_bwd_cuda: extension not loaded (call ensure_loaded() eagerly first)"
    M, N = dy.shape
    dev = dy.device
    rows = ROWS if rows <= 0 else int(rows)
    group, ngroups = _plan(M)
    pstride = 2 * N if out_mxfp8 else N
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    part = torch.empty(M // MX_BLOCK, pstride, dtype=torch.float32, device=dev)
    part2 = torch.empty(ngroups, pstride, dtype=torch.float32, device=dev)
    part3 = torch.empty(-(-ngroups // FAN2), pstride, dtype=torch.float32, device=dev)
    ring = _tickets(dev.index if dev.index is not None else torch.cuda.current_device())
    slot = _RING["next"]
    _RING["next"] = (slot + 1) % TICKET_RING
    tickets = ring[slot]
    if out_mxfp8:
        assert dbias_out is not None, "the MXFP8 variant computes dbias: pass dbias_out"
        nblk_pad = _round_up(N // MX_BLOCK, 4)
        npad_c = _round_up(N, 128)
        rd = torch.empty(M, N, dtype=torch.uint8, device=dev)
        cd = torch.empty(M, N, dtype=torch.uint8, device=dev)
        rs_alloc = torch.zeros if _round_up(M, 128) != M else torch.empty
        cs_alloc = torch.zeros if _round_up(M // MX_BLOCK, 4) != M // MX_BLOCK else torch.empty
        rs = rs_alloc(_round_up(M, 128), nblk_pad, dtype=torch.uint8, device=dev)
        cs = cs_alloc(_round_up(M // MX_BLOCK, 4), npad_c, dtype=torch.uint8, device=dev)
        ntx_r = nblk_pad // 4
        ntx_c = _round_up(M // MX_BLOCK, 4) // 4
    else:
        assert dbias_out is None, "dbias is only computed by the MXFP8 (norm2) variant"
        nblk_pad = npad_c = ntx_r = ntx_c = 1
        rd, rs, cd, cs = (torch.empty(0, dtype=torch.uint8, device=dev) for _ in range(4))
    _MOD.rmsnorm_bwd_fused(dy, x, rstd, gamma, dres, out, part, part2, part3, tickets, dgamma_out, dbias_out, rd, rs, cd, cs,
                           bool(zero_centered_gamma), int(fp8_code), bool(accumulate), rows, group, ngroups,
                           ntx_r, ntx_c, nblk_pad, npad_c, DBG)
    _STATE["calls"] += 1
    return out, rd, rs, cd, cs


@rmsnorm_bwd_fused_cuda.register_fake
def _(dy, x, rstd, gamma, dres, dgamma_out, dbias_out, zero_centered_gamma, out_mxfp8, fp8_code, accumulate, rows):
    M, N = dy.shape
    dev = dy.device
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    if out_mxfp8:
        return (out, torch.empty(M, N, dtype=torch.uint8, device=dev),
                torch.empty(_round_up(M, 128), _round_up(N // MX_BLOCK, 4), dtype=torch.uint8, device=dev),
                torch.empty(M, N, dtype=torch.uint8, device=dev),
                torch.empty(_round_up(M // MX_BLOCK, 4), _round_up(N, 128), dtype=torch.uint8, device=dev))
    return (out,) + tuple(torch.empty(0, dtype=torch.uint8, device=dev) for _ in range(4))


# ------------------------------------------------------------------------------------------------ hook-facing helpers
def eligible(M: int, N: int) -> bool:
    return _STATE["ok"] and M % 32 == 0 and N % 32 == 0 and N <= 3072 and N % (32 // ROWS * 4) == 0


def rmsnorm_bwd(dy2, x2, rstd, gamma, dres, zero_centered_gamma, want_mxfp8, fp8_code, dgamma_out, dbias_out, accumulate):
    """One launch: (out, rd, rs, cd, cs); dgamma_out (and dbias_out when want_mxfp8) are written in place."""
    return torch.ops.gpt_oss.rmsnorm_bwd_fused_cuda(
        dy2, x2, rstd, gamma, dres, dgamma_out, dbias_out if want_mxfp8 else None,
        bool(zero_centered_gamma), bool(want_mxfp8), int(fp8_code), bool(accumulate), ROWS)


# ------------------------------------------------------------------------------------------------ standalone self-test
def get_inputs():
    M, N = 4096, 2880
    torch.manual_seed(0)
    dy = torch.randn(M, N, device="cuda").to(torch.bfloat16)
    x = (torch.randn(M, N, device="cuda") * 2.0).to(torch.bfloat16)
    rstd = torch.rsqrt(x.float().pow(2).mean(dim=1) + 1e-5).contiguous()
    gamma = (1.0 + 0.1 * torch.randn(N, device="cuda")).to(torch.bfloat16)
    dres = torch.randn(M, N, device="cuda").to(torch.bfloat16)
    return [dy, x, rstd, gamma, dres]


def kernel_fn(dy, x, rstd, gamma, dres):
    assert ensure_loaded(), _STATE["why"]
    N = x.shape[1]
    dgamma = torch.empty(N, dtype=torch.bfloat16, device=dy.device)
    dbias = torch.empty(N, dtype=torch.bfloat16, device=dy.device)
    out, rd, rs, cd, cs = rmsnorm_bwd(dy, x, rstd, gamma, dres, False, True, 7, dgamma, dbias, False)
    return out, dgamma, dbias


def reference_fn(dy, x, rstd, gamma, dres):
    from rmsnorm_bwd_fused import ref_rmsnorm_bwd  # the Triton module's fp32 reference
    out, dgamma = ref_rmsnorm_bwd(dy, x, rstd, gamma, dres, False)
    return out, dgamma.to(torch.bfloat16), out.float().sum(dim=0).to(torch.bfloat16)


if __name__ == "__main__":
    sys.path.insert(0, _HERE)
    xs = get_inputs()
    outs = kernel_fn(*xs)
    refs = reference_fn(*xs)
    torch.cuda.synchronize()
    for n, a, r in zip(("dx+dres", "dgamma", "dbias"), outs, refs):
        print(n, "max abs diff", (a.float() - r.float()).abs().max().item(), "ref max", r.float().abs().max().item())
