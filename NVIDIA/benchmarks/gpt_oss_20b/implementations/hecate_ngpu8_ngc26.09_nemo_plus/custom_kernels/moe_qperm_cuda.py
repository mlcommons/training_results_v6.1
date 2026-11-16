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

"""CUDA C++ backend of the MoE gather + MXFP8 quantize ops (moe_qperm_cuda.cu).

Builds the .cu and its binding with nvcc into <build dir>/moe_qperm_cuda.so (once per source hash, lock file
shared by all ranks) and exposes gather_quant / gather_quant_scaled with the return contract of moe_qperm.py's
Triton ops, which dispatch here once it is loaded.  Same bytes as the Triton kernels (rowwise scale layout 0, columnwise
layout 3, amax * (1/448)), bitwise tex.group_quantize on the rows in use; dprob is a fixed-order fp32 sum (no atomics).
Persistent CTAs take 128x128 units from a per-launch global counter (re-armed by the last CTA, so a CUDA-graph replay
needs no memset) instead of a static partition, which under concurrent NCCL / GEMM / optimizer kernels ran the displaced
CTAs' share as a second wave (bwd 1.47-1.66x, fwd 1.15-1.36x of isolated).  Knobs (read at import): MOE_QPERM_CUDA (0
keeps the Triton kernels); MOE_QPERM_CUDA_LOAD cpasync (default) / tma (tma: 192.6 / 437.0 us vs 141.9 / 304.6 fwd /
bwd); MOE_QPERM_CUDA_NSTAGE_FWD / MOE_QPERM_CUDA_NSTAGE_BWD pipeline depth 2-4, defaults 3 / 2 (142.0 / 250.6 us vs
Triton 180.8 / 337.3; depth 2 fits 5 bwd CTAs/SM vs 3); MOE_QPERM_CUDA_SCHED dynamic / static; MOE_QPERM_CUDA_GRAB_FWD /
MOE_QPERM_CUDA_GRAB_BWD units per grab 1 / 2, default 1 (fwd 139.9 vs 143.7 us, bwd 230.7 vs 232.6);
MOE_QPERM_CUDA_BUILD_DIR, MOE_QPERM_CUDA_ARCH, MOE_QPERM_CUDA_STD.
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
_NAME = "moe_qperm_cuda"
_SOURCES = ("moe_qperm_cuda.cu", "moe_qperm_cuda_binding.cpp", "moe_qperm_cuda.h")

ENABLED = os.getenv("MOE_QPERM_CUDA", "1") == "1"
LOAD_MODE = {"tma": 1, "cpasync": 0}.get(os.getenv("MOE_QPERM_CUDA_LOAD", "cpasync").strip().lower(), 0)
NSTAGE_FWD = int(os.getenv("MOE_QPERM_CUDA_NSTAGE_FWD", "3"))
NSTAGE_BWD = int(os.getenv("MOE_QPERM_CUDA_NSTAGE_BWD", "2"))
SCHED_DYNAMIC = os.getenv("MOE_QPERM_CUDA_SCHED", "dynamic").strip().lower() != "static"
GRAB_FWD = int(os.getenv("MOE_QPERM_CUDA_GRAB_FWD", "1"))
GRAB_BWD = int(os.getenv("MOE_QPERM_CUDA_GRAB_BWD", "1"))
BUILD_TIMEOUT_S = int(os.getenv("MOE_QPERM_CUDA_BUILD_TIMEOUT", "900"))
E4M3_MAX_RCP = 1.0 / 448.0

_MOD = None            # the loaded extension module
_STATE = {"tried": False, "ok": False, "why": "", "build_s": 0.0, "mode": "", "failed_calls": 0}
_SCHED_BUF = {}        # device index -> int32[64] scheduler counters (fwd at 0, bwd at 32), zero between launches
_EMPTY = {}            # device index -> int32[0] (the static-partition request)


def _say(msg: str) -> None:
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"MOE_QPERM_CUDA rank={rank} {msg}\n")
    sys.__stdout__.flush()


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
    """nvcc -gencode flag sets to try, best first: the device's own SM, then nvcc's view of the device, then PTX for the
    newest architecture nvcc lists (JIT-compiled by the driver)."""
    forced = os.getenv("MOE_QPERM_CUDA_ARCH", "").strip()
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
        if num in known:
            out.append([f"-gencode=arch=compute_{num},code=[sm_{num},compute_{num}]"])
    out.append(["-arch=native"])
    if known:
        same_major = sorted((int(k) for k in known if k[: len(str(major))] == str(major) and int(k) <= int(num)), reverse=True)
        for k in same_major[:2]:
            out.append([f"-gencode=arch=compute_{k},code=compute_{k}"])
        newest = max(int(k) for k in known)
        out.append([f"-gencode=arch=compute_{newest},code=compute_{newest}"])
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
    d = os.getenv("MOE_QPERM_CUDA_BUILD_DIR", "").strip()
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
    """include dirs, link flags, extra defines the way torch.utils.cpp_extension builds its extensions."""
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
    """The -std= value torch.utils.cpp_extension passes for this torch (the newest one its source names; the torch 2.15
    headers refuse anything below C++20), overridable by MOE_QPERM_CUDA_STD."""
    forced = os.getenv("MOE_QPERM_CUDA_STD", "").strip()
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


def _start(cmd: List[str], log) -> subprocess.Popen:
    log.write("CMD " + " ".join(cmd) + "\n")
    log.flush()
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _finish(pr: subprocess.Popen, label: str, log, t0: float) -> bool:
    try:
        out, err = pr.communicate(timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pr.kill()
        out, err = pr.communicate()
        err += "\n(timeout)"
    log.write(f"RC {pr.returncode} in {time.time() - t0:.1f}s for {label}\n{out}\n{err}\n")
    log.flush()
    return pr.returncode == 0


def _run(cmd: List[str], label: str, log, t0: float) -> bool:
    return _finish(_start(cmd, log), label, log, t0)


def _binding_cmd(nvcc: str, d: str) -> Tuple[List[str], str]:
    """Host-only compile of the pybind11 binding against the torch headers; arch-independent, so compiled once per build."""
    common, _ = _nvcc_common(nvcc)
    obj = os.path.join(d, f"{os.path.splitext(_SOURCES[1])[0]}.o")
    return common + ["-c", os.path.join(_HERE, _SOURCES[1]), "-o", obj], obj


def _kernel_cmd(nvcc: str, d: str, arch_flags: List[str]) -> Tuple[List[str], str]:
    """Device compile of the kernel file for one -gencode target (-Xptxas -v: registers/spills per kernel in build.log)."""
    common, _ = _nvcc_common(nvcc)
    obj = os.path.join(d, f"{os.path.splitext(_SOURCES[0])[0]}.o")
    return common + arch_flags + ["-Xptxas", "-v", "-c", os.path.join(_HERE, _SOURCES[0]), "-o", obj], obj


def _link(nvcc: str, d: str, arch_flags: List[str], objs: List[str], log, t0: float) -> bool:
    _, links = _nvcc_common(nvcc)
    ok = _run([nvcc, "--shared", "-Xcompiler", "-fPIC"] + arch_flags + objs + ["-o", _so_path(d) + ".tmp"] + links,
              f"link {' '.join(arch_flags)}", log, t0)
    if not ok:
        try:
            os.remove(_so_path(d) + ".tmp")
        except OSError:
            pass
        return False
    os.replace(_so_path(d) + ".tmp", _so_path(d))
    return True


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _build(d: str, want: str) -> str:
    """Compile into d (caller holds the lock); returns the arch flags used.  The binding (host C++ against the torch
    headers) is arch-independent: compiled once, concurrently with the first kernel compile, and its failure ends the
    build -- no -gencode candidate can fix it.  The kernel file is retried per -gencode candidate."""
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found (CUDA_HOME, torch CUDA_HOME, /usr/local/cuda, PATH)")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "build.log"), "a") as log:
        log.write(f"==== build {time.strftime('%Y-%m-%d %H:%M:%S')} hash={want} torch={torch.__version__} cuda={torch.version.cuda} "
                  f"nvcc={nvcc} std={_cxx_std()}\n")
        cands = _arch_candidates(nvcc)
        t0 = time.time()
        bcmd, bobj = _binding_cmd(nvcc, d)
        kcmd, kobj = _kernel_cmd(nvcc, d, cands[0])
        bproc = _start(bcmd, log)
        kproc = _start(kcmd, log)
        b_ok = _finish(bproc, f"binding {_SOURCES[1]}", log, t0)
        k_ok = _finish(kproc, f"kernel {_SOURCES[0]} {' '.join(cands[0])}", log, t0)
        errors = []
        try:
            if not b_ok:
                raise RuntimeError(f"nvcc could not compile the binding (host C++ against the torch headers, -std={_cxx_std()}); "
                                   f"see {os.path.join(d, 'build.log')}")
            for i, flags in enumerate(cands):
                try:
                    if i > 0:
                        kcmd, kobj = _kernel_cmd(nvcc, d, flags)
                        k_ok = _run(kcmd, f"kernel {_SOURCES[0]} {' '.join(flags)}", log, time.time())
                    if k_ok and _link(nvcc, d, flags, [kobj, bobj], log, time.time()):
                        with open(_stamp_path(d), "w") as f:
                            f.write(f"{want} {' '.join(flags)} {time.time():.0f}\n")
                        return " ".join(flags)
                    errors.append(" ".join(flags))
                except Exception as ex:  # noqa
                    errors.append(f"{' '.join(flags)}: {type(ex).__name__}: {ex}")
                finally:
                    _rm(kobj)
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
                try:
                    os.remove(lock)
                except OSError:
                    pass
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
            try:
                os.remove(lock)
            except OSError:
                pass


def ensure_loaded() -> bool:
    """Load (building if needed) the extension; False (with the reason logged once) leaves the Triton kernels in charge."""
    global _MOD
    if _STATE["tried"]:
        return _STATE["ok"]
    _STATE["tried"] = True
    if not ENABLED:
        _STATE["why"] = "disabled by MOE_QPERM_CUDA=0 (control path: Triton kernels)"
        _say(_STATE["why"])
        return False
    if not torch.cuda.is_available():
        _STATE["why"] = "CUDA not available"
        _say(_STATE["why"])
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("first use inside a CUDA graph capture; build/load must happen eagerly")
        d = build_dir()
        want = _source_hash()
        mode = _acquire_build(d, want)
        _MOD = _import_so(d)
        info = _MOD.device_info()
        if NSTAGE_FWD not in (2, 3, 4) or NSTAGE_BWD not in (2, 3, 4):
            raise RuntimeError(f"NSTAGE must be 2/3/4 (fwd {NSTAGE_FWD}, bwd {NSTAGE_BWD})")
        if GRAB_FWD not in (1, 2) or GRAB_BWD not in (1, 2):
            raise RuntimeError(f"GRAB must be 1/2 (fwd {GRAB_FWD}, bwd {GRAB_BWD})")
        _sched_buffers(torch.cuda.current_device())    # eager allocation: never inside a CUDA-graph capture
        need = max(_MOD.smem_bytes(False, NSTAGE_FWD, False), _MOD.smem_bytes(True, NSTAGE_BWD, True))
        if need > int(info["smem_optin"]):
            raise RuntimeError(f"shared memory {need} B exceeds the device opt-in {info['smem_optin']} B")
        _STATE["ok"] = True
        _STATE["mode"] = mode
        _STATE["attrs"] = {"fwd": kernel_attrs(False, False, False), "bwd": kernel_attrs(True, False, True)}
        _say(f"loaded {mode}: {d}; device sms={info['sms']} cc={info['cc_major']}.{info['cc_minor']} "
             f"smem_optin={info['smem_optin']}; load={'tma' if LOAD_MODE == 1 else 'cpasync'} nstage fwd={NSTAGE_FWD} "
             f"bwd={NSTAGE_BWD} sched={'dynamic' if SCHED_DYNAMIC else 'static'} grab fwd={GRAB_FWD} bwd={GRAB_BWD}; "
             f"attrs fwd={_STATE['attrs']['fwd']} bwd={_STATE['attrs']['bwd']}")
        return True
    except Exception as ex:  # noqa
        _STATE["why"] = f"not loaded: {type(ex).__name__}: {str(ex)[:600]}"
        _STATE["ok"] = False
        _say(_STATE["why"] + " -- the Triton kernels stay in charge for this run")
        return False


def state() -> dict:
    return dict(_STATE)


def _sched_buffers(dev_index: int):
    """The device's scheduler counters (allocated once, zero; the kernels leave them zero) and the empty tensor that
    requests the static partition."""
    if dev_index not in _SCHED_BUF:
        dev = torch.device("cuda", dev_index)
        _SCHED_BUF[dev_index] = torch.zeros((64,), dtype=torch.int32, device=dev)
        _EMPTY[dev_index] = torch.empty((0,), dtype=torch.int32, device=dev)
    return _SCHED_BUF[dev_index], _EMPTY[dev_index]


def _sched_arg(dev: torch.device, scaled: bool, dynamic: Optional[bool]):
    """(sched tensor, grab) for one launch: the counter slot of the direction, or the empty tensor for static."""
    buf, empty = _sched_buffers(dev.index if dev.index is not None else torch.cuda.current_device())
    dyn = SCHED_DYNAMIC if dynamic is None else bool(dynamic)
    if not dyn:
        return empty
    return buf[32:34] if scaled else buf[0:2]


def kernel_attrs(scaled: bool, write_bf16: bool, colsum: bool, load_mode: Optional[int] = None,
                 nstage: Optional[int] = None) -> dict:
    """Registers per thread, resident CTAs per SM, dynamic/static shared and local memory of one kernel variant on this
    device (cudaFuncGetAttributes + the occupancy calculator)."""
    lm = LOAD_MODE if load_mode is None else load_mode
    ns = (NSTAGE_BWD if scaled else NSTAGE_FWD) if nstage is None else nstage
    return dict(_MOD.kernel_attrs(bool(scaled), bool(write_bf16), bool(colsum), int(lm), int(ns)))


def disable(reason: str) -> None:
    """A kernel call raised: count it, drop to the Triton kernels for the rest of the run, say so once."""
    _STATE["failed_calls"] += 1
    _STATE["ok"] = False
    _STATE["why"] = reason
    _say(f"disabled: {reason}")


# ------------------------------------------------------------------------------------------------ eligibility
def enabled_for(rw_layout: int, cw_layout: int, div_mode: int, pad_multiple: int, H: int, E: int) -> bool:
    """The CUDA kernels implement rowwise layout 0 / columnwise layout 3 / amax * (1/448), 128-row units inside one
    expert group (pad_multiple % 128 == 0), E <= 256; anything else stays on the Triton path."""
    if not ENABLED:
        return False
    if not _STATE["ok"]:
        if _STATE["tried"]:
            return False
        if not ensure_loaded():
            return False
    return (rw_layout == 0 and cw_layout == 3 and div_mode == 0 and pad_multiple % 128 == 0 and H % 32 == 0
            and 1 <= E <= 256)


def _scale_numel(P: int, H: int) -> int:
    return (P // 128) * ((H + 127) // 128) * 512


# ------------------------------------------------------------------------------------------------ the ops
def gather_quant(src: torch.Tensor, row_to_token: torch.Tensor, tokens_per_expert: torch.Tensor, P: int,
                 pad_multiple: int, write_bf16: bool, nstage: Optional[int] = None, load_mode: Optional[int] = None,
                 dynamic: Optional[bool] = None, grab: Optional[int] = None):
    T, H = src.shape
    dev = src.device
    n_scale = _scale_numel(P, H)
    rw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    cw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    rws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)    # tail beyond the rows in use reads as zero
    cws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)
    out = torch.empty((P, H) if write_bf16 else (0, H), dtype=torch.bfloat16, device=dev)
    if P == 0:
        return rw, rws, cw, cws, out
    _MOD.gather_quant(src, row_to_token, tokens_per_expert, rw, rws, cw, cws, out, P, pad_multiple, bool(write_bf16),
                      LOAD_MODE if load_mode is None else load_mode, NSTAGE_FWD if nstage is None else nstage,
                      _sched_arg(dev, False, dynamic), GRAB_FWD if grab is None else int(grab))
    return rw, rws, cw, cws, out


def gather_quant_scaled(src: torch.Tensor, row_to_token: torch.Tensor, permuted_probs: torch.Tensor, z: torch.Tensor,
                        tokens_per_expert: torch.Tensor, P: int, pad_multiple: int, write_bf16: bool, colsum: bool,
                        nstage: Optional[int] = None, load_mode: Optional[int] = None, dynamic: Optional[bool] = None,
                        grab: Optional[int] = None):
    T, H = src.shape
    E = tokens_per_expert.numel()
    dev = src.device
    n_scale = _scale_numel(P, H)
    KT = (H + 127) // 128
    rw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    cw = torch.empty((P * H,), dtype=torch.uint8, device=dev)
    rws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)
    cws = torch.zeros((n_scale,), dtype=torch.uint8, device=dev)
    dprob = torch.empty((P,), dtype=torch.float32, device=dev)
    out = torch.empty((P, H) if write_bf16 else (0, H), dtype=torch.bfloat16, device=dev)
    dbias = torch.zeros((E, H) if colsum else (0, H), dtype=torch.float32, device=dev)
    if P == 0:
        return rw, rws, cw, cws, dprob, out, dbias
    part = torch.empty((KT * P,), dtype=torch.float32, device=dev)    # per column tile partial dot products
    _MOD.gather_quant_scaled(src, row_to_token, permuted_probs, z, tokens_per_expert, rw, rws, cw, cws, out, part,
                             dprob, dbias, P, pad_multiple, bool(write_bf16), bool(colsum),
                             LOAD_MODE if load_mode is None else load_mode, NSTAGE_BWD if nstage is None else nstage,
                             _sched_arg(dev, True, dynamic), GRAB_BWD if grab is None else int(grab))
    return rw, rws, cw, cws, dprob, out, dbias
