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

"""Flat per-bucket capturable master Adam as a CUDA C++ extension (torch op gpt_oss::flat_adam_cuda).

Builds flat_adam_cuda.cu + flat_adam_cuda_binding.cpp with nvcc into <build dir>/flat_adam_cuda.so
(one build per source hash, shared by every rank through a lock file) and runs the update of TE's
multi_tensor_adam_capturable_master bit for bit (same expressions, --use_fast_math like TE's adam.cu) as ONE launch per
bucket shard: bf16 grad / param shard and flat fp32 master / exp_avg / exp_avg_sq indexed by the same element offset, 16-B
loads/stores, the device scalars (noop, clip coefficient, lr, step) read once per block, a sorted segment table mapping
element runs to their param group.  25 launches per step (96.6M-element shards, ~40 segments) replace TE's 150
per-tensor-list launches (20.55 ms/step); 28 B/element (write_g off) against TE's 30.  flat_adam() documents the arguments.

Knobs (read at import): FLAT_ADAM=0 keeps TE's per-list launches; FLAT_ADAM_THREADS 128 (default; 6.89 ms standalone for
25 shards vs 7.08 at 256) / 256 / 512; FLAT_ADAM_WAVES 0 (default: one CTA per thread-tile; every grid-stride variant
measured 7.5-7.8 ms) or the number of full-occupancy waves; FLAT_ADAM_WRITE_G=1 writes the scaled grad back like TE;
FLAT_ADAM_BUILD_DIR / _ARCH / _STD / _BUILD_TIMEOUT as in rmsnorm_bwd_cuda.py.
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
from typing import List, Optional, Sequence, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAME = "flat_adam_cuda"
_SOURCES = ("flat_adam_cuda.cu", "flat_adam_cuda_binding.cpp", "flat_adam_cuda.h")

ENABLED = os.getenv("FLAT_ADAM", "1") == "1"
THREADS = int(os.getenv("FLAT_ADAM_THREADS", "128"))
WAVES = int(os.getenv("FLAT_ADAM_WAVES", "0"))
WRITE_G = os.getenv("FLAT_ADAM_WRITE_G", "0") == "1"
BUILD_TIMEOUT_S = int(os.getenv("FLAT_ADAM_BUILD_TIMEOUT", "900"))
VEC = 8

_MOD = None
_STATE = {"tried": False, "ok": False, "why": "", "build_s": 0.0, "mode": "", "calls": 0, "sms": 0, "blocks_per_sm": 0,
          "max_seg": 0, "max_groups": 0}


def _say(msg: str) -> None:
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"FLAT_ADAM rank={rank} {msg}\n")
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
    forced = os.getenv("FLAT_ADAM_ARCH", "").strip()
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
    d = os.getenv("FLAT_ADAM_BUILD_DIR", "").strip()
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
    forced = os.getenv("FLAT_ADAM_STD", "").strip()
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
                # --use_fast_math: TE compiles multi_tensor/adam.cu with it (common/CMakeLists.txt nvte_sources_with_fast_math);
                # the same flag + the same expressions = the same update bits.
                ok = _run(common + flags + ["--use_fast_math", "-Xptxas", "-v", "-c", os.path.join(_HERE, _SOURCES[0]), "-o", kobj],
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


def ensure_loaded() -> bool:
    """Load (building if needed) the extension; False (reason logged once) leaves TE's per-list launches in charge."""
    global _MOD
    if _STATE["tried"]:
        return _STATE["ok"]
    _STATE["tried"] = True
    if not ENABLED:
        _STATE["why"] = "disabled by FLAT_ADAM=0 (control path: TE multi_tensor_adam_capturable_master per tensor list)"
        _say(_STATE["why"])
        return False
    if not torch.cuda.is_available():
        _STATE["why"] = "CUDA not available"
        _say(_STATE["why"])
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("first use inside a CUDA graph capture; build/load must happen eagerly")
        if THREADS not in (128, 256, 512):
            raise RuntimeError(f"FLAT_ADAM_THREADS must be 128, 256 or 512 (got {THREADS})")
        d = build_dir()
        want = _source_hash()
        mode = _acquire_build(d, want)
        _MOD = _import_so(d)
        info = _MOD.device_info()
        at = dict(_MOD.kernel_attrs(THREADS))
        _STATE["ok"] = True
        _STATE["mode"] = mode
        _STATE["sms"] = int(info["sms"])
        _STATE["blocks_per_sm"] = int(at["blocks_per_sm"])
        _STATE["max_seg"] = int(info["max_seg"])
        _STATE["max_groups"] = int(info["max_groups"])
        _STATE["attrs"] = at
        _say(f"loaded {mode}: {d}; device sms={info['sms']} cc={info['cc_major']}.{info['cc_minor']}; threads={THREADS} waves={WAVES} "
             f"write_g={WRITE_G}; attrs {at}")
        return True
    except Exception as ex:  # noqa
        _STATE["why"] = f"not loaded: {type(ex).__name__}: {str(ex)[:600]}"
        _say(_STATE["why"] + " -- TE's per-list launches stay in charge")
        _STATE["ok"] = False
        return False


def state() -> dict:
    return dict(_STATE, enabled=ENABLED, threads=THREADS, waves=WAVES, write_g=WRITE_G)


def kernel_attrs(threads: int) -> dict:
    assert _MOD is not None
    return dict(_MOD.kernel_attrs(int(threads)))


# ------------------------------------------------------------------------------------------------ planning helpers
def grid_for(numel: int, threads: int = -1, waves: int = -1) -> int:
    """Grid-stride grid: the full-occupancy grid x waves, never more CTAs than thread-tiles."""
    threads = THREADS if threads <= 0 else threads
    waves = WAVES if waves < 0 else waves
    tiles = max(1, -(-(numel // VEC) // threads))
    if waves == 0 or not _STATE["ok"]:
        return tiles
    occ = max(1, _STATE["blocks_per_sm"] if threads == THREADS else int(_MOD.kernel_attrs(threads)["blocks_per_sm"]))
    return max(1, min(tiles, _STATE["sms"] * occ * waves))


def plan_segments(pieces: Sequence[Tuple[int, int, int]], numel: int, device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """pieces: (offset, length, group) of every parameter piece inside the shard (disjoint).  Returns the device
    segment table (seg_start int64 [nseg+1], seg_group int32 [nseg]) with adjacent same-group pieces merged and the
    uncovered ranges marked -1, and the number of segments."""
    ps = sorted((int(o), int(k), int(g)) for o, k, g in pieces if k > 0)
    starts: List[int] = []
    groups: List[int] = []
    cur = 0
    for o, k, g in ps:
        if o < cur:
            raise ValueError(f"overlapping pieces at offset {o} (< {cur})")
        if o > cur:                       # uncovered range (bucket padding): its own -1 segment
            starts.append(cur)
            groups.append(-1)
            cur = o
        if not (groups and groups[-1] == g):   # a new run (the previous run, if any, ends exactly at o)
            starts.append(o)
            groups.append(g)
        cur = o + k
    if cur > numel:
        raise ValueError(f"pieces extend past the shard ({cur} > {numel})")
    if cur < numel:
        starts.append(cur)
        groups.append(-1)
    if not starts:
        starts, groups = [0], [-1]
    starts.append(numel)
    seg_start = torch.tensor(starts, dtype=torch.int64, device=device)
    seg_group = torch.tensor(groups, dtype=torch.int32, device=device)
    return seg_start, seg_group, len(groups)


# ------------------------------------------------------------------------------------------------ the op
@torch.library.custom_op("gpt_oss::flat_adam_cuda", mutates_args=("g", "p", "master", "m", "v"))
def flat_adam_cuda(
    g: torch.Tensor, p: torch.Tensor, master: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    noop: torch.Tensor, inv_scale: torch.Tensor, seg_start: torch.Tensor, seg_group: torch.Tensor,
    lrs: List[torch.Tensor], steps: List[torch.Tensor],
    beta1: List[float], beta2: List[float], eps: List[float], decay: List[float],
    bias_correction: List[int], mode: List[int], grid: int, threads: int, write_g: bool,
) -> None:
    assert _MOD is not None, "flat_adam_cuda: extension not loaded (call ensure_loaded() eagerly first)"
    _MOD.flat_adam(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, list(lrs), list(steps),
                   [float(x) for x in beta1], [float(x) for x in beta2], [float(x) for x in eps], [float(x) for x in decay],
                   [int(x) for x in bias_correction], [int(x) for x in mode], int(grid), int(threads), bool(write_g))
    _STATE["calls"] += 1


@flat_adam_cuda.register_fake
def _(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, lrs, steps, beta1, beta2, eps, decay, bias_correction, mode,
      grid, threads, write_g):
    return None


def flat_adam(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, lrs, steps, beta1, beta2, eps, decay,
              bias_correction, mode, grid: int = 0, threads: int = 0, write_g: Optional[bool] = None) -> None:
    """One launch over a bucket shard (hook-facing).

    g             bf16 [n]  this rank's reduce-scatter shard of a bucket's grad_data (read; scaled by *inv_scale on load)
    p             bf16 [n]  the same shard of the bucket's param_data (written: the all-gather source)
    master, m, v  fp32 [n]  flat master / exp_avg / exp_avg_sq of the shard (written), element-aligned with g / p
    noop          int32 [1] device flag, 1 = do nothing;  inv_scale  fp32 [1] device clip coefficient
    seg_start     int64 [nseg+1], seg_group int32 [nseg]: sorted runs of elements -> param-group index (-1 = padding)
    lrs / steps   per-group device scalars (fp32 [1] / int32 [1]); beta1 / beta2 / eps / decay / bias_correction / mode:
                  per-group constants.  grid / threads: launch shape (0 = grid_for() / FLAT_ADAM_THREADS); write_g: also
                  store the scaled grad back like TE (None = FLAT_ADAM_WRITE_G)."""
    threads = THREADS if threads <= 0 else threads
    grid = grid_for(g.numel(), threads) if grid <= 0 else grid
    torch.ops.gpt_oss.flat_adam_cuda(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, list(lrs), list(steps),
                                     list(beta1), list(beta2), list(eps), list(decay), list(bias_correction), list(mode),
                                     int(grid), int(threads), WRITE_G if write_g is None else bool(write_g))


# ------------------------------------------------------------------------------------------------ standalone check
def get_inputs():
    """A 2^20-element shard, two groups (decay / no decay) in 64-element-aligned pieces with one padding gap."""
    n = 1 << 20
    torch.manual_seed(0)
    dev = "cuda"
    g = (torch.randn(n, device=dev) * 1e-3).to(torch.bfloat16)
    p = torch.randn(n, device=dev).to(torch.bfloat16)
    master = p.float().clone()
    m = torch.randn(n, device=dev) * 1e-4
    v = torch.rand(n, device=dev) * 1e-6
    pieces = [(0, 300032, 0), (300032, 64, 1), (300096, 400000, 0), (700160, 2880, 1), (703040, n - 703040 - 4096, 0)]
    seg_start, seg_group, _ = plan_segments(pieces, n, dev)
    noop = torch.zeros(1, dtype=torch.int32, device=dev)
    inv_scale = torch.full((1,), 0.7, dtype=torch.float32, device=dev)
    lr0 = torch.full((1,), 3e-4, dtype=torch.float32, device=dev)
    lr1 = torch.full((1,), 3e-4, dtype=torch.float32, device=dev)
    step = torch.full((1,), 7, dtype=torch.int32, device=dev)
    return [g, p, master, m, v, noop, inv_scale, seg_start, seg_group, [lr0, lr1], [step, step], pieces]


def kernel_fn(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, lrs, steps, pieces):
    assert ensure_loaded(), _STATE["why"]
    g, p, master, m, v = (t.clone() for t in (g, p, master, m, v))
    flat_adam(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, lrs, steps, [0.9, 0.9], [0.95, 0.95],
              [1e-5, 1e-5], [0.1, 0.0], [1, 1], [1, 1], write_g=False)
    return p, master, m, v


def reference_fn(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, lrs, steps, pieces):
    """fp32 AdamW on every piece in torch (same formula; the bit-exact reference is TE's kernel, see the hook's self-check)."""
    p, master, m, v = (t.clone() for t in (p, master, m, v))
    wd = [0.1, 0.0]
    for off, k, grp in pieces:
        sl = slice(off, off + k)
        gg = g[sl].float() * inv_scale
        m[sl] = 0.9 * m[sl] + 0.1 * gg
        v[sl] = 0.95 * v[sl] + 0.05 * gg * gg
        st = int(steps[grp].item())
        bc1 = 1 - 0.9 ** st
        bc2 = 1 - 0.95 ** st
        upd = (m[sl] / bc1) / ((v[sl] / bc2).sqrt() + 1e-5) + wd[grp] * master[sl]
        master[sl] = master[sl] - lrs[grp] * upd
        p[sl] = master[sl].to(torch.bfloat16)
    return p, master, m, v


if __name__ == "__main__":
    xs = get_inputs()
    outs = kernel_fn(*xs)
    refs = reference_fn(*xs)
    torch.cuda.synchronize()
    for nme, a, r in zip(("p", "master", "m", "v"), outs, refs):
        print(nme, "max abs diff", (a.float() - r.float()).abs().max().item(), "ref max", r.float().abs().max().item())
