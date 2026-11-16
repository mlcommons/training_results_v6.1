# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from generative_recommenders.ops.utils import arm_print_all_ranks

ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"
HIPCC_ENV_VAR: str = ENV_VAR + "_HIPCC"
CACHE_ENV_VAR: str = ENV_VAR + "_CACHE"
MLPS_ENV_VAR: str = ENV_VAR + "_MLPS"
FIN_ENV_VAR: str = ENV_VAR + "_FIN"
TAG: str = "[preproc-l2-x2lb]"
ARMED_PREFIX: str = "armed OK: " + TAG
PASS_NAME: str = "x2lb_pass"
PR7_FIN_NAME: str = "ln_bwd_flydsl_fin_d256_n3_wf32"
SHIPPED: str = "shipped dgrad + SwishLN bwd"

ABI_VERSION: int = 2026101202
VARIANT: int = 0


SOURCE_SHA256: str = "f2b0a277e5266c3f8431a5a978f67e17a8ad691d8c78a62935cd0fa5a99ee783"


PINNED_ASM_NORM_SHA256: str = "57de20b6ef509eba1429f19a9486e1475c390a2cf9b851f6bda6683960511e50"
LAB_CODE_OBJECT_SHA256: Dict[str, str] = {
    ".text": "1873c0cbded354588b8f08a384036bb938b63bd7d5d766aebde586472ee39334",
    ".rodata": "c5c7b8872343f81069d9685890e98561bdf0547d78759934a1e54e6ad7cbb484",
}

ISA_FILES_SHA256: Dict[str, str] = {
    "isa_check.py": "812a0d3c92315feecb55ff77780db12b6b3c4a3f45aad6e40d9bb638e61a66d9",
    "isa_dataflow.py": "8cbe79b7d1773736662dc1a225a3e1ae9837d3d44ad00caefc6c4cb517c6244a",
    "isa_exec.py": "570e9221847db6d483ee62aa7204c585db504924b34e71814dc1e5c4a3f0f538",
    "emu/x2lb_emu.py": "f1e299f1b5922cfb9c67ee62b4f735c42bbea812cee3833f34b2b300c4de732c",
}

HID: int = 256
KD: int = 512
BM: int = 64
M_MIN: int = 1 << 16
M_MAX: int = 4_200_000
LIM_VGPR: int = 248
LIM_LDS: int = 163_840
LDS_S1: int = 99_328
STRESS_GRID: int = 7

M_DP: int = 2_356_560
M_A: int = 2_424_832
M_B: int = 2_492_120
M_FIN: int = 16_383
SPLIT_TAIL: int = 131_072
CHECK_POINTS: Tuple[Tuple[int, str], ...] = ((M_DP, "all"), (M_A, "dp_region"), (M_B, "dp_region"),
                                             (M_FIN, "report"))
CANARY_ROWS: int = 64
HEADROOM_SLACK: int = 1 << 30

KERNEL_S1: str = "_ZN4x2lb9x2lb_passILi0ELi0EEEvNS_6ParamsE"
KERNEL_EMIT: str = "_ZN4x2lb9x2lb_passILi0ELi1EEEvNS_6ParamsE"
KERNEL_S0: str = "_ZN4x2lb9x2lb_passILi1ELi0EEEvNS_6ParamsE"
KERNEL_FIN: str = "_ZN4x2lb8x2lb_finEPKfiPf"

_ARCH: str = "gfx950"
_HERE: Path = Path(__file__).resolve().parent
_SOURCE: Path = _HERE / "x2lb_pass.cu"
_ISA_DIR: Path = _HERE / "x2lb_isa"

_FLAGS: Tuple[str, ...] = (
    "-DKITTENS_CDNA4",
    "-std=c++20",
    "-w",
    "-O3",
    "-mllvm",
    "-amdgpu-sched-strategy=iterative-ilp",
    "-ffp-contract=off",
    "--save-temps",
    "-shared",
    "-fPIC",
    "-Rpass-analysis=kernel-resource-usage",
)

_HIPCC_ENV: Tuple[str, ...] = ("HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND", "HIP_CLANG_PATH",
                               "HIP_DEVICE_LIB_PATH")
_CACHE_ROOT: str = "/tmp/dlrm_hstu_preproc_l2_x2lb"
_LIB_NAME: str = "libx2lb_pass.so"
_ASM_NAME: str = "x2lb_pass.gfx950.s"
_META_NAME: str = "meta.json"
_PRIVATE_PREFIX: str = "x2lb_private_"
_BUILD_TIMEOUT_S: int = 600
_ISA_TIMEOUT_S: int = 1200

_REQUESTED: Optional[bool] = None
_MLPS: Optional[Tuple[int, ...]] = None
_FIN: Optional[str] = None
_BROKEN: Optional[str] = None
_lock = threading.Lock()
_lib: Optional["_Lib"] = None
_toolchains: Dict[str, str] = {}
_armed: Set[int] = set()
_ONCE: Set[Tuple[str, str]] = set()
_ZERO: Dict[int, torch.Tensor] = {}
_PENDING: "weakref.WeakSet[Any]" = weakref.WeakSet()
_P1_STATE: Optional[str] = None
_P2_STATE: Optional[str] = None
_P2_SKIP: Optional[str] = None
_LB: Any = None
PHASE1_INFO: Dict[str, Any] = {}
PREBUILD_INFO: Dict[int, Dict[str, Any]] = {}
FUSED_CALLS: Dict[int, int] = {}
DECLINES: Dict[str, int] = {}


def _say(msg: str) -> None:
    arm_print_all_ranks(msg)


def _once(site: str, reason: str, msg: str) -> None:
    if (site, reason) not in _ONCE:
        _ONCE.add((site, reason))
        _say(msg)


def _cls(why: str) -> str:
    return re.sub(r"\d+", "N", why)


def requested() -> bool:
    global _REQUESTED
    if _REQUESTED is None:
        _REQUESTED = os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
    return _REQUESTED


def mlps() -> Tuple[int, ...]:
    global _MLPS
    if _MLPS is None:
        raw = os.environ.get(MLPS_ENV_VAR, "").strip()
        if raw in ("", "0,1", "1,0"):
            _MLPS = (0, 1)
        elif raw in ("0", "1"):
            _MLPS = (int(raw),)
        else:
            raise ValueError(f"{MLPS_ENV_VAR}={raw!r}: must be 0,1 (default), 0 or 1")
    return _MLPS


def fin_mode() -> str:
    global _FIN
    if _FIN is None:
        raw = os.environ.get(FIN_ENV_VAR, "").strip().lower()
        if raw not in ("", "pr7", "x2lb"):
            raise ValueError(f"{FIN_ENV_VAR}={raw!r}: must be pr7 (default) or x2lb")
        _FIN = raw or "pr7"
    return _FIN


def broken() -> Optional[str]:
    return _BROKEN


def latch_off(where: str, exc: BaseException) -> None:
    global _BROKEN
    if _BROKEN is None:
        _BROKEN = f"{where}: {type(exc).__name__}: {str(exc)[:300]}"
        try:
            _say(f"{TAG} FAILED CLOSED -> {SHIPPED} for the rest of the run ({_BROKEN})")
        except Exception:
            pass


def ignored(why: str) -> None:
    _once("ignored", _cls(why), f"{TAG} {ENV_VAR}=1 IGNORED (NOT armed): {why}; {SHIPPED} for the whole run")


def phase2_not_run(why: str) -> None:
    global _P2_SKIP
    if _P2_SKIP is None:
        _P2_SKIP = str(why)


def _dev_index(dev: torch.device) -> int:
    if dev.index is not None:
        return int(dev.index)
    return int(torch.cuda.current_device()) if dev.type == "cuda" else -1


def active(dev: torch.device) -> bool:
    try:
        return requested() and _BROKEN is None and _dev_index(dev) in _armed
    except Exception:
        return False


def _not_armed_line(dev: torch.device) -> None:
    if _BROKEN is not None or _P1_STATE in ("ignored", "failed") or _P2_STATE == "failed":
        return
    idx = _dev_index(dev)
    if _P1_STATE == "ok" and _P2_STATE is None:
        why = _P2_SKIP or ("the startup preload did not run it (needs mode streaming-train-eval with "
                           "$BLAS_WARMUP and DL_PRESTART)")
        ignored(f"phase-2 self-check not run ({why})")
        return
    _once("not_armed", str(idx), f"{TAG} NOT armed on cuda:{idx}: no pre-run_start prebuild passed on this device "
          f"(BLAS_WARMUP off?) -> {SHIPPED} (nothing is built after RUN_START)")


def _hipcc() -> str:
    override = os.environ.get(HIPCC_ENV_VAR, "").strip()
    if override:
        return override
    rocm = os.environ.get("ROCM_PATH", "").strip()
    if rocm:
        cand = os.path.join(rocm, "bin", "hipcc")
        if os.path.exists(cand):
            return cand
    return "/opt/rocm/bin/hipcc"


def _toolchain(hipcc: str) -> str:
    v = _toolchains.get(hipcc)
    if v is None:
        proc = subprocess.run([hipcc, "--version"], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"{hipcc} --version rc={proc.returncode}: "
                               f"{((proc.stderr or '') + (proc.stdout or '')).strip()[-400:]}")
        v = proc.stdout
        _toolchains[hipcc] = v
    return v


def build_flags(src: bytes) -> List[str]:
    return [f"--offload-arch={_ARCH}", *_FLAGS, f'-DX2LB_SRC_SHA256="{hashlib.sha256(src).hexdigest()}"']


def cache_dir(source: Path = _SOURCE) -> Path:
    root = os.environ.get(CACHE_ENV_VAR, "").strip() or _CACHE_ROOT
    src = source.read_bytes()
    h = hashlib.sha256()
    h.update(src)
    h.update(b"\0" + "\0".join(build_flags(src)).encode())
    h.update(b"\0" + _ARCH.encode())

    h.update(b"\0" + _toolchain(_hipcc()).encode())
    h.update(b"\0" + "\0".join(f"{k}={os.environ.get(k, '')}" for k in _HIPCC_ENV).encode())
    return Path(root) / h.hexdigest()[:16]


def _trusted(p: Path) -> bool:
    st = p.stat()
    return st.st_uid == os.getuid() and not st.st_mode & 0o022


def _ensure_root(root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    else:
        os.chmod(root, 0o700)
    st = root.stat()
    if st.st_uid == os.getuid() and st.st_mode & 0o022:
        os.chmod(root, st.st_mode & 0o7755)


def _run_group(cmd: List[str], timeout: int, cwd: Optional[str] = None,
               env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True, cwd=cwd, env=env)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise RuntimeError(f"timed out after {timeout}s: {' '.join(cmd)[:300]}")
    return proc.returncode, out or ""


def parse_resource_remarks(log: str) -> List[Dict[str, str]]:
    kernels: List[Dict[str, str]] = []
    for line in log.splitlines():
        m = re.search(r"remark:\s+\S+?:\d+:\d+:\s+(.*?)\s*\[-Rpass-analysis", line)
        if not m:
            continue
        kv = re.match(r"\s*([^:]+):\s*(\S+)", m.group(1))
        if not kv:
            continue
        k, v = kv.group(1).strip(), kv.group(2)
        if k == "Function Name":
            kernels.append({})
        if kernels:
            kernels[-1][k] = v
    return kernels


def check_resources(kernels: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k in kernels:
        nm = k.get("Function Name", "")
        if PASS_NAME not in nm and nm != KERNEL_FIN:
            continue
        out[nm] = dict(vgpr=int(k.get("VGPRs", "-1")), agpr=int(k.get("AGPRs", "-1")),
                       sgpr=int(k.get("TotalSGPRs", "-1")), vgpr_spill=int(k.get("VGPRs Spill", "-1")),
                       sgpr_spill=int(k.get("SGPRs Spill", "-1")),
                       scratch=int(k.get("ScratchSize [bytes/lane]", "-1")),
                       lds=int(k.get("LDS Size [bytes/block]", "-1")),
                       occupancy=int(k.get("Occupancy [waves/SIMD]", "-1")))
    for nm in (KERNEL_S1, KERNEL_EMIT, KERNEL_S0, KERNEL_FIN):
        if nm not in out:
            raise RuntimeError(f"resource census: no remarks for {nm}")
    for nm, v in out.items():
        if nm == KERNEL_FIN:
            if v["vgpr_spill"] != 0 or v["sgpr_spill"] != 0 or v["scratch"] != 0:
                raise RuntimeError(f"resource census: x2lb_fin {v} (needs 0 spill, 0 scratch)")
            continue
        unified = ((v["vgpr"] + 3) // 4) * 4 + max(v["agpr"], 0)
        v["unified_vgpr"] = unified
        if (v["vgpr"] < 0 or v["vgpr_spill"] != 0 or v["sgpr_spill"] != 0 or v["scratch"] != 0
                or unified > LIM_VGPR or v["occupancy"] != 2 or not (0 < v["lds"] <= LIM_LDS)
                or (nm == KERNEL_S1 and v["lds"] != LDS_S1)):
            raise RuntimeError(f"resource census: {nm} {v} (needs 0 spill, 0 scratch, occupancy 2, <= {LIM_VGPR} "
                               f"unified VGPR, LDS <= {LIM_LDS}; x2lb_pass<0,0> LDS == {LDS_S1})")
    return out


def asm_norm_sha256(text: bytes) -> str:
    h = hashlib.sha256()
    for ln in text.splitlines(keepends=True):
        if b"__hip_cuid_" not in ln:
            h.update(ln)
    return h.hexdigest()


def isa_files_sha256() -> Dict[str, str]:
    return {n: hashlib.sha256((_ISA_DIR / n).read_bytes()).hexdigest() for n in ISA_FILES_SHA256}


_ISA_CHILD = r"""
import json, os, sys, types
d, asm, pinned = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
sys.dont_write_bytecode = True
sys.path.insert(0, d)
if pinned:
    df = types.ModuleType("isa_dataflow")
    df.check = lambda name, lines: dict(errors=[], info=["C15 not re-run: pinned device assembly"])
    ex = types.ModuleType("isa_exec")
    ex.check = lambda *a, **k: "C16/C17 not re-run: pinned device assembly"
    sys.modules["isa_dataflow"], sys.modules["isa_exec"] = df, ex
import isa_check as IC
assert os.path.realpath(os.path.dirname(IC.__file__)) == os.path.realpath(d), IC.__file__
kern, desc = IC.split_asm(open(asm).read())
per = {}
for name in list(IC.KERNELS) + [IC.FIN]:
    if name not in kern:
        per[name] = dict(ok=False, failed=["missing"], errors=["kernel not found"])
        continue
    try:
        r = IC.check_fin(kern[name], desc) if name == IC.FIN else IC.check_kernel(name, kern[name], desc)
    except Exception as e:
        per[name] = dict(ok=False, failed=["raised"], errors=[f"{type(e).__name__}: {e}"[:300]])
        continue
    per[name] = dict(label=r["kernel"], ok=bool(r["ok"]), failed=sorted({e.split()[0] for e in r["errors"]}),
                     errors=r["errors"][:6])
print("X2LB_ISA_VERDICT " + json.dumps(per), flush=True)
"""


def isa_gate(asm_path: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()
    got = isa_files_sha256()
    if got != ISA_FILES_SHA256:
        bad = sorted(n for n in ISA_FILES_SHA256 if got.get(n) != ISA_FILES_SHA256[n])
        raise RuntimeError(f"ISA gate files differ from their pinned hashes: {bad}")
    norm = asm_norm_sha256(asm_path.read_bytes())
    pinned = norm == PINNED_ASM_NORM_SHA256
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    rc, out = _run_group([sys.executable, "-B", "-c", _ISA_CHILD, str(_ISA_DIR), str(asm_path), "1" if pinned else "0"],
                         _ISA_TIMEOUT_S, env=env)
    line = [ln for ln in out.splitlines() if ln.startswith("X2LB_ISA_VERDICT ")]
    per: Dict[str, Any] = json.loads(line[-1][len("X2LB_ISA_VERDICT "):]) if (line and rc == 0) else {}
    want = (KERNEL_S1, KERNEL_EMIT, KERNEL_S0, KERNEL_FIN)
    ok = bool(per) and all(per.get(k, {}).get("ok") is True for k in want)
    return dict(ok=ok, mode="pinned" if pinned else "full", asm_norm_sha256=norm, kernels=per, rc=rc,
                checker=got, seconds=round(time.perf_counter() - t0, 1),
                tail=None if ok else out.strip()[-1500:])


def _tool(hipcc: str, name: str) -> Optional[str]:
    root = os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))
    for c in (os.path.join(root, "lib", "llvm", "bin", name), os.path.join(root, "llvm", "bin", name),
              os.path.join("/opt/rocm/llvm/bin", name), os.path.join("/opt/rocm/lib/llvm/bin", name)):
        if os.path.isfile(c):
            return c
    return shutil.which(name)


def code_object_sha256(so: Path, hipcc: str) -> Dict[str, str]:
    objcopy, bundler = _tool(hipcc, "llvm-objcopy"), _tool(hipcc, "clang-offload-bundler")
    if not (objcopy and bundler):
        return {"status": "unknown (llvm-objcopy / clang-offload-bundler not found)"}
    d = tempfile.mkdtemp(prefix="x2lb_co_")
    try:
        fat, co, so_out = os.path.join(d, "fat.bin"), os.path.join(d, "co.o"), os.path.join(d, "so_out.so")
        subprocess.run([objcopy, "--dump-section", f".hip_fatbin={fat}", str(so), so_out], check=True,
                       capture_output=True, timeout=120)
        lst = subprocess.run([bundler, "--list", "--type=o", f"--input={fat}"], capture_output=True, text=True,
                             timeout=120).stdout.split()
        tgt = next((x for x in lst if x.endswith(_ARCH)), None)
        if tgt is None:
            return {"status": f"unknown (no {_ARCH} bundle entry)"}
        subprocess.run([bundler, "--unbundle", "--type=o", f"--targets={tgt}", f"--input={fat}", f"--output={co}"],
                       check=True, capture_output=True, timeout=120)
        res = {"status": "ok"}
        for sec in (".text", ".rodata"):
            out = os.path.join(d, "sec" + sec)
            subprocess.run([objcopy, "-O", "binary", f"--only-section={sec}", co, out], check=True,
                           capture_output=True, timeout=120)
            res[sec] = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        res["lab"] = "identical" if all(res[s] == LAB_CODE_OBJECT_SHA256[s] for s in LAB_CODE_OBJECT_SHA256) \
            else "differs"
        return res
    except Exception as exc:
        return {"status": f"unknown ({type(exc).__name__}: {str(exc)[:160]})"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextmanager
def _entry_lock(d: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError:
        yield
        return
    fd = os.open(str(d / ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_name(f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}-{path.name}")
    try:
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _verdict_current(meta: Dict[str, Any], asm: Path) -> bool:
    v = meta.get("isa") or {}
    return (v.get("ok") is True and v.get("asm_norm_sha256") == asm_norm_sha256(asm.read_bytes())
            and v.get("checker") == ISA_FILES_SHA256 and meta.get("src_sha256") == SOURCE_SHA256)


def build(source: Path = _SOURCE, fresh: bool = False) -> Tuple[Path, Dict[str, Any], float, bool]:
    d = cache_dir(source)
    _ensure_root(d.parent)
    d.mkdir(exist_ok=True, mode=0o700)
    if fresh:
        with _entry_lock(d):
            for n in (_LIB_NAME, _META_NAME, _ASM_NAME):
                try:
                    (d / n).unlink()
                except OSError:
                    pass
    private = fresh or not (_trusted(d.parent) and _trusted(d))
    if private:
        d = Path(tempfile.mkdtemp(prefix=_PRIVATE_PREFIX))
    with _entry_lock(d):
        so, meta_p, asm = d / _LIB_NAME, d / _META_NAME, d / _ASM_NAME
        meta = None
        if so.is_file() and meta_p.is_file() and asm.is_file():
            try:
                meta = json.loads(meta_p.read_text())
            except (OSError, ValueError):
                meta = None
        if meta is not None:
            meta["resources"] = check_resources(meta["kernels_raw"])
            if not _verdict_current(meta, asm):
                meta["isa"] = isa_gate(asm)
                if not meta["isa"]["ok"]:
                    raise RuntimeError(f"ISA gate refuses the cached {so}: {_isa_why(meta['isa'])}")
                _write_json(meta_p, meta)
            return so, meta, 0.0, True
        src = source.read_bytes()
        hipcc = _hipcc()
        uniq = f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
        tmp, tmp_asm, work = d / f"{uniq}-{_LIB_NAME}", d / f"{uniq}-{_ASM_NAME}", d / f"{uniq}-build"
        cmd = [hipcc, str(source.resolve()), *build_flags(src), "-o", str(tmp.resolve())]
        t0 = time.perf_counter()
        try:
            work.mkdir(parents=True, exist_ok=True, mode=0o700)
            rc, out = _run_group(cmd, _BUILD_TIMEOUT_S, cwd=str(work))
            if rc != 0 or not tmp.is_file():
                raise RuntimeError(f"hipcc rc={rc}: {' '.join(cmd)}: {out.strip()[-1200:]}")
            kernels = parse_resource_remarks(out)
            resources = check_resources(kernels)
            dev_s = sorted(work.glob(f"*{_ARCH}*.s"))
            if not dev_s:
                raise RuntimeError("--save-temps produced no gfx950 device assembly: cannot run the ISA gate")
            shutil.copyfile(dev_s[0], tmp_asm)
            isa = isa_gate(tmp_asm)
            if not isa["ok"]:
                raise RuntimeError(f"ISA gate refuses this build: {_isa_why(isa)}")
            build_s = time.perf_counter() - t0
            meta = dict(src_sha256=hashlib.sha256(src).hexdigest(), cmd=cmd, hipcc_version=_toolchain(hipcc),
                        kernels_raw=kernels, resources=resources, isa=isa,
                        code_object=code_object_sha256(tmp, hipcc), build_s=round(build_s, 1))
            os.replace(tmp_asm, asm)
            _write_json(meta_p, meta)
            os.replace(tmp, so)
        finally:
            for p in (tmp, tmp_asm):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            shutil.rmtree(work, ignore_errors=True)
        return so, meta, build_s, False


def _isa_why(isa: Dict[str, Any]) -> str:
    bad = {k: v.get("failed") for k, v in (isa.get("kernels") or {}).items() if not v.get("ok")}
    return (f"mode {isa.get('mode')} rc {isa.get('rc')} failures {bad or '(no verdict)'}"
            + ("" if isa.get("ok") else f" tail {str(isa.get('tail'))[-300:]!r}"))


_VP, _I, _LL, _CP = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_char_p
_LAUNCH_ARGS: Tuple[Any, ...] = (_VP,) * 12 + (_LL,) * 3 + (_VP,)
SIGNATURES: Dict[str, Tuple[Any, Tuple[Any, ...]]] = {
    "x2lb_abi_version": (_I, ()),
    "x2lb_variant": (_I, ()),
    "x2lb_impl": (_CP, ()),
    "x2lb_plan": (_LL, (_LL, _LL)),
    "x2lb_launch": (_I, _LAUNCH_ARGS),
    "x2lb_launch_emit": (_I, _LAUNCH_ARGS),
    "x2lb_launch_s0": (_I, _LAUNCH_ARGS),
    "x2lb_fin_launch": (_I, (_VP, _LL, _VP, _VP)),
    "x2lb_drain_last_error": (_I, ()),
}


class _Lib:
    def __init__(self, so: Path, meta: Dict[str, Any], build_s: float, cached: bool) -> None:
        self.path, self.meta, self.build_s, self.cached = so, meta, build_s, cached
        self.cdll = ctypes.CDLL(str(so))
        for name, (res, args) in SIGNATURES.items():
            fn = getattr(self.cdll, name)
            fn.restype = res
            fn.argtypes = list(args)
            setattr(self, name, fn)
        got = int(self.x2lb_abi_version())
        if got != ABI_VERSION:
            raise RuntimeError(f"{so}: x2lb_abi_version()={got}, expected {ABI_VERSION}")
        var = int(self.x2lb_variant())
        if var != VARIANT:
            raise RuntimeError(f"{so}: x2lb_variant()={var} (a negative-control build), expected {VARIANT}")
        impl = (self.x2lb_impl() or b"").decode(errors="replace")
        if f"src_sha256={SOURCE_SHA256}" not in impl:
            raise RuntimeError(f"{so}: x2lb_impl() does not carry src_sha256={SOURCE_SHA256[:16]}...: {impl[-120:]}")
        self.impl = impl

    def plan(self, M: int, num_cu: int) -> int:
        g = int(self.x2lb_plan(int(M), int(num_cu)))
        if g < 1:
            raise ValueError(f"x2lb_plan(M={M}, num_cu={num_cu}) rejected ({g})")
        return g


def load() -> "_Lib":
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                built = build()
                try:
                    lib = _Lib(*built)
                except Exception:
                    _rm_private(built[0])
                    if not built[3]:
                        raise
                    built = build(fresh=True)
                    try:
                        lib = _Lib(*built)
                    finally:
                        _rm_private(built[0])
                _rm_private(built[0])
                _lib = lib
    return _lib


def _rm_private(so: Path) -> None:
    if so.parent.name.startswith(_PRIVATE_PREFIX):
        shutil.rmtree(so.parent, ignore_errors=True)


def _check_rc(rc: int, what: str) -> None:
    if rc < 0:
        _once("stale", str(-rc), f"{TAG} stale HIP error {-rc} drained before {what} "
              "(left on this thread by an earlier call, not ours)")
    elif rc != 0:
        raise RuntimeError(f"{what} returned hip error {rc}")


def _ptr(t: Optional[torch.Tensor]) -> Optional[int]:
    return None if t is None else int(t.data_ptr())


def _fin_pr7(part: torch.Tensor, dw: torch.Tensor, db: torch.Tensor, dxs: torch.Tensor, grid: int,
             stream: Any) -> None:


    lb = _LB
    if lb is None:
        raise RuntimeError("ops/flydsl/ln_bwd not loaded (phase 2 did not run)")
    lb.build_finalize(HID, 3, True)(part, dw, db, dxs, int(grid), stream=lb.fx.Stream(stream))


class _CudaRuntime:
    def __init__(self) -> None:
        self._ncu: Dict[int, int] = {}

    @staticmethod
    def stream(dev: torch.device) -> Any:
        return torch.cuda.current_stream(dev)

    @staticmethod
    def capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def num_cu(self, dev: torch.device) -> int:
        idx = _dev_index(dev)
        n = self._ncu.get(idx)
        if n is None:
            n = int(torch.cuda.get_device_properties(dev).multi_processor_count)
            self._ncu[idx] = n
        return n

    @staticmethod
    def make_zero(dev: torch.device) -> torch.Tensor:
        z = torch.zeros((), dtype=torch.bfloat16, device=dev)
        torch.cuda.current_stream(dev).synchronize()
        return z

    @staticmethod
    def launch(lib: "_Lib", mode: str, dh8: Optional[torch.Tensor], w8t: Optional[torch.Tensor],
               sa: Optional[torch.Tensor], sb: Optional[torch.Tensor], h1: torch.Tensor, mean1: torch.Tensor,
               rstd1: torch.Tensor, nw: torch.Tensor, nb: torch.Tensor, dh1: torch.Tensor,
               da1: Optional[torch.Tensor], part: torch.Tensor, M: int, grid: int, stream: Any) -> None:
        fn = {"s1": lib.x2lb_launch, "emit": lib.x2lb_launch_emit, "s0": lib.x2lb_launch_s0}[mode]
        if int(part.shape[1]) != int(grid):
            raise ValueError(f"PART rows {int(part.shape[1])} != grid {grid}")
        rc = fn(_ptr(dh8), _ptr(w8t), _ptr(sa), _ptr(sb), _ptr(h1), _ptr(mean1), _ptr(rstd1), _ptr(nw), _ptr(nb),
                _ptr(dh1), _ptr(da1), _ptr(part), int(grid), int(M), int(grid), int(stream.cuda_stream))
        _check_rc(int(rc), f"x2lb_launch({mode})")

    @staticmethod
    def fin(lib: "_Lib", mode: str, part: torch.Tensor, grid: int, stream: Any,
            out: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        dev = part.device
        if mode == "x2lb":
            buf = torch.empty((3, HID), dtype=torch.float32, device=dev) if out is None else out
            _check_rc(int(lib.x2lb_fin_launch(part.data_ptr(), int(grid), buf.data_ptr(),
                                              int(stream.cuda_stream))), "x2lb_fin_launch")
            return buf[0], buf[1], buf[2]
        if out is None:
            out = tuple(torch.empty(HID, dtype=torch.float32, device=dev) for _ in range(3))
        dw, db, dxs = out
        _fin_pr7(part, dw, db, dxs, grid, stream)
        return dw, db, dxs

    def fused(self, lib: "_Lib", dh8: torch.Tensor, w8_t: torch.Tensor, dh_scale: torch.Tensor,
              w_scale: torch.Tensor, h1: torch.Tensor, mean1: torch.Tensor, rstd1: torch.Tensor, nw: torch.Tensor,
              nb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], int, Any]:


        if lib is None:
            raise RuntimeError("library not loaded (phase 1 did not run)")
        dev = h1.device
        cur = torch.cuda.current_device()
        if dev.index is not None and dev.index != cur:
            raise ValueError(f"tensors on {dev} but the current device is cuda:{cur}")
        M = int(h1.shape[0])
        grid = lib.plan(M, self.num_cu(dev))
        st = torch.cuda.current_stream(dev)
        dh1 = torch.empty((M, HID), dtype=torch.bfloat16, device=dev)
        part = torch.empty((3, grid, HID), dtype=torch.float32, device=dev)
        self.launch(lib, "s1", dh8, w8_t, dh_scale, w_scale, h1, mean1, rstd1, nw, nb, dh1, None, part, M, grid, st)
        mode = fin_mode()
        dw, db, dxs = self.fin(lib, mode, part, grid, st)
        if mode == "x2lb":
            dw, db, dxs = dw.clone(), db.clone(), dxs.clone()
        return (dh1, dw, db, dxs), grid, st


_RT: Any = _CudaRuntime()


class X2LBHandle:
    __slots__ = ("k", "p0", "proxy", "stream", "result", "sentinel", "state", "grid", "task", "__weakref__")

    def __init__(self, k: int, p0_ctx: Any, stream: Any) -> None:
        self.k = k
        self.p0 = weakref.ref(p0_ctx)
        self.stream = stream
        self.proxy: Optional[weakref.ref] = None
        self.result: Optional[Tuple[torch.Tensor, ...]] = None
        self.sentinel: Optional[torch.Tensor] = None
        self.state = "published"
        self.grid = 0
        self.task: Optional[int] = None

    def foreign(self) -> bool:
        return self.result is not None and self.task is not None and self.task != _graph_task_id()

    def take(self, dy: Optional[torch.Tensor]) -> Optional[Tuple[torch.Tensor, ...]]:
        z = _on_zero(dy)
        if z is not None and z._version != 0:
            _abort_task(self)
            raise RuntimeError(f"{TAG} handshake: mlp {self.k}: the sentinel was written in place (a hook on the "
                               "proxy's gradient modified it)")
        if self.result is None:
            if z is not None:
                _abort_task(self)
                raise RuntimeError(f"{TAG} handshake: mlp {self.k}: a sentinel gradient without fused results")
            return None
        if self.foreign():
            if z is not None:
                _abort_task(self)
                raise RuntimeError(f"{TAG} handshake: mlp {self.k}: a sentinel gradient in a backward other than the "
                                   "one that fused")
            return None
        s = self.sentinel
        if z is None or s is None or dy.data_ptr() != s.data_ptr() or tuple(dy.shape) != tuple(s.shape):
            _abort_task(self)
            raise RuntimeError(f"{TAG} handshake: mlp {self.k}: fused results but the proxy gradient is not the "
                               "sentinel (another consumer or hook changed it)")
        r = self.result
        self.result, self.sentinel, self.state, self.task = None, None, "consumed", None
        _PENDING.discard(self)
        if r[0].is_cuda:
            cur = torch.cuda.current_stream(r[0].device)
            if cur != self.stream:
                cur.wait_stream(self.stream)
                for t in r:
                    t.record_stream(cur)
        return r

    def _drop_own(self) -> None:
        if not self.foreign():
            self.result, self.sentinel, self.state, self.task = None, None, "declined", None
            _PENDING.discard(self)


def _graph_task_id() -> int:
    f = getattr(torch._C, "_current_graph_task_id", None)
    return -1 if f is None else int(f())


def _abort_task(hd: X2LBHandle) -> None:
    hd._drop_own()
    t = _graph_task_id()
    for h in list(_PENDING):
        if h.task == t:
            h._drop_own()


def _on_zero(dy: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if dy is None or dy.dim() != 2 or dy.stride() != (0, 0):
        return None
    p = dy.data_ptr()
    for z in _ZERO.values():
        if z.data_ptr() == p and z.device == dy.device:
            return z
    return None


def _saved_tensor_hooks_active() -> bool:
    f = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    return f is not None and f(False) is not None


def _tracing(x: Any) -> bool:
    from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs

    return bool(xs.tracing(x))


def publish_reason(k: int, h: torch.Tensor, train: bool) -> Optional[str]:
    if not train:
        return "not train"
    if _tracing(h):
        return "tracing"
    if _saved_tensor_hooks_active():
        return "saved-tensor hooks active"
    if _graph_task_id() != -1:
        return "inside a backward"
    if int(k) not in mlps():
        return f"mlp {k} not in {MLPS_ENV_VAR}"
    return None


def publish(ctx: Any, k: int, h: torch.Tensor, train: bool) -> Optional[X2LBHandle]:
    try:
        if not requested() or _BROKEN is not None:
            return None
        if publish_reason(k, h, train) is not None:
            return None
        if not active(h.device):
            _not_armed_line(h.device)
            return None
        return X2LBHandle(int(k), ctx, _RT.stream(h.device))
    except Exception as exc:
        latch_off("publish", exc)
        return None


def _decline(hd: Optional[X2LBHandle], where: str, why: str, say: bool = True) -> None:
    if hd is not None and hd.result is None:
        hd.state = "declined"
    c = _cls(why)
    DECLINES[c] = DECLINES.get(c, 0) + 1
    if say:
        _once("decline", c, f"{TAG} DECLINED at {where} ({why}) -> {SHIPPED} for such calls")


def attach(hd: Optional[X2LBHandle], a1_rm: torch.Tensor, proxy: torch.Tensor) -> Optional[X2LBHandle]:
    if hd is None:
        return None
    try:
        why = None
        if hd.state != "published":
            why = f"handle state {hd.state}"
        elif hd.p0() is None:

            _decline(hd, "attach", "pair 0's ctx gone", say=False)
            return None
        elif _tracing(proxy):
            why = "tracing"
        elif not active(a1_rm.device):
            why = "device not armed"
        elif _RT.stream(a1_rm.device) != hd.stream:
            why = "the tail's forward stream differs from pair 0's"
        elif proxy.dim() != 2 or int(proxy.shape[1]) != HID or proxy.device != a1_rm.device:
            why = f"proxy {tuple(proxy.shape)} on {proxy.device}"
        if why is not None:
            _decline(hd, "attach", why)
            return None
        hd.proxy = weakref.ref(proxy)
        hd.state = "attached"
        return hd
    except Exception as exc:
        latch_off("attach", exc)
        return None


def will_run(p0: Any) -> bool:
    f = getattr(torch._C, "_will_engine_execute_node", None)
    if f is None or p0 is None:
        return False
    try:
        nf = p0.next_functions
        nw_node = nf[3][0] if len(nf) > 3 else None
        return nw_node is not None and bool(f(p0)) and bool(f(nw_node))
    except Exception:
        return False


def call_reason(hd: X2LBHandle, p0: Any) -> Optional[str]:
    if torch.is_grad_enabled():
        return "create_graph backward"
    if torch.is_anomaly_enabled():
        return "anomaly mode"
    if p0 is None:
        return "pair 0's ctx gone"
    if hd.state not in ("attached", "consumed", "declined", "fused"):
        return f"handle state {hd.state}"
    if hd.foreign():
        return "a fused result of another backward is pending (overlapping or aborted backward)"
    if hd.result is not None:
        return "an earlier fused result was never consumed"
    if hd.proxy is None or hd.proxy() is not None:
        return "proxy still referenced (capture, retain_grad or hook)"
    if not will_run(p0):
        return "pair 0 does not run in this backward or is only captured (autograd.grad / backward(inputs=))"
    if _RT.capturing():
        return "CUDA graph capture"
    return None


def _a16(t: torch.Tensor) -> bool:
    return int(t.data_ptr()) % 16 == 0


def tensor_reason(dh8: torch.Tensor, w8_t: torch.Tensor, dh_scale: Optional[torch.Tensor],
                  w_scale: Optional[torch.Tensor], h1: torch.Tensor, nw: torch.Tensor, nb: torch.Tensor,
                  mean1: torch.Tensor, rstd1: torch.Tensor, x_dt: torch.dtype) -> Optional[str]:


    dev = dh8.device
    if h1.dim() != 2:
        return f"h1 dim {h1.dim()}"
    M = int(h1.shape[0])
    if not (M_MIN <= M <= M_MAX):
        return f"M {M} outside [{M_MIN}, {M_MAX}]"
    if x_dt != torch.bfloat16:
        return f"proxy dtype {x_dt}"
    if dh8.dtype != torch.float8_e5m2 or tuple(dh8.shape) != (M, KD) or not dh8.is_contiguous() or not _a16(dh8):
        return f"dh8 {dh8.dtype} {tuple(dh8.shape)} stride {tuple(dh8.stride())} (needs e5m2 [M,512] contiguous, 16-B)"
    if (w8_t.dtype != torch.float8_e4m3fn or tuple(w8_t.shape) != (HID, KD) or not w8_t.is_contiguous()
            or not _a16(w8_t)):
        return f"w8_t {w8_t.dtype} {tuple(w8_t.shape)} (needs e4m3 [256,512] contiguous, 16-B)"
    for name, s in (("dh_scale", dh_scale), ("w_scale", w_scale)):
        if s is None:
            return f"{name} missing"
        if s.dtype != torch.float32 or s.numel() != 1 or s.device != dev or int(s.data_ptr()) % 4:
            return f"{name} {s.dtype} {tuple(s.shape)} on {s.device} (needs fp32, one element, on {dev})"
    if h1.dtype != torch.bfloat16 or tuple(h1.shape) != (M, HID) or not h1.is_contiguous() or not _a16(h1):
        return f"h1 {h1.dtype} {tuple(h1.shape)} stride {tuple(h1.stride())} (needs bf16 [M,256] contiguous, 16-B)"
    for name, t in (("mean1", mean1), ("rstd1", rstd1)):
        if t.dtype != torch.float32 or tuple(t.shape) != (M,) or not t.is_contiguous() or int(t.data_ptr()) % 4:
            return f"{name} {t.dtype} {tuple(t.shape)} (needs fp32 [M] contiguous)"
    for name, t in (("norm weight", nw), ("norm bias", nb)):
        if t.dtype != torch.float32 or tuple(t.shape) != (HID,) or not t.is_contiguous() or not _a16(t):
            return f"{name} {t.dtype} {tuple(t.shape)} (needs fp32 [256] contiguous, 16-B)"
    for name, t in (("w8_t", w8_t), ("h1", h1), ("mean1", mean1), ("rstd1", rstd1), ("norm weight", nw),
                    ("norm bias", nb)):
        if t.device != dev:
            return f"{name} on {t.device}, dh8 on {dev}"
    from generative_recommenders.ops.triton import triton_layer_norm as TLN

    why = TLN.flydsl_ln_bwd_decline_reason(h1, h1, nw, nb, mean1, rstd1, is_swish=True, sum_dx=True)
    if why is not None:
        return f"the FlyDSL ln_bwd would not own the SwishLN backward: {why}"
    return None


def _drop(hd: X2LBHandle) -> None:
    hd._drop_own()


def release(hds: Optional[List[Optional[X2LBHandle]]]) -> None:
    for hd in hds or ():
        if hd is not None:
            try:
                hd._drop_own()
            except Exception:
                pass


def try_fused(hd: X2LBHandle, dh8: torch.Tensor, w8_t: torch.Tensor, dh_scale: Optional[torch.Tensor],
              w_scale: Optional[torch.Tensor], x_dt: torch.dtype) -> Optional[torch.Tensor]:


    try:
        dev = dh8.device
        z = _ZERO.get(_dev_index(dev))
        if z is not None and z._version != 0:
            raise RuntimeError("the per-device zero sentinel was written in place (a hook on a proxy gradient)")
        if not active(dev):
            _drop(hd)
            return None
        p0 = hd.p0()
        why = call_reason(hd, p0)
    except Exception as exc:
        _drop(hd)
        latch_off("x2lb call checks", exc)
        return None
    if why is None:
        sv = p0.saved_tensors
        h1, nw, nb, mean1, rstd1 = sv[3], sv[4], sv[5], sv[6], sv[7]
        try:
            why = tensor_reason(dh8, w8_t, dh_scale, w_scale, h1, nw, nb, mean1, rstd1, x_dt)
        except Exception as exc:
            _drop(hd)
            latch_off("x2lb call checks", exc)
            return None
    if why is not None:
        _drop(hd)
        _decline(hd, "call", why)
        return None
    try:
        result, grid, st = _RT.fused(_lib, dh8.detach(), w8_t.detach(), dh_scale.detach(), w_scale.detach(),
                                     h1.detach(), mean1.detach(), rstd1.detach(), nw.detach(), nb.detach())
        M = int(h1.shape[0])
        sentinel = _ZERO[_dev_index(dev)].expand(M, HID)
    except Exception as exc:
        _drop(hd)
        latch_off("x2lb call", exc)
        return None
    hd.result, hd.stream, hd.sentinel, hd.state, hd.grid = tuple(result), st, sentinel, "fused", int(grid)
    hd.task = _graph_task_id()
    _PENDING.add(hd)
    FUSED_CALLS[hd.k] = FUSED_CALLS.get(hd.k, 0) + 1
    idx = _dev_index(dev)
    if ("fused", str(idx)) not in _ONCE:
        _once("fused", str(idx),
              f"{ARMED_PREFIX} fused L2 dgrad + SwishLN256 bwd on cuda:{idx} mlp={hd.k} M={M} grid={grid} "
              f"fin={PR7_FIN_NAME if fin_mode() == 'pr7' else 'x2lb_fin (diagnostic)'}; da1 not written; kill "
              f"switch {ENV_VAR}=0")
    return sentinel


def _phase1_ignored_reason(device: torch.device) -> Optional[str]:
    from generative_recommenders.ops import preproc_fp8_l2 as l2
    from generative_recommenders.ops.triton import triton_layer_norm as TLN

    if not l2.enabled():
        return "DLRM_HSTU_FP8_PREPROC_L2 is off or IGNORED"
    if not TLN._wln_bwd_flydsl_enabled():
        return "the FlyDSL ln_bwd (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) is off"
    if device.type != "cuda" or not torch.cuda.is_available():
        return f"device {device} is not a GPU"
    arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
    if arch != _ARCH:
        return f"device arch {arch} is not {_ARCH} (x2lb_pass.cu is {_ARCH}-only)"
    return None


def prebuild_build(device: torch.device) -> bool:
    global _P1_STATE
    if not requested() or _BROKEN is not None or _P1_STATE is not None:
        return False
    t0 = time.perf_counter()
    try:
        mlps()
        fin_mode()
        why = _phase1_ignored_reason(device)
    except Exception as exc:
        _P1_STATE = "failed"
        latch_off("phase 1 (configuration)", exc)
        return False
    if why is not None:
        _P1_STATE = "ignored"
        ignored(why)
        return False
    try:
        lib = load()
        res = lib.meta["resources"][KERNEL_S1]
        isa = lib.meta["isa"]
        co = lib.meta.get("code_object") or {"status": "unknown (not recorded)"}
    except Exception as exc:
        _P1_STATE = "failed"
        latch_off("phase 1 (build / census / ISA gate / ABI)", exc)
        return False
    _P1_STATE = "ok"
    secs = time.perf_counter() - t0
    if co.get("status") == "ok":
        co_txt = (f"code object .text {co['.text'][:12]} "
                  + ("(== reference build)" if co.get("lab") == "identical" else
                     f"(differs from the reference build {LAB_CODE_OBJECT_SHA256['.text'][:12]})"))
    else:
        co_txt = f"code object {co.get('status')}"
    isa_txt = ("full" if isa.get("mode") == "full" else f"pinned .s {PINNED_ASM_NORM_SHA256[:12]}")
    PHASE1_INFO.clear()
    PHASE1_INFO.update(lib=str(lib.path), cached=lib.cached, build_s=lib.build_s, resources=res, isa=isa,
                       code_object=co, seconds=secs, mlps=mlps(), fin=fin_mode())
    _say(f"{ARMED_PREFIX} pre-run_start build OK ({PASS_NAME}<0,0> VGPR {res.get('unified_vgpr')} / {LIM_VGPR}, "
         f"spill {res.get('vgpr_spill')}, scratch {res.get('scratch')}, LDS {res.get('lds')}, occupancy "
         f"{res.get('occupancy')}; isa_check C1-C18 PASS ({isa_txt}); src {SOURCE_SHA256[:12]}; {co_txt}; "
         f"{lib.path.parent.name}/{_LIB_NAME} "
         f"{'cached' if lib.cached else f'built in {lib.build_s:.1f}s'}; mlps {','.join(map(str, mlps()))}; "
         f"fin {fin_mode()}; phase 1 {secs:.1f}s)")
    return True


_M32 = 0xFFFFFFFF
_CH: int = 1 << 16


def _mix32(x: torch.Tensor) -> torch.Tensor:
    x = x & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    return x ^ (x >> 16)


def _hashed(n: int, salt: int, start: int, device: torch.device) -> torch.Tensor:
    return _mix32(torch.arange(start, start + n, dtype=torch.int64, device=device) * 0x9E3779B1 + salt)


def _unit(h: torch.Tensor) -> torch.Tensor:
    return (h & 0xFFFFFF).to(torch.float32) * (1.0 / (1 << 24))


class _Live:
    def __init__(self) -> None:
        self.live: Dict[str, int] = {}
        self.peak = 0

    def alloc(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        t = torch.empty(shape, dtype=dtype, device=device)
        self.add(name, t)
        return t

    def add(self, name: str, t: torch.Tensor) -> None:
        self.live[name] = t.numel() * t.element_size()
        self.peak = max(self.peak, sum(self.live.values()))

    def free(self, *names: str) -> None:
        for n in names:
            self.live.pop(n, None)


_BF16_CANARY: int = 0x7FA5
_F32_CANARY: int = 0x7FC0A5A5


def _canary_bf16(live: _Live, name: str, M: int, device: torch.device) -> torch.Tensor:
    t = live.alloc(name, (M + CANARY_ROWS, HID), torch.bfloat16, device)
    t.view(torch.int16).fill_(_BF16_CANARY)
    return t


def _canary_f32(live: _Live, name: str, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    t = live.alloc(name, shape, torch.float32, device)
    t.view(torch.int32).fill_(_F32_CANARY)
    return t


def cancel_data(M: int, device: torch.device, live: _Live, salt: int = 0x2B1D) -> Dict[str, torch.Tensor]:
    dh8 = live.alloc("dh8", (M, KD), torch.float8_e5m2, device)
    h1 = live.alloc("h1", (M, HID), torch.bfloat16, device)
    mean1 = live.alloc("mean1", (M,), torch.float32, device)
    rstd1 = live.alloc("rstd1", (M,), torch.float32, device)
    u8 = dh8.view(torch.uint8)
    for s in range(0, M, _CH):
        e = min(M, s + _CH)
        n = e - s
        hb = _hashed(n * 128, salt + 1, s * 128, device).view(n, 128)
        big = (0x50 + hb % 43) | (((hb >> 16) & 1) << 7)
        u8[s:e, 0:128] = big.to(torch.uint8)
        u8[s:e, 128:256] = (big ^ 0x80).to(torch.uint8)
        del hb, big
        hs = _hashed(n * 256, salt + 2, s * 256, device).view(n, 256)
        u8[s:e, 256:512] = ((0x01 + hs % 47) | (((hs >> 16) & 1) << 7)).to(torch.uint8)
        hr = _hashed(n * 256, salt + 3, s * 256, device).view(n, 256)
        rr = _hashed(n, salt + 4, s, device)
        sc = 0.5 + 1.5 * _unit(rr)
        off = _unit(rr >> 8) - 0.5
        hb16 = ((_unit(hr) - 0.5) * 3.4641016 * sc[:, None] + off[:, None]).to(torch.bfloat16)
        h1[s:e] = hb16
        hd = hb16.double()
        mean1[s:e] = hd.mean(1).float()
        rstd1[s:e] = (1.0 / torch.sqrt(hd.var(1, unbiased=False) + 1e-5)).float()
        del hs, hr, rr, sc, off, hb16, hd
    hw = _hashed(HID * KD, salt + 5, 0, device).view(HID, KD)
    w = (0x20 + hw % 0x58) | (((hw >> 16) & 1) << 7)
    w = torch.where(((hw >> 20) & 15) == 0, torch.zeros_like(w), w).to(torch.uint8)
    w[:, 128:256] = w[:, 0:128]
    w8t = w.contiguous().view(torch.float8_e4m3fn)
    hn = _hashed(2 * HID, salt + 6, 0, device)
    nw = (1.0 + 0.2 * (2.0 * _unit(hn[:HID]) - 1.0)).float().contiguous()
    nb = (0.3 * (2.0 * _unit(hn[HID:]) - 1.0)).float().contiguous()
    sa = torch.full((1,), 2.0 ** -20, dtype=torch.float32, device=device)
    sb = torch.full((1,), 2.0 ** -10, dtype=torch.float32, device=device)
    return dict(dh8=dh8, w8t=w8t, sa=sa, sb=sb, h1=h1, mean1=mean1, rstd1=rstd1, nw=nw, nb=nb)


def _diff_rows(a: torch.Tensor, b: torch.Tensor, lo: int, hi: int) -> Tuple[int, Optional[int]]:
    n, first = 0, None
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    for s in range(lo, hi, _CH):
        e = min(hi, s + _CH)
        d = (ai[s:e] != bi[s:e]).any(dim=1)
        c = int(d.sum().item())
        if c and first is None:
            first = s + int(torch.nonzero(d)[0].item())
        n += c
    return n, first


def _n_eq_bits(t: torch.Tensor, bits: int) -> int:
    v = t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)
    return int((v == bits).sum().item())


def _canaries_ok(t: torch.Tensor, M: int, what: str) -> None:
    rows = t[M:]
    bad = int((rows.view(torch.int16) != _BF16_CANARY).sum().item())
    if bad:
        raise RuntimeError(f"{what}: {bad} canary words in rows >= M {M} were written")


def _written(t: torch.Tensor, what: str) -> None:
    left = _n_eq_bits(t, _F32_CANARY)
    if left:
        raise RuntimeError(f"{what}: {left} of {t.numel()} elements never written (sentinel left)")


def _gamma(h: int) -> float:
    u = 2.0 ** -24
    return h * u / (1.0 - h * u)


def finals_bound_check(d: Dict[str, torch.Tensor], da1: torch.Tensor, finals: Tuple[torch.Tensor, ...],
                       grid: int, M: int) -> float:


    D = HID
    dy = da1[:M].double()
    x = d["h1"][:M].double()
    mean, rstd = d["mean1"][:M].double()[:, None], d["rstd1"][:M].double()[:, None]
    w, b = d["nw"].double()[None, :], d["nb"].double()[None, :]
    xhat = (x - mean) * rstd
    s = torch.sigmoid(xhat * w + b)
    t = dy * x * s * (1.0 - s)
    xu = w * t
    c1 = (xhat * xu).sum(1, keepdim=True) / D
    c2 = xu.sum(1, keepdim=True) / D
    dx = (xu - (xhat * c1 + c2)) * rstd + dy * s
    e_t = (dy * x * s).abs()
    xhat_env = xhat.abs() + rstd * (x.abs() + mean.abs())
    xu_env = w.abs() * e_t
    a_r = (xhat_env * xu_env).sum(1, keepdim=True) / D
    b_r = xu_env.sum(1, keepdim=True) / D
    refs = ((t * xhat).sum(0), t.sum(0), dx.sum(0))
    mags = ((e_t * xhat_env).sum(0), e_t.sum(0),
            (rstd * (xu_env + xhat_env * a_r + b_r) + (dy * s).abs()).sum(0))
    ntiles = (M + BM - 1) // BM
    h = 4 * ((ntiles + grid - 1) // grid) + 4 + (grid - 1)
    g = _gamma(h) + 2.0 ** -18
    worst = 0.0
    for name, out, ref, mag in zip(("dW", "dB", "dXS"), finals, refs, mags):
        o = out.double()
        if not bool(torch.isfinite(o).all().item()):
            raise RuntimeError(f"finals {name}: non-finite on finite data")
        bound = g * mag + 1e-300
        r = float(((o - ref).abs() / bound).max().item())
        if r > 1.0:
            raise RuntimeError(f"finals {name} outside (gamma_{h} + 2^-18) x sum|term| (use {r:.3g}) at M {M}")
        worst = max(worst, r)
    return worst


class HeadroomError(RuntimeError):
    pass


def _headroom(device: torch.device, need: int, M: int) -> None:
    free, _ = torch.cuda.mem_get_info(device)
    free += int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device))
    if free < need:
        raise HeadroomError(f"phase-2 headroom: {free / 2**30:.2f} GiB free < 5U + 1 GiB = {need / 2**30:.2f} GiB "
                          f"at M {M}")


def check_point(lib: "_Lib", M: int, rule: str, device: torch.device, live: _Live) -> Dict[str, Any]:
    from generative_recommenders.ops.triton import triton_fp8 as tf8

    rt = _RT
    st = torch.cuda.current_stream(device)
    G = lib.plan(M, rt.num_cu(device))
    out: Dict[str, Any] = dict(M=M, grid=G, rule=rule)
    d = cancel_data(M, device, live)

    da1_t = tf8._scaled_mm(d["dh8"], d["w8t"].t(), d["sa"], d["sb"], out_dtype=torch.bfloat16)
    live.add("da1_t", da1_t)
    out["torch_mm"] = "triton_fp8._scaled_mm"

    da1_x = _canary_bf16(live, "da1_x", M, device)
    dh1_x = _canary_bf16(live, "dh1_x", M, device)
    part_x = _canary_f32(live, "part_x", (3, G, HID), device)
    rt.launch(lib, "emit", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"], d["nb"],
              dh1_x, da1_x, part_x, M, G, st)
    lim = M if rule == "all" else max(0, M - SPLIT_TAIL) if rule == "dp_region" else 0
    bad, first = _diff_rows(da1_x, da1_t, 0, lim)
    if bad:
        raise RuntimeError(f"(a) S1+DA1 da1 != hipBLASLt (triton_fp8._scaled_mm) at M {M}: {bad} of {lim} gated rows "
                           f"differ (first {first})")
    out["da1_gated_rows"] = lim
    out["da1_reported_diff_rows"] = _diff_rows(da1_x, da1_t, lim, M)[0]
    del da1_t
    live.free("da1_t")
    _canaries_ok(da1_x, M, "(e) da1_x")
    _canaries_ok(dh1_x, M, "(e) dh1_x")
    _written(part_x, "(e) S1+DA1 PART")

    dx_p, dw_p, db_p, dxs_p = _LB.ln_bwd(da1_x[:M], d["h1"], d["nw"], d["nb"], d["mean1"], d["rstd1"],
                                         is_swish=True, sum_dx=True)
    live.add("dx_p", dx_p)
    bad, first = _diff_rows(dh1_x, dx_p, 0, M)
    if bad:
        raise RuntimeError(f"(b) S1+DA1 dh1 != FlyDSL ln_bwd(da1) at M {M}: {bad} rows differ (first {first})")
    if rule == "report":
        fin_x = rt.fin(lib, fin_mode(), part_x, G, st)
        out["finals_use"] = finals_bound_check(d, da1_x, fin_x, G, M)
        out["finals_vs_pr7_max_abs"] = [float((a.double() - b.double()).abs().max().item())
                                        for a, b in zip(fin_x, (dw_p, db_p, dxs_p))]
        del fin_x
    del dx_p, dw_p, db_p, dxs_p, da1_x
    live.free("dx_p", "da1_x")

    dh1_s = _canary_bf16(live, "dh1_s", M, device)
    part_s = _canary_f32(live, "part_s", (3, G, HID), device)
    rt.launch(lib, "s1", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"], d["nb"],
              dh1_s, None, part_s, M, G, st)
    bad, first = _diff_rows(dh1_s, dh1_x, 0, M)
    if bad:
        raise RuntimeError(f"(c) S1 dh1 != S1+DA1 dh1 at M {M}: {bad} rows differ (first {first})")
    nbad = int((part_s.view(torch.int32) != part_x.view(torch.int32)).sum().item())
    if nbad:
        raise RuntimeError(f"(c) S1 PART != S1+DA1 PART at M {M}: {nbad} words differ")
    _canaries_ok(dh1_s, M, "(e) dh1_s")
    _written(part_s, "(e) S1 PART")

    if fin_mode() == "pr7":
        fins = rt.fin(lib, "pr7", part_s, G, st, out=tuple(_canary_f32(live, f"fin{i}", (HID,), device)
                                                            for i in range(3)))
    else:
        fins = rt.fin(lib, "x2lb", part_s, G, st, out=_canary_f32(live, "fin0", (3, HID), device))
    for nm, t in zip(("dW", "dB", "dXS"), fins):
        _written(t, f"(e) finals {nm}")
    del dh1_x, part_x, fins
    live.free("dh1_x", "part_x", "fin0", "fin1", "fin2")

    dh1_7 = _canary_bf16(live, "dh1_7", M, device)
    part_7 = _canary_f32(live, "part_7", (3, STRESS_GRID, HID), device)
    rt.launch(lib, "s1", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"], d["nb"],
              dh1_7, None, part_7, M, STRESS_GRID, st)
    bad, first = _diff_rows(dh1_7, dh1_s, 0, M)
    if bad:
        raise RuntimeError(f"(d) S1 grid {STRESS_GRID} dh1 != grid {G} dh1 at M {M}: {bad} rows differ "
                           f"(first {first})")
    _canaries_ok(dh1_7, M, "(e) dh1_7")
    _written(part_7, "(e) grid-7 PART")
    del dh1_7, part_7, dh1_s, part_s, d
    live.free(*list(live.live))
    return out


def prebuild_check(device: torch.device) -> str:
    global _LB, _P2_STATE
    if not requested():
        return ""
    if _BROKEN is not None:
        return "x2lb=off"
    if _P1_STATE != "ok" or _lib is None:
        phase2_not_run(f"phase 1 did not pass ({_P1_STATE or 'not run'})")
        return "x2lb=off"
    from generative_recommenders.ops.triton import triton_layer_norm as TLN

    if not TLN._wln_bwd_flydsl_enabled() or TLN._WLN_BWD_FLYDSL_BROKEN is not None:
        phase2_not_run("the FlyDSL ln_bwd (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) off or latched off")
        return "x2lb=off"
    t0 = time.perf_counter()
    live = _Live()
    points: List[Dict[str, Any]] = []
    try:
        if device.type != "cuda":
            raise RuntimeError(f"device {device} is not a GPU")
        with torch.cuda.device(device):
            from generative_recommenders.ops.flydsl import ln_bwd as lb

            _LB = lb
            for M, rule in CHECK_POINTS:
                _headroom(device, 5 * M * KD + HEADROOM_SLACK, M)
                points.append(check_point(_lib, M, rule, device, live))
            idx = _dev_index(device)
            _ZERO[idx] = _RT.make_zero(device)
            torch.cuda.synchronize(device)
    except HeadroomError as exc:
        _P2_STATE = "failed"
        latch_off("phase-2 headroom", exc)
        return "x2lb=error"
    except Exception as exc:
        _P2_STATE = "failed"
        latch_off("phase-2 self-check", exc)
        return "x2lb=error"
    _armed.add(idx)
    _P2_STATE = "ok"
    secs = time.perf_counter() - t0
    PREBUILD_INFO[idx] = dict(points=points, peak_bytes=live.peak, seconds=secs)
    big = [p for p in points if p["rule"] != "report"]
    fin = next(p for p in points if p["rule"] == "report")
    dp = ", ".join(f"M {p['M']} grid {p['grid']} beyond {p['da1_reported_diff_rows']} rows differ"
                   for p in big if p["rule"] == "dp_region")
    _say(f"{ARMED_PREFIX} pre-run_start self-check OK (cancel: M {M_DP} da1 == hipBLASLt all rows; M {M_A} / {M_B} "
         f"da1 == hipBLASLt on rows < M-{SPLIT_TAIL} ({dp}); dh1 == ln_bwd(da1) all rows; S1 == S1+DA1; grids "
         f"{'/'.join(str(p['grid']) for p in big)}/{STRESS_GRID}; canaries; partials within gamma at M {M_FIN} "
         f"(use {fin['finals_use']:.3g}, fin {fin_mode()}); peak transient {live.peak / 1e9:.2f} GB; phase 2 "
         f"{secs:.1f}s; device=cuda:{idx})")
    return "x2lb=1"


def reset_for_testing() -> None:
    global _REQUESTED, _MLPS, _FIN, _BROKEN, _lib, _RT, _P1_STATE, _P2_STATE, _P2_SKIP, _LB
    _REQUESTED = _MLPS = _FIN = _BROKEN = None
    _lib = None
    _RT = _CudaRuntime()
    _P1_STATE = _P2_STATE = _P2_SKIP = None
    _LB = None
    _armed.clear()
    _ONCE.clear()
    _ZERO.clear()
    _PENDING.clear()
    PHASE1_INFO.clear()
    PREBUILD_INFO.clear()
    FUSED_CALLS.clear()
    DECLINES.clear()
