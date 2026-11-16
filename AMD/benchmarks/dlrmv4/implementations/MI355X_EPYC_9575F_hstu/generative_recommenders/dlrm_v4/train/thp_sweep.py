# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import os
import time
from typing import Dict, List, Optional, Set, Tuple

ENV = "DLRM_HSTU_FP8_THP_NOHUGE_SWEEP"
MADV_NOHUGEPAGE = 15
_ON = ("1", "true", "on", "yes")
_THP = "/sys/kernel/mm/transparent_hugepage"
_MODES = ("always", "madvise", "never")


Vma = Tuple[int, int, str, int, str, int, int, Tuple[str, ...]]


def _say(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() in _ON


def _is_header(parts: List[str]) -> bool:
    if len(parts) < 5 or "-" not in parts[0] or parts[0].endswith(":"):
        return False
    perms = parts[1]
    return len(perms) == 4 and perms[3] in "ps"


def parse_smaps(text_lines) -> List[Vma]:
    out: List[Vma] = []
    cur: Optional[List[object]] = None

    def _flush() -> None:
        if cur is not None:
            out.append(
                (int(cur[0]), int(cur[1]), str(cur[2]), int(cur[3]), str(cur[4]),
                 int(cur[5]), int(cur[6]), tuple(cur[7]))
            )

    for line in text_lines:
        parts = line.split()
        if not parts:
            continue
        if _is_header(parts):
            _flush()
            lo_s, hi_s = parts[0].split("-", 1)
            path = " ".join(parts[5:]) if len(parts) > 5 else ""
            try:
                inode = int(parts[4])
            except ValueError:
                inode = -1
            cur = [int(lo_s, 16), int(hi_s, 16), parts[1], inode, path, 0, 0, ()]
            continue
        if cur is None:
            continue
        key = parts[0]
        if key == "Rss:" and len(parts) >= 2:
            cur[5] = int(parts[1])
        elif key == "AnonHugePages:" and len(parts) >= 2:
            cur[6] = int(parts[1])
        elif key == "VmFlags:":
            cur[7] = tuple(parts[1:])
    _flush()
    return out


def _anon_private(v: Vma) -> bool:
    _lo, _hi, perms, inode, path, _rss, _ahp, _fl = v
    if perms[3] != "p" or inode != 0:
        return False
    return path == "" or path == "[heap]" or path.startswith("[anon:")


def select(vmas: List[Vma], thp_mode: str = "madvise", pmd: int = 2 * 1024 * 1024
           ) -> Tuple[List[Vma], Dict[str, int]]:


    sel: List[Vma] = []
    c: Dict[str, int] = {
        "n_vma": len(vmas), "sel_n": 0, "sel_kb": 0, "sel_4k_kb": 0, "sel_thp_kb": 0,
        "keep_thp_n": 0, "keep_thp_kb": 0, "keep_other_n": 0, "keep_other_kb": 0,
    }
    if thp_mode == "never":
        return sel, c
    for v in vmas:
        lo, hi, _perms, _inode, _path, rss, ahp, flags = v
        if "hg" not in flags:
            continue
        if (hi & ~(pmd - 1)) - ((lo + pmd - 1) & ~(pmd - 1)) < pmd:
            continue
        size_kb = (hi - lo) // 1024
        if not _anon_private(v):
            c["keep_other_n"] += 1
            c["keep_other_kb"] += size_kb
            continue
        four_k = max(0, rss - ahp)
        if four_k > 0:
            sel.append(v)
            c["sel_n"] += 1
            c["sel_kb"] += size_kb
            c["sel_4k_kb"] += four_k
            c["sel_thp_kb"] += ahp
        else:
            c["keep_thp_n"] += 1
            c["keep_thp_kb"] += size_kb
    return sel, c


def _read_word(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            txt = f.read()
        i, j = txt.index("["), txt.index("]")
        return txt[i + 1:j].strip()
    except (OSError, ValueError):
        return None


def pmd_size(thp_dir: str = _THP) -> int:
    try:
        with open(os.path.join(thp_dir, "hpage_pmd_size")) as f:
            v = int(f.read().strip())
        return v if v > 0 and (v & (v - 1)) == 0 else 2 * 1024 * 1024
    except (OSError, ValueError):
        return 2 * 1024 * 1024


def thp_mode(thp_dir: str = _THP, pmd: int = 2 * 1024 * 1024) -> str:
    per = _read_word(os.path.join(thp_dir, "hugepages-%dkB" % (pmd // 1024), "enabled"))
    if per in _MODES:
        return per
    top = _read_word(os.path.join(thp_dir, "enabled"))
    return top if top in _MODES else "unknown"


def _live_ranges(maps_path: str = "/proc/self/maps") -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    with open(maps_path, "r", errors="replace") as f:
        for line in f:
            r = line.split(" ", 1)[0]
            lo_s, sep, hi_s = r.partition("-")
            if sep:
                out.add((int(lo_s, 16), int(hi_s, 16)))
    return out


def _libc() -> ctypes.CDLL:
    lib = ctypes.CDLL(None, use_errno=True)
    lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.madvise.restype = ctypes.c_int
    return lib


def sweep(rank: int = -1, smaps_path: str = "/proc/self/smaps", maps_path: str = "/proc/self/maps",
          thp_dir: str = _THP, mode: Optional[str] = None) -> Dict[str, object]:


    boot0 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    t0 = time.monotonic()
    pmd = pmd_size(thp_dir)
    m = thp_mode(thp_dir, pmd) if mode is None else mode
    with open(smaps_path, "r", errors="replace") as f:
        vmas = parse_smaps(f)
    t_read = time.monotonic() - t0
    sel, c = select(vmas, m, pmd)
    live = _live_ranges(maps_path) if sel else set()
    todo = [v for v in sel if (v[0], v[1]) in live]
    lib = _libc()
    errs: Dict[int, int] = {}
    for lo, hi, *_rest in todo:
        if lib.madvise(ctypes.c_void_p(lo), hi - lo, MADV_NOHUGEPAGE) != 0:
            e = ctypes.get_errno() or -1
            errs[e] = errs.get(e, 0) + 1
    res: Dict[str, object] = dict(c)
    res.update(
        att_kb=sum((v[1] - v[0]) // 1024 for v in todo), att_4k_kb=sum(max(0, v[5] - v[6]) for v in todo),
        att_thp_kb=sum(v[6] for v in todo),
        rank=rank, stale=len(sel) - len(todo), attempted=len(todo), errors=sum(errs.values()),
        errnos=dict(errs), t_read=t_read, t_total=time.monotonic() - t0, mode=m, pmd=pmd,
        pid=os.getpid(), boot_ns0=boot0, boot_ns1=time.clock_gettime_ns(time.CLOCK_BOOTTIME),
    )
    return res


def _gib(kb: object) -> float:
    return float(kb) / (1024.0 * 1024.0)


def format_line(r: Dict[str, object]) -> str:
    head = "armed OK" if not r["errors"] else "PARTIAL"
    return (
        "%s: [thp-sweep] rank=%d before run_start: MADV_NOHUGEPAGE on %d VMA(s) %.2f GiB "
        "(4K-mapped %.2f GiB, THP-mapped %.2f GiB); kept hg: %d THP-only VMA(s) %.2f GiB, "
        "%d shared/file VMA(s) %.2f GiB; stale %d; errors %d%s; smaps %d VMAs read in %.3fs, total %.3fs; "
        "thp=%s pid=%d boot_ns0=%d boot_ns1=%d"
        % (
            head, r["rank"], r["attempted"], _gib(r["att_kb"]), _gib(r["att_4k_kb"]), _gib(r["att_thp_kb"]),
            r["keep_thp_n"], _gib(r["keep_thp_kb"]), r["keep_other_n"], _gib(r["keep_other_kb"]),
            r["stale"], r["errors"], (" errnos=%s" % r["errnos"]) if r["errors"] else "",
            r["n_vma"], r["t_read"], r["t_total"], r["mode"], r["pid"], r["boot_ns0"], r["boot_ns1"],
        )
    )


def sweep_from_env(rank: int = -1) -> Optional[Dict[str, object]]:
    try:
        if not enabled():
            return None
        r = sweep(rank=rank)
        _say(format_line(r))
        return r
    except Exception as e:
        try:
            _say("NOT armed: [thp-sweep] rank=%s: %s: %s" % (rank, type(e).__name__, e))
        except Exception:
            pass
        return None
