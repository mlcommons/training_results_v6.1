#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate generative_recommenders/dlrm_v4/train/hip_preload_table.jsonl (offline, CPU only).

The table lists, for every statically registered HIP fat-binary bundle (one per translation unit, "TU") whose kernels
a production training step launches, ONE host kernel stub that belongs to that bundle. At run time
(dlrm_v4/train/hip_preload.py, DLRM_HSTU_FP8_HIP_PRELOAD=1, the yambda_5b.gin default since S14-d v2)
hipFuncGetAttributes on that stub makes the HIP runtime
digest the bundle and load its gfx950 code object on the current device -- the work HIP's deferred loading
(ROCm 7.2.1 default, HIP_ENABLE_DEFERRED_LOADING unset) would otherwise do inside the first launch of any kernel of
that TU, i.e. inside training step 1.

Inputs:
  --lib PATH        a library with a .hip_fatbin section (libtorch_hip.so, fbgemm_gpu*.so); repeatable
  --trace SET=GLOB  Kineto chrome traces of production steps; SET labels the run shape (1n, 2n, 4n); repeatable
  --out PATH        output .jsonl (first line kind=header, then one kind=entry row per bundle)

Method:
  1. Walk .hip_fatbin bundle by bundle (CCOB v2/v3 zstd-compressed or plain __CLANG_OFFLOAD_BUNDLE__), extract the
     gfx950 code object and its kernel descriptor symbols '<mangled>.kd'; demangle with c++filt (the trace names
     kernels as '<demangled> [clone .kd]').
  2. HIP registers a host stub once; the first registration wins and module constructors run in bundle order, so a
     kernel name belongs to the FIRST bundle (of its library) that contains it.
  3. Kernel names launched in the traces -> (library, first bundle). A demangled name present in several libraries
     (rocprim templates, host stubs not exported) is attributed by the launching op (fbgemm::* -> fbgemm libraries,
     else libtorch_hip); what stays ambiguous is kept as a union (an extra preload costs untimed init only).
  4. Per touched bundle pick a kernel whose first bundle is this one and whose host stub is a symbol of the library:
     prefer .dynsym (GLOBAL/WEAK, default/protected visibility; resolved with dlsym at run time, mode "dlsym"),
     else .symtab (local stubs, e.g. anonymous-namespace kernels; mode "symtab": load base + st_value, valid only
     when the loaded library's GNU build-id equals the one recorded in the header; the base comes from a dlsym'd
     anchor symbol and is cross-checked against /proc/self/maps). Bundles without any such symbol (stripped
     libraries) are listed in header.uncovered.
No sample data, no binaries: kernel/symbol names, sizes and build ids only.

S14-d v2 amendment (2026-10-11): the checked-in table also carries rows that NO steady-step trace shows. They are listed
in its header "amendments" (source: rccl iter85 S14D85 section 6; rows file sha256 51b052db...): libtorch_hip b10 and
b144 widened from 1n to every shape, and 7 symtab rows b2, b49, b55, b73, b74, b75, b77, the step-1-only first-use code
objects of once-per-process checks and self-tests (valid for libtorch_hip build-id 9889ac94... only). This script does
not produce them: regenerating from the steady traces alone (steps 42-46 / 52-56) DROPS those 7 rows and narrows b10 /
b144 back to 1n. After any regeneration, re-apply the amendment (append the rows file byte-identically, widen the two
rows, recompute n_entries / n_symtab / entries_by_set, keep the "amendments" record) or include step-1 (first-use)
traces in each shape's --trace glob. The 7 rows hold st_values of that one build: after a libtorch_hip rebuild re-derive
them (header "amendments" records rows_valid_for_build_id). Never re-append them to a table generated for another
build: the run-time build-id gate is per library (header), so stale values would pass it and only fail open as HIP
errors (no registered stub at that address).
"""
import argparse
import collections
import glob
import gzip
import json
import multiprocessing
import os
import re
import shutil
import struct
import subprocess
import sys
import time

GFX = "gfx950"
# First-use cost model (dlstart iter 34, ledger lazyload-bundle-cost-model; node, fresh process, untraced):
#   libtorch_hip (compressed, multi-target): ms = -2.05 + 0.980 * uncompressed MiB
#   fbgemm (uncompressed, gfx950 only):      ms = 3.79 * gfx950 code-object MiB
COST_LIBTORCH = (-2.05, 0.980)
COST_FBGEMM = 3.79


# ----------------------------------------------------------------------------------------------- ELF helpers
def _sections(elf):
    e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    sh = []
    for i in range(e_shnum):
        sh.append(struct.unpack_from("<IIQQQQIIQQ", elf, e_shoff + i * e_shentsize))
    names_off = sh[e_shstrndx][4] if e_shstrndx < len(sh) else 0
    out = []
    for (n, typ, fl, addr, off, size, link, info, al, ent) in sh:
        e = elf.index(b"\0", names_off + n)
        out.append({"name": elf[names_off + n:e].decode(errors="replace"), "type": typ, "addr": addr, "off": off,
                    "size": size, "link": link, "ent": ent})
    return out


def _symbols(elf, secs, sh_type):
    """{name: (st_value, bind, vis, type)} of defined, nonzero symbols of SHT_SYMTAB(2) / SHT_DYNSYM(11)."""
    out = {}
    for s in secs:
        if s["type"] != sh_type or not s["ent"]:
            continue
        stoff = secs[s["link"]]["off"]
        for k in range(s["size"] // s["ent"]):
            st_name, st_info, st_other, st_shndx, st_value, _sz = struct.unpack_from(
                "<IBBHQQ", elf, s["off"] + k * s["ent"])
            if not st_value or not st_shndx:
                continue
            e = elf.index(b"\0", stoff + st_name)
            nm = elf[stoff + st_name:e].decode(errors="replace")
            out.setdefault(nm, (st_value, st_info >> 4, st_other & 3, st_info & 0xF))
    return out


def build_id(elf, secs):
    for s in secs:
        if s["type"] != 7:  # SHT_NOTE
            continue
        q, end = s["off"], s["off"] + s["size"]
        while q + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, q)
            name = elf[q + 12:q + 12 + namesz]
            dq = q + 12 + ((namesz + 3) & ~3)
            if ntype == 3 and name.rstrip(b"\0") == b"GNU":
                return elf[dq:dq + descsz].hex()
            q = dq + ((descsz + 3) & ~3)
    return None


def kd_symbols(co):
    if co[:4] != b"\x7fELF":
        return []
    secs = _sections(co)
    names = set()
    for t in (2, 11):
        for nm in _symbols(co, secs, t):
            if nm.endswith(".kd"):
                names.add(nm)
    return sorted(names)


# ----------------------------------------------------------------------------------------------- fat binary walk
def _zstd_d(payload):
    try:
        import zstandard  # type: ignore

        return zstandard.ZstdDecompressor().decompress(payload, max_output_size=1 << 34)
    except ImportError:
        p = subprocess.run(["zstd", "-d", "-c", "-q"], input=payload, capture_output=True, check=True)
        return p.stdout


def _parse_offload_bundle(ub):
    assert ub[:24] == b"__CLANG_OFFLOAD_BUNDLE__", ub[:24]
    n = struct.unpack_from("<Q", ub, 24)[0]
    q, out, end = 32, [], 32
    for _ in range(n):
        eoff, esz, tl = struct.unpack_from("<QQQ", ub, q)
        q += 24
        out.append((ub[q:q + tl].decode(errors="replace"), eoff, esz))
        q += tl
        end = max(end, eoff + esz, q)
    return out, end


MAGIC = re.compile(rb"CCOB|__CLANG_OFFLOAD_BUNDLE__")


def walk_fatbin(lib):
    """[{index, offset, kind, compressed, uncompressed, co_bytes, kernels_mangled}] in section order. Sequential walk:
    the next bundle is searched only after the end of the current one (no false magic inside payloads)."""
    with open(lib, "rb") as f:
        elf = f.read()
    secs = _sections(elf)
    fb = [s for s in secs if s["name"] == ".hip_fatbin"]
    if not fb:
        return elf, secs, []
    sec = elf[fb[0]["off"]:fb[0]["off"] + fb[0]["size"]]
    rows, pos = [], 0
    while True:
        m = MAGIC.search(sec, pos)
        if not m:
            break
        o = m.start()
        r = {"index": len(rows), "offset": o}
        if sec[o:o + 4] == b"CCOB":
            ver, _meth = struct.unpack_from("<HH", sec, o + 4)
            if ver == 3:
                tot, unc, _h = struct.unpack_from("<QQQ", sec, o + 8)
                hdr = 32
            elif ver == 2:
                tot, unc, _h = struct.unpack_from("<IIQ", sec, o + 8)
                hdr = 24
            else:
                raise SystemExit(f"{lib}: unsupported CCOB version {ver} at {o}")
            ub = _zstd_d(sec[o + hdr:o + tot])
            r.update(kind="ccob", compressed=tot, uncompressed=unc)
            pos = o + tot
        else:
            ents, end = _parse_offload_bundle(sec[o:])
            ub = sec[o:o + end]
            r.update(kind="plain", compressed=None, uncompressed=end)
            pos = o + end
        ents, _ = _parse_offload_bundle(ub)
        co = [(eo, es) for t, eo, es in ents if GFX in t]
        r["co_bytes"] = co[0][1] if co else 0
        r["kernels_mangled"] = kd_symbols(ub[co[0][0]:co[0][0] + co[0][1]]) if co else []
        rows.append(r)
    return elf, secs, rows


def demangle(names):
    if not names:
        return []
    p = subprocess.run(["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True, check=True)
    out = p.stdout.splitlines()
    assert len(out) == len(names), (len(out), len(names))
    return out


# ----------------------------------------------------------------------------------------------- traces
def _trace_kernels(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    cpu = {}
    for e in ev:
        if e.get("cat") == "cpu_op":
            x = (e.get("args") or {}).get("External id")
            if x is not None:
                cpu[x] = e.get("name", "")
    out = {}
    for e in ev:
        if e.get("cat") == "kernel":
            out.setdefault(e["name"], set()).add(cpu.get((e.get("args") or {}).get("External id"), ""))
    return path, {k: sorted(v) for k, v in out.items()}


def candidates(name, ctxs, libs):
    """(lib, first bundle) pairs a launched kernel name can load. A demangled name found in several libraries (rocprim
    templates; their host stubs are not exported, so no cross-library interposition) is attributed by the launching
    op: fbgemm::* -> the fbgemm libraries, any other op -> the non-fbgemm libraries (libtorch_hip); no usable context
    -> every candidate. Ambiguity left inside one group is kept as a union (an extra preload is untimed init)."""
    cands = [(lib, L["name_first"][name]) for lib, L in libs.items() if name in L["name_first"]]
    if len(cands) <= 1:
        return cands
    out = set()
    for ctx in ctxs or [""]:
        if ctx.startswith("fbgemm::"):
            sel = [c for c in cands if c[0].startswith("fbgemm")]
        elif ctx:
            sel = [c for c in cands if not c[0].startswith("fbgemm")]
        else:
            sel = []
        out.update(sel or cands)
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True)
    ap.add_argument("--trace", action="append", required=True, help="SET=GLOB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--summary", default=None, help="optional json with the per-set bundle census")
    a = ap.parse_args(argv)
    t0 = time.time()

    # libraries
    libs = {}
    for lib in a.lib:
        base = os.path.basename(lib)
        elf, secs, rows = walk_fatbin(lib)
        dyn = _symbols(elf, secs, 11)
        tab = _symbols(elf, secs, 2)
        flat = [m for r in rows for m in r["kernels_mangled"]]
        dem = dict(zip(sorted(set(flat)), demangle(sorted(set(flat))))) if flat else {}
        first = {}
        for r in rows:
            for m in r["kernels_mangled"]:
                first.setdefault(m, r["index"])
        name_first = {}  # demangled '... [clone .kd]' -> first bundle
        for r in rows:
            for m in r["kernels_mangled"]:
                name_first.setdefault(dem[m], r["index"])
        # anchor: a defined GLOBAL FUNC in .dynsym (for the symtab-mode base); deterministic choice
        anchor = None
        for nm in sorted(dyn):
            v, bind, vis, typ = dyn[nm]
            if bind == 1 and vis in (0, 3) and typ == 2:
                anchor = {"sym": nm, "value": v}
                break
        libs[base] = {"path": lib, "rows": rows, "dyn": dyn, "tab": tab, "first": first, "name_first": name_first,
                      "build_id": build_id(elf, secs), "size": len(elf), "anchor": anchor,
                      "has_symtab": any(s["type"] == 2 for s in secs)}
        print(f"{base}: {len(rows)} bundles, {sum(r['co_bytes'] for r in rows) / 2**20:.1f} MiB {GFX} code, "
              f"{len(flat)} kernels, dynsym {len(dyn)}, symtab {len(tab)}, build-id {libs[base]['build_id']}",
              flush=True)
        del elf

    # traces
    sets = collections.OrderedDict()
    files = []
    for spec in a.trace:
        label, pat = spec.split("=", 1)
        fs = sorted(glob.glob(pat))
        if not fs:
            raise SystemExit(f"no trace matches {pat}")
        sets[label] = fs
        files += fs
    with multiprocessing.Pool(a.jobs) as pool:
        per_file = dict(pool.map(_trace_kernels, sorted(set(files))))
    touched = collections.defaultdict(lambda: {"sets": set(), "names": set()})
    unmapped = collections.Counter()
    for label, fs in sets.items():
        for f in fs:
            for name, ctxs in per_file[f].items():
                cands = candidates(name, ctxs, libs)
                if not cands:
                    unmapped[name] += 1
                    continue
                for key in cands:
                    touched[key]["sets"].add(label)
                    touched[key]["names"].add(name)

    entries, uncovered = [], []
    for (lib, b), t in sorted(touched.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        L = libs[lib]
        r = L["rows"][b]
        own = [m[:-3] if m.endswith(".kd") else m for m in r["kernels_mangled"] if L["first"].get(m) == b]
        pick = None
        for m in own:
            if m in L["dyn"] and L["dyn"][m][1] in (1, 2) and L["dyn"][m][2] in (0, 3):
                pick = {"sym": m, "mode": "dlsym"}
                break
        if pick is None and L["anchor"] is not None:
            for m in own:
                if m in L["tab"]:
                    pick = {"sym": m, "mode": "symtab", "value": L["tab"][m][0]}
                    break
        unc, co = r["uncompressed"] or 0, r["co_bytes"] or 0
        if lib.startswith("libtorch"):
            est = max(0.0, COST_LIBTORCH[0] + COST_LIBTORCH[1] * unc / 2**20)
        else:
            est = COST_FBGEMM * co / 2**20
        meta = {"lib": lib, "bundle": b, "unc_mib": round(unc / 2**20, 3), "co_mib": round(co / 2**20, 3),
                "est_ms": round(est, 1), "sets": sorted(t["sets"]), "example": sorted(t["names"])[0][:160]}
        if pick is None:
            uncovered.append(meta | {"reason": "no host symbol (stripped library or no kernel registered first here)"})
            continue
        entries.append({"kind": "entry"} | pick | meta)

    def census(rows):
        by = collections.OrderedDict()
        for lab in sets:
            rs = [x for x in rows if lab in x["sets"]]
            by[lab] = {"bundles": len(rs), "est_s": round(sum(x["est_ms"] for x in rs) / 1e3, 3)}
        return by

    header = {
        "kind": "header",
        "format": 1,
        "generator": "scripts/gen_hip_preload_table.py",
        "note": ("one host kernel stub per statically registered HIP fat-binary bundle (translation unit) launched by "
                 "the listed production traces; kernel/symbol names, sizes and build ids only, no data, no binaries. "
                 "hipFuncGetAttributes on the stub = deferred code-object load of that bundle on the current device "
                 "(dlrm_v4/train/hip_preload.py). Regenerate after any torch/fbgemm/ROCm rebuild or trainer op change."),
        "arch": GFX,
        "libs": {lib: {"build_id": L["build_id"], "size": L["size"], "bundles": len(L["rows"]),
                       "anchor": L["anchor"], "path_at_generation": L["path"]} for lib, L in libs.items()},
        "traces": {lab: {"files": len(fs), "glob_example": fs[0]} for lab, fs in sets.items()},
        "cost_model": {"libtorch_ms_per_unc_mib": COST_LIBTORCH[1], "libtorch_ms_offset": COST_LIBTORCH[0],
                       "fbgemm_ms_per_co_mib": COST_FBGEMM},
        "n_entries": len(entries),
        "n_dlsym": sum(1 for e in entries if e["mode"] == "dlsym"),
        "n_symtab": sum(1 for e in entries if e["mode"] == "symtab"),
        "entries_by_set": census(entries),
        "uncovered": uncovered,
        "uncovered_by_set": census(uncovered),
        "unmapped_kernel_names": len(unmapped),
    }
    with open(a.out + ".tmp", "w") as f:
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    shutil.move(a.out + ".tmp", a.out)
    if a.summary:
        json.dump({"header": header, "unmapped": unmapped.most_common()}, open(a.summary, "w"), indent=1)
    print(f"wrote {a.out}: {len(entries)} entries ({header['n_dlsym']} dlsym, {header['n_symtab']} symtab), "
          f"uncovered {len(uncovered)}, by set {json.dumps(header['entries_by_set'])}, "
          f"unmapped names {len(unmapped)}, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
