#!/usr/bin/env python3
# pyre-strict
"""S14-d HIP code-object preload (dlrm_v4/train/hip_preload.py): code default off = no-op, the checked-in table parses
and is consistent, every failure is counted and skipped (fail-open), symtab entries need a verified build-id, and the
train_ranker hook sits after the allocator pre-grow and before RUN_START. S14-d v2 (yambda_5b.gin binds the knob on):
the v2 table's per-shape counts, its widened b10/b144 rows and its 7 amendment rows, a dry run of every row through
preload_entries / preload_from_env with fakes, and the arming lines the gin item documents.

CPU-only: the HIP call is replaced by fakes; dlsym/maps/build-id checks run against this process's libc."""

import collections
import ctypes
import hashlib
import importlib.util
import json
import logging
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from generative_recommenders.dlrm_v4.train import hip_preload as hp

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_SHAPES = ("1n", "2n", "4n")
# S14-d v2 (rccl iter85 S14D85 section 6): rows appended byte-identically from proposed_rows.jsonl (this sha256) and
# libtorch_hip bundles widened from 1n to every shape.
_V2_ROWS_SHA256 = "51b052db4c6fd655d76ab8195069abaec881ec298f8a5f78540da0f41543abff"
_V2_ADDED = (2, 49, 55, 73, 74, 75, 77)
_V2_WIDENED = (10, 144)
_V2_PER_SHAPE = {"1n": (46, 2), "2n": (48, 0), "4n": (48, 0)}  # (selected, other_shape)
_TORCH = "libtorch_hip.so"


def _libc() -> Dict[str, Any]:
    maps = hp.mapped_libraries()
    paths = [p for p in maps if os.path.basename(p).startswith("libc.so")]
    if not paths:
        raise unittest.SkipTest("libc not found in /proc/self/maps")
    path = paths[0]
    base = maps[path]["base"]
    malloc = ctypes.cast(ctypes.CDLL(path, mode=hp.RTLD_NOLOAD | hp.RTLD_LAZY)["malloc"], ctypes.c_void_p).value
    return {"maps": maps, "path": path, "name": os.path.basename(path), "base": base, "malloc": malloc}


class _Fake:
    def __init__(self, rc: int = 0, raise_on: int = -1) -> None:
        self.rc = rc
        self.raise_on = raise_on
        self.calls: List[int] = []
        self.cleared = 0

    def fga(self, addr: int) -> int:
        self.calls.append(addr)
        if len(self.calls) - 1 == self.raise_on:
            raise RuntimeError("boom")
        return self.rc

    def clear(self) -> int:
        self.cleared += 1
        return 0


class ResolveAndDisabledTest(unittest.TestCase):
    def test_resolve_table(self) -> None:
        for v in (None, "", "0", "off", "OFF", "false", "no", "none", " 0 "):
            self.assertEqual(hp.resolve_table(v), (None, False), v)
        self.assertEqual(hp.resolve_table("1"), (hp.DEFAULT_TABLE, False))
        self.assertEqual(hp.resolve_table("on"), (hp.DEFAULT_TABLE, False))
        self.assertEqual(hp.resolve_table("all"), (hp.DEFAULT_TABLE, True))
        self.assertEqual(hp.resolve_table("/x/t.jsonl"), ("/x/t.jsonl", True))

    def test_disabled_is_noop(self) -> None:
        called: List[str] = []
        orig_maps, orig_hip = hp.mapped_libraries, hp._Hip
        hp.mapped_libraries = lambda: called.append("maps") or {}  # type: ignore[assignment]
        hp._Hip = lambda *a: called.append("hip")  # type: ignore[assignment,misc]
        try:
            with self.assertNoLogs(hp.logger, level=logging.DEBUG):
                for env in ({}, {hp.ENV: "0"}, {hp.ENV: ""}, {hp.ENV: "off"}):
                    self.assertIsNone(hp.preload_from_env(rank=0, world_size=8, local_world_size=8, env=env))
        finally:
            hp.mapped_libraries, hp._Hip = orig_maps, orig_hip
        self.assertEqual(called, [])

    def test_missing_table_never_raises(self) -> None:
        with self.assertLogs(hp.logger, level=logging.WARNING) as cm:
            res = hp.preload_from_env(rank=0, env={hp.ENV: "/nonexistent/hip_preload_table.jsonl"})
        self.assertEqual(res and res["armed"], False)
        self.assertIn(hp.NOT_ARMED, "\n".join(cm.output))


class TableTest(unittest.TestCase):
    def test_checked_in_table(self) -> None:
        header, entries, bad = hp.load_table(hp.DEFAULT_TABLE)
        self.assertEqual(bad, 0)
        self.assertEqual(header.get("kind"), "header")
        self.assertEqual(header.get("n_entries"), len(entries))
        self.assertGreaterEqual(len(entries), 30)
        libs = header["libs"]
        shapes = set(header["traces"])
        self.assertTrue({"1n", "2n", "4n"} <= shapes)
        seen = set()
        for e in entries:
            key = (e["lib"], e["bundle"])
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertIn(e["lib"], libs)
            self.assertTrue(e["sets"] and set(e["sets"]) <= shapes, e)
            if e["mode"] == "symtab":
                self.assertIsInstance(e["value"], int)
                self.assertTrue(libs[e["lib"]]["build_id"])
                self.assertIsInstance(libs[e["lib"]]["anchor"]["value"], int)
        names = {e["lib"] for e in entries}
        self.assertIn("libtorch_hip.so", names)
        self.assertTrue(any(n.startswith("fbgemm_gpu") for n in names))
        for shape in ("1n", "2n", "4n"):
            sel, other = hp.select_entries(header, entries, shape, False)
            self.assertEqual(len(sel) + other, len(entries))
            self.assertGreaterEqual(len(sel), 30)

    def test_parse_counts_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"kind": "header", "traces": {"1n": {}}, "libs": {}}) + "\n")
                f.write("not json\n[1, 2]\n\n")
                f.write(json.dumps({"kind": "entry", "lib": "a.so", "sym": "s", "mode": "dlsym", "sets": ["1n"]}) + "\n")
                f.write(json.dumps({"kind": "entry", "lib": "a.so", "sym": "t", "mode": "symtab"}) + "\n")  # no value
                f.write(json.dumps({"kind": "entry", "lib": "a.so", "sym": "u", "mode": "weird"}) + "\n")
            header, entries, bad = hp.load_table(p)
        self.assertEqual(header["kind"], "header")
        self.assertEqual([e["sym"] for e in entries], ["s"])
        self.assertEqual(bad, 4)

    def test_select_entries(self) -> None:
        header = {"traces": {"1n": {}, "2n": {}}}
        entries = [{"lib": "a", "sets": ["1n"]}, {"lib": "b", "sets": ["1n", "2n"]}, {"lib": "c", "sets": ["2n"]}]
        self.assertEqual([e["lib"] for e in hp.select_entries(header, entries, "1n", False)[0]], ["a", "b"])
        self.assertEqual(hp.select_entries(header, entries, "2n", False)[1], 1)
        self.assertEqual(len(hp.select_entries(header, entries, "8n", False)[0]), 3)  # unknown shape -> all
        self.assertEqual(len(hp.select_entries(header, entries, "1n", True)[0]), 3)


def _census(rows: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Dict[str, Any]]:
    """The per-set census rule of scripts/gen_hip_preload_table.py (header entries_by_set)."""
    out: Dict[str, Dict[str, Any]] = collections.OrderedDict()
    for lab in labels:
        rs = [x for x in rows if lab in x["sets"]]
        out[lab] = {"bundles": len(rs), "est_s": round(sum(x["est_ms"] for x in rs) / 1e3, 3)}
    return out


class TableV2Test(unittest.TestCase):
    """The S14-d v2 table: v1 rows + header.amendments (b10/b144 widened, 7 added symtab rows)."""

    def setUp(self) -> None:
        self.header, self.entries, self.bad = hp.load_table(hp.DEFAULT_TABLE)
        with open(hp.DEFAULT_TABLE) as f:
            self.lines: List[str] = f.read().splitlines()

    def test_per_shape_counts_and_census(self) -> None:
        h = self.header
        self.assertEqual((self.bad, len(self.entries)), (0, 48))
        self.assertEqual((h["n_entries"], h["n_dlsym"], h["n_symtab"]), (48, 26, 22))
        self.assertEqual(sum(1 for e in self.entries if e["mode"] == "symtab"), 22)
        for shape, (n, other) in _V2_PER_SHAPE.items():
            sel, o = hp.select_entries(h, self.entries, shape, False)
            self.assertEqual((len(sel), o), (n, other), shape)
            self.assertEqual(h["entries_by_set"][shape]["bundles"], n, shape)
        self.assertEqual(h["entries_by_set"], _census(self.entries, list(h["traces"])))
        self.assertEqual(len(hp.select_entries(h, self.entries, "8n", False)[0]), 48)  # unknown shape -> every row
        self.assertEqual(len(hp.select_entries(h, self.entries, "1n", True)[0]), 48)  # "all"

    def test_widened_and_added_rows(self) -> None:
        torch_rows = {e["bundle"]: e for e in self.entries if e["lib"] == _TORCH}
        for b in _V2_WIDENED + _V2_ADDED:
            e = torch_rows[b]
            self.assertEqual(e["sets"], list(_SHAPES), b)
            self.assertEqual(e["mode"], "symtab", b)
            self.assertTrue(isinstance(e["value"], int) and not isinstance(e["value"], bool) and e["value"] > 0, b)
        # appended byte-identically from the rows file, in its order, after every v1 row
        tail = self.lines[-len(_V2_ADDED):]
        self.assertEqual([json.loads(x)["bundle"] for x in tail], list(_V2_ADDED))
        self.assertEqual(hashlib.sha256(("\n".join(tail) + "\n").encode()).hexdigest(), _V2_ROWS_SHA256)
        # the only rows left out at 1n are the two 2n/4n-only v1 rows
        self.assertEqual(sorted((e["lib"], e["bundle"]) for e in self.entries if "1n" not in e["sets"]),
                         [("fbgemm_gpu_py.so", 37), (_TORCH, 98)])

    def test_amendment_record_and_build_id_gate(self) -> None:
        h = self.header
        meta = h["libs"][_TORCH]
        self.assertRegex(meta["build_id"], r"^[0-9a-f]{40}$")
        self.assertTrue(meta["anchor"]["sym"])
        self.assertIsInstance(meta["anchor"]["value"], int)
        (am,) = h["amendments"]
        self.assertEqual(am["rows_sha256"], _V2_ROWS_SHA256)
        self.assertEqual(am["rows_valid_for_build_id"], meta["build_id"])
        self.assertEqual([a["bundle"] for a in am["added"]], list(_V2_ADDED))
        self.assertEqual([w["bundle"] for w in am["widened"]], list(_V2_WIDENED))
        for shape in _SHAPES:
            new = [e for e in self.entries if e["lib"] == _TORCH and (
                e["bundle"] in _V2_ADDED or (e["bundle"] in _V2_WIDENED and shape != "1n"))]
            before = am["entries_by_set_before"][shape]["bundles"]
            self.assertEqual(h["entries_by_set"][shape]["bundles"] - before, len(new), shape)
            self.assertAlmostEqual(am["est_ms_added_by_set"][shape], sum(e["est_ms"] for e in new), places=6)


class _FakeHandle:
    """dlsym stand-in: one distinct fake address per symbol."""

    def __init__(self, base: int) -> None:
        self.base = base
        self.addr: Dict[str, int] = {}

    def __getitem__(self, sym: str) -> ctypes.c_void_p:
        return ctypes.c_void_p(self.addr.setdefault(sym, self.base + 16 * (len(self.addr) + 1)))


class _Opener:
    """open_library stand-in: every table library is 'loaded'; libtorch_hip's symtab base verified unless a reason is
    given (what open_library returns for a build-id / base / anchor mismatch)."""

    TORCH_BASE = 0x7F0000000000

    def __init__(self, torch_reason: str = "") -> None:
        self.torch_reason = torch_reason
        self.handles: Dict[str, _FakeHandle] = {}

    def __call__(self, name: str, meta: Dict[str, Any], maps: Dict[str, Dict[str, int]]) -> Any:
        h = self.handles.setdefault(name, _FakeHandle(0x100000000 * (len(self.handles) + 1)))
        if name == _TORCH and self.torch_reason:
            return hp._Lib(h, "/fake/" + name, None, self.torch_reason)  # type: ignore[arg-type]
        return hp._Lib(h, "/fake/" + name, self.TORCH_BASE if name == _TORCH else None, "")  # type: ignore[arg-type]


class _FakeHip:
    path = "/fake/" + _TORCH

    def __init__(self, maps: Dict[str, Dict[str, int]]) -> None:
        self.calls: List[int] = []

    def device(self) -> int:
        return 0

    def func_attributes(self, addr: int) -> int:
        self.calls.append(addr)
        return 0

    def clear_error(self) -> int:
        return 0


class DryRunV2Test(unittest.TestCase):
    """Every v2 row through the real preload_entries / preload_from_env with fake libraries and a fake HIP (CPU)."""

    def setUp(self) -> None:
        self.header, self.entries, _ = hp.load_table(hp.DEFAULT_TABLE)

    def test_every_row_resolves(self) -> None:
        for shape, (n, _other) in _V2_PER_SHAPE.items():
            sel, _ = hp.select_entries(self.header, self.entries, shape, False)
            f = _Fake()
            r = hp.preload_entries(sel, self.header["libs"], f.fga, f.clear, maps={}, opener=_Opener())
            c = r["counts"]
            self.assertEqual(c["ok"], n, shape)
            self.assertEqual(c["nolib"] + c["nosym"] + c["symtab_unverified"] + c["hip"] + c["error"], 0, (shape, c))
            self.assertEqual((len(f.calls), len(set(f.calls))), (n, n), shape)  # one distinct stub per bundle
            symtab = {_Opener.TORCH_BASE + e["value"] for e in sel if e["mode"] == "symtab"}
            self.assertEqual(len(symtab), 22, shape)
            self.assertTrue(symtab <= set(f.calls), shape)  # load base + st_value, as the table says

    def test_build_id_or_base_mismatch_skips_only_symtab_rows(self) -> None:
        sel, _ = hp.select_entries(self.header, self.entries, "2n", False)
        for reason in ("buildid", "base", "anchor"):
            f = _Fake()
            r = hp.preload_entries(sel, self.header["libs"], f.fga, f.clear, maps={},
                                   opener=_Opener(torch_reason=reason))
            c = r["counts"]
            self.assertEqual((c["symtab_unverified"], c["ok"]), (22, 26), reason)
            self.assertEqual(r["libs"][_TORCH]["reason"], reason)
            self.assertFalse({_Opener.TORCH_BASE + e["value"] for e in sel if e["mode"] == "symtab"} & set(f.calls))

    def _from_env(self, value: str, world: int, rank: int = 0, torch_reason: str = "") -> Any:
        real = hp.preload_entries
        op = _Opener(torch_reason)

        def entries_with_fake_libs(sel: List[Dict[str, Any]], libs: Dict[str, Any], fga: Any, clr: Any,
                                   maps: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Any]:
            return real(sel, libs, fga, clr, maps, opener=op)

        with mock.patch.object(hp, "mapped_libraries", return_value={}), mock.patch.object(hp, "_Hip", _FakeHip), \
                mock.patch.object(hp, "preload_entries", entries_with_fake_libs):
            with self.assertLogs(hp.logger, level=logging.INFO) as cm:
                res = hp.preload_from_env(rank=rank, world_size=world, local_world_size=8, env={hp.ENV: value})
        return res, "\n".join(cm.output)

    def test_arming_lines_per_shape(self) -> None:
        for world, shape in ((8, "1n"), (16, "2n"), (32, "4n")):
            n, other = _V2_PER_SHAPE[shape]
            res, log = self._from_env("1", world)
            self.assertTrue(res and res["armed"], shape)
            self.assertIn(f"{hp.ARMED} table=hip_preload_table.jsonl shape={shape} entries={n} (other_shape={other} "
                          f"bad_lines=0) ok={n} ", log)
            self.assertIn(" failed=0 (nolib=0 nosym=0 symtab_unverified=0 hip=0 error=0) in ", log)
            self.assertNotIn(hp.NOT_ARMED, log)
            self.assertNotIn("[hip-preload] rank ", log)
        _res, log = self._from_env("all", 8)  # env-only value: every row at any shape
        self.assertIn("shape=1n entries=48 (other_shape=0 bad_lines=0) ok=48 ", log)

    def test_failure_marker_of_other_ranks(self) -> None:
        res, log = self._from_env("1", 16, rank=3, torch_reason="buildid")
        self.assertEqual(res and res["failed"], 22)
        self.assertIn("[hip-preload] rank 3: ", log)
        self.assertIn("symtab_unverified=22", log)
        self.assertNotIn(hp.ARMED, log)  # the arming line is rank 0's


class FailOpenTest(unittest.TestCase):
    def test_dlsym_missing_lib_missing_symbol_hip_error(self) -> None:
        c = _libc()
        entries = [
            {"lib": "libdoesnotexist_hip_preload.so", "sym": "x", "mode": "dlsym", "bundle": 0},
            {"lib": c["name"], "sym": "malloc", "mode": "dlsym", "bundle": 1},
            {"lib": c["name"], "sym": "no_such_symbol_hip_preload_xyz", "mode": "dlsym", "bundle": 2},
            {"lib": c["name"], "sym": "free", "mode": "dlsym", "bundle": 3},
        ]
        ok = _Fake(rc=0)
        r = hp.preload_entries(entries, {}, ok.fga, ok.clear, maps=c["maps"])
        self.assertEqual(r["counts"]["nolib"], 1)
        self.assertEqual(r["counts"]["nosym"], 1)
        self.assertEqual(r["counts"]["ok"], 2)
        self.assertEqual(ok.calls[0], c["malloc"])
        self.assertEqual(ok.cleared, 0)
        bad = _Fake(rc=98)  # hipErrorInvalidDeviceFunction
        r = hp.preload_entries(entries, {}, bad.fga, bad.clear, maps=c["maps"])
        self.assertEqual((r["counts"]["hip"], r["counts"]["ok"]), (2, 0))
        self.assertEqual(bad.cleared, 2)  # every HIP error is cleared (no sticky last error for torch)

    def test_exception_is_counted_and_loop_continues(self) -> None:
        c = _libc()
        entries = [{"lib": c["name"], "sym": s, "mode": "dlsym", "bundle": i} for i, s in enumerate(("malloc", "free"))]
        f = _Fake(rc=0, raise_on=0)
        r = hp.preload_entries(entries, {}, f.fga, f.clear, maps=c["maps"])
        self.assertEqual((r["counts"]["error"], r["counts"]["ok"]), (1, 1))
        self.assertEqual(len(f.calls), 2)

    def test_symtab_needs_verified_build_id(self) -> None:
        c = _libc()
        bid = hp.memory_build_id(c["base"])
        if not bid:
            self.skipTest("libc has no GNU build-id")
        off = c["malloc"] - c["base"]
        meta = {"anchor": {"sym": "malloc", "value": off}, "build_id": bid}
        entries = [{"lib": c["name"], "sym": "anything", "mode": "symtab", "value": off, "bundle": 0}]
        f = _Fake()
        r = hp.preload_entries(entries, {c["name"]: meta}, f.fga, f.clear, maps=c["maps"])
        self.assertEqual(r["counts"]["ok"], 1)
        self.assertEqual(f.calls, [c["base"] + off])
        for bad_meta, reason in (
            ({"anchor": {"sym": "malloc", "value": off}, "build_id": "00" * 20}, "buildid"),
            ({"anchor": {"sym": "malloc", "value": off + 16}, "build_id": bid}, "base"),
            ({"anchor": {"sym": "no_such_symbol_hip_preload_xyz", "value": off}, "build_id": bid}, "anchor"),
            ({}, "anchor"),
        ):
            f = _Fake()
            r = hp.preload_entries(entries, {c["name"]: bad_meta}, f.fga, f.clear, maps=c["maps"])
            self.assertEqual(r["counts"]["symtab_unverified"], 1, reason)
            self.assertEqual(f.calls, [], reason)
            self.assertEqual(r["libs"][c["name"]]["reason"], reason)


class HookOrderTest(unittest.TestCase):
    def test_train_ranker_hook_position(self) -> None:
        with open(os.path.join(os.path.dirname(hp.__file__), "train_ranker.py")) as f:
            src = f.read()
        i_pregrow = src.index("_alloc_pregrow.pregrow_before_run_start()")
        i_hook = src.index("_hip_preload.preload_from_env(")
        i_guard = src.index('_tspec.recheck_globals("run_start")')
        i_run = src.index("mlperf_logger.log_run_start(")
        i_setup = src.index("    setup(\n")
        i_fp8 = src.index("run_fp8_first_use_warmup(\n            device=device")
        self.assertTrue(i_setup < i_fp8 < i_pregrow < i_hook < i_guard < i_run)
        self.assertEqual(src.count("_hip_preload.preload_from_env("), 1)


class GeneratorTest(unittest.TestCase):
    def _gen(self) -> Any:
        path = os.path.join(_REPO, "scripts", "gen_hip_preload_table.py")
        spec = importlib.util.spec_from_file_location("gen_hip_preload_table", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_candidates_attribution(self) -> None:
        g = self._gen()
        libs = {"libtorch_hip.so": {"name_first": {"k": 14, "only_torch": 3}},
                "fbgemm_gpu_py.so": {"name_first": {"k": 19}},
                "fbgemm_gpu_tbe_utils.so": {"name_first": {"k": 2}}}
        self.assertEqual(g.candidates("only_torch", ["aten::add"], libs), [("libtorch_hip.so", 3)])
        self.assertEqual(g.candidates("k", ["fbgemm::jagged_unique_indices"], libs),
                         [("fbgemm_gpu_py.so", 19), ("fbgemm_gpu_tbe_utils.so", 2)])
        self.assertEqual(g.candidates("k", ["aten::cumsum"], libs), [("libtorch_hip.so", 14)])
        self.assertEqual(len(g.candidates("k", [""], libs)), 3)
        self.assertEqual(g.candidates("absent", ["aten::add"], libs), [])

    def test_build_id_matches_memory_reader(self) -> None:
        g = self._gen()
        c = _libc()
        with open(c["path"], "rb") as f:
            elf = f.read()
        self.assertEqual(g.build_id(elf, g._sections(elf)), hp.memory_build_id(c["base"]))


if __name__ == "__main__":
    unittest.main()
