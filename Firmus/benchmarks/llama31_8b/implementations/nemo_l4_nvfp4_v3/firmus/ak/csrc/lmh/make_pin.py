#!/usr/bin/env python3
"""make_pin.py -- the generator of lmh_pin.json (gap M-1), and the check that a pin still holds on a given image.

    python3 make_pin.py --ext <dir with ak_l8b_lmh.so> --out <new pin json>             write a pin (needs one GB300)
    python3 make_pin.py --ext <dir> --out <new pin json> --check <shipped lmh_pin.json>  ... and compare it

The pin is written by test_lmh.py's `enum` mode, unchanged (the function that wrote the shipped lmh_pin.json,
60104c48..., on 2026-10-08 in the 20260929 image; test_lmh.py here is that file, validated sha256 c5cedef1..., with only
its command line moved to argparse and the lmh force override dropped; enum() is unchanged -- csrc/CHANGES_vs_validated.md): for the recipe's
two LM-head GEMMs (forward x[16384, 4096] @ W[128256, 4096]^T and dgrad dy[16384, 128256] @ W, bf16), it enumerates
cuBLASLt's heuristic list through ak_l8b_lmh.heuristics, runs every algorithm, records the kernel each launches (torch
profiler), its time and whether its output equals torch's matmul byte for byte, and pins the algorithm that launches the
target kernel (forward nvjet_sm103_tst_256x256_64x4_2x2f_2cta_h_bz_TNT, dgrad nvjet_sm103_tst_128x512_64x4_2x1_2cta_v_bz_NNT)
by its 64 bytes.  It also records the sha256 of Megatron's stock LinearWithGradAccumulationAndAsyncCommunication.backward
source, which the lmh patch's dgrad path is made from.

--check compares the fields the lmh patch relies on with the shipped pin and exits non-zero on any difference:
the two algorithms' 64 bytes, the kernels they launch, their byte-equality with torch, bwd_src_sha256 and
bwd_src_dgrad_lines.  cuBLASLt / torch versions and timings are reported, not compared.
"""
import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def entry(pin, kind):
    e = pin.get("%s_T16384" % kind)
    if isinstance(e, str):          # the shipped pin stores the two entries as Python reprs
        e = ast.literal_eval(e)
    return e or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ext", required=True, help="directory holding ak_l8b_lmh.so")
    ap.add_argument("--out", required=True, help="the pin to write (never the shipped file)")
    ap.add_argument("--check", help="a pin to compare the new one with (the shipped lmh_pin.json)")
    a = ap.parse_args()
    if a.check and os.path.abspath(a.check) == os.path.abspath(a.out):
        sys.exit("make_pin: --out must not overwrite --check")
    sys.path.insert(0, HERE)
    import test_lmh                                  # noqa: E402  (the validated enumerator)
    test_lmh.enum(a.ext, a.out)
    if test_lmh.FAIL:
        print("[make_pin] enum FAILURES %d %s" % (len(test_lmh.FAIL), test_lmh.FAIL), flush=True)
        sys.exit(1)
    if not a.check:
        return
    new, old = json.load(open(a.out)), json.load(open(a.check))
    bad = []
    for k in ("bwd_src_sha256", "bwd_src_dgrad_lines"):
        if new.get(k) != old.get(k):
            bad.append("%s: %r v shipped %r" % (k, new.get(k), old.get(k)))
    for kind in ("forward", "dgrad"):
        n, o = entry(new, kind), entry(old, kind)
        for k in ("algo", "kernel", "equal_torch"):
            if n.get(k) != o.get(k):
                bad.append("%s %s: %r v shipped %r" % (kind, k, n.get(k), o.get(k)))
        print("[make_pin] %s: algo %s kernel %s equal_torch %s, %.1f us (shipped %.1f us); cuBLASLt %s (shipped %s)" % (
            kind, str(n.get("algo_sha"))[:16], n.get("kernel"), n.get("equal_torch"), n.get("time_us") or 0.0,
            o.get("time_us") or 0.0, new.get("cublasLt"), old.get("cublasLt")), flush=True)
    for b in bad:
        print("[make_pin] DIFFERS: %s" % b, flush=True)
    print("[make_pin] pin %s" % ("HOLDS" if not bad else "DOES NOT HOLD"), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
