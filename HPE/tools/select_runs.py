#!/usr/bin/env python3
"""Select the MLPerf-compliant set of N consecutive runs from a larger pool.

MLPerf Training allows running M>N runs and designating N *consecutive* runs for
scoring, but the Scoring section requires picking the window closest to the
MEDIAN result and explicitly prohibits picking the fastest:

    "provided that the submitter chooses the N consecutive runs that are closest
     to the median result."
    "Running multiple iterations of N runs to try find the lowest one is against
     the spirit of MLPerf and is prohibited."

"Consecutive" must be an objective, deterministic ordering; we use run_start
timestamps. This script prints the chosen files in order and, with --json, emits
machine-readable output for the packaging script.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

RUNS_REQUIRED = {"llama31_8b": 10, "llama2_70b_lora": 10, "flux1": 10, "gpt_oss_20b": 10, "dlrmv4": 10}


def parse_log(path):
    start = stop = samples = status = init = None
    with open(path, errors="replace") as fh:
        for line in fh:
            i = line.find(":::MLLOG")
            if i < 0:
                continue
            try:
                d = json.loads(line[i + 9 :])
            except json.JSONDecodeError:
                continue
            k = d.get("key")
            if k == "init_start" and init is None:
                init = d["time_ms"]
            if k == "run_start":
                start = d["time_ms"]
            elif k == "run_stop":
                stop = d["time_ms"]
                samples = d["metadata"].get("samples_count")
                status = d["metadata"].get("status")
    order = start or init
    if order is None:
        return None          # no timed region at all: not a run
    if stop is None or status != "success":
        # MLPerf rule: a failed run counts as infinite time in every window
        # that contains it (it is not simply dropped).
        return {"path": path, "start": order, "ttt_min": math.inf, "samples": math.inf}
    return {"path": path, "start": order, "ttt_min": (stop - start) / 60000.0,
            "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=sorted(RUNS_REQUIRED))
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    n = RUNS_REQUIRED[args.benchmark]
    runs = []
    for d in args.dirs:
        for f in glob.glob(os.path.join(d, "*.log")):
            bn = os.path.basename(f)
            # Primus also writes a train.*.exp.log copy of each run: skip duplicates.
            if bn.startswith("mlperf_logging") or ".exp." in bn or bn.startswith("train"):
                continue
            r = parse_log(f)
            if r:
                runs.append(r)
    runs.sort(key=lambda r: r["start"])

    if len(runs) < n:
        print(f"ERROR: only {len(runs)} runs, need {n}", file=sys.stderr)
        sys.exit(1)

    windows = []
    for i in range(len(runs) - n + 1):
        w = runs[i : i + n]
        t = sorted(x["ttt_min"] for x in w)
        s = sorted(x["samples"] for x in w)
        windows.append({"i": i, "score": statistics.mean(t[1:-1]),
                        "samples": statistics.mean(s[1:-1])})

    by_score = sorted(windows, key=lambda w: w["score"])
    chosen = by_score[(len(by_score) - 1) // 2]          # median window
    picked = runs[chosen["i"] : chosen["i"] + n]

    if args.json:
        print(json.dumps({
            "benchmark": args.benchmark,
            "window_start_index": chosen["i"],
            "olympic_score_min": chosen["score"],
            "olympic_samples": chosen["samples"],
            "files": [r["path"] for r in picked],
        }, indent=1))
        return

    nfail = sum(1 for r in runs if math.isinf(r["ttt_min"]))
    print(f"{len(runs)} runs ({nfail} failed = infinite time), {len(windows)} windows of {n}")
    if math.isinf(chosen["score"]):
        print("WARNING: the median window is invalid (2+ failed runs): the whole result is invalid")
    for w in windows:
        mark = "  <-- SELECTED (median)" if w is chosen else ""
        print(f"  runs {w['i']+1:>2}-{w['i']+n:<2}  olympic {w['score']:7.3f} min"
              f"  samples {w['samples']:>9.0f}{mark}")
    print(f"\nfastest window was {by_score[0]['score']:.3f} min "
          f"(runs {by_score[0]['i']+1}-{by_score[0]['i']+n}) - NOT submittable")
    print("\nselected files, in order:")
    for j, r in enumerate(picked, 1):
        print(f"  result_{j}.txt  <- {os.path.basename(r['path'])}"
              f"  ({r['ttt_min']:.3f} min, {r['samples']} samples)")


if __name__ == "__main__":
    main()
