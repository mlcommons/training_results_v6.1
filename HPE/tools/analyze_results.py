#!/usr/bin/env python3
"""Summarise MLPerf run logs: per-run TTT, samples to converge, throughput, and
the MLPerf score (olympic mean = drop fastest and slowest, average the rest).

Usage: analyze_results.py <benchmark> <logdir> [<logdir> ...]
       benchmark is 'llama31_8b', 'llama2_70b_lora', 'flux1', 'gpt_oss_20b' or 'dlrmv4'
"""
import glob
import json
import os
import statistics
import sys

# RCP reference data, training_6.0.0.
RCP = {
    # llama31_8b @ global_batch_size 32
    "llama31_8b": {
        "rcp_mean": 178858.67,
        "min_epochs": 173649.71,
        "runs_required": 10,
        "published": {"AMD MI355X": 86.84, "Dell XE9785L": 91.413},
        "target": 3.3,
        "lower_is_better_metric": "eval loss",
    },
    "llama2_70b_lora": {
        # Values taken from the official mlperf_logging rcp_checker output for
        # training_6.0.0, not recomputed here.
        "rcp_mean": 3178.67,
        "min_epochs": 2960.63,
        "runs_required": 10,
        "published": {"AMD MI355X": 8.27, "Dell XE9785L": 8.640, "Dell XE9785": 8.585},
        "target": 0.925,
        "lower_is_better_metric": "eval loss",
    },
    # flux1 @ global_batch_size 512, the smallest BS with a reference RCP and
    # therefore the cheapest convergence budget (BS 1024/2048/4096 need
    # 8.49M/10.30M/15.38M samples). A single node should always run BS 512.
    #
    # Recomputed here with the official rcp_checker algorithm (olympic mean
    # rejecting 1 sample per side; Min Epochs from the binary-search t-test
    # against alpha=0.05, submission_runs=10). Verified by reproducing the
    # published llama31_8b BS 32 pair exactly.
    "flux1": {
        "rcp_mean": 7223523.56,
        "min_epochs": 7112400.32,
        "runs_required": 10,
        # NOTE: no 8-GPU flux1 result was published in v6.0, so none of these
        # are like-for-like against a single node. Scale is in the label.
        "published": {
            "AMD MI325X 64gpu": 92.36,
            "Nebius B300 32gpu": 77.48,
            "GB300 NVL72 32gpu": 65.85,
        },
        "scale_note": "no 8-GPU flux1 result exists in v6.0; the entries below "
                      "are larger-scale and NOT comparable to a 1x8 run",
        "target": 0.586,
        "lower_is_better_metric": "validation loss",
    },
    # gpt_oss_20b @ global_batch_size 32. The official rcp_checker PRUNES the
    # BS 32 reference set (its "pruned_rcps" pass) and interpolates BS 32 from
    # the BS 16 and BS 64 sets, so the values that score a submission are the
    # interpolated ones below (checked 2026-10-09 against rcp_checker 6.0.0-rc5),
    # not the raw BS 32 statistics (234,837 / 229,259).
    "gpt_oss_20b": {
        "rcp_mean": 230058.67,
        "min_epochs": 222720.78,
        "runs_required": 10,
        "published": {"best B300 8gpu": 82.55},
        "scale_note": "no MI355X gpt_oss_20b result was published in v6.0",
        "target": 3.34,
        "lower_is_better_metric": "eval loss",
    },
    # dlrmv4 @ global_batch_size 8192 (one node), training_6.1.0 rcp_checker
    # (pruned pass), 2026-10-11. Eval grid = 2,293,760 samples, so reachable
    # stops are 66.52M (block 29, legal, scaled x1.038) / 68.81M (30) / 71.11M (31).
    "dlrmv4": {
        "rcp_mean": 69067662.22,
        "min_epochs": 66297354.0,
        "runs_required": 10,
        "published": {},
        "scale_note": "new benchmark in 6.1; no published result yet",
        "target": 0.75,
        "lower_is_better_metric": "AUC (higher is better)",
    },
}


def parse(path):
    start = stop = None
    samples = status = None
    evals = []
    pending = None
    evaltime = 0.0
    seed = None
    with open(path, errors="replace") as fh:
        for line in fh:
            i = line.find(":::MLLOG")
            if i < 0:
                continue
            try:
                d = json.loads(line[i + 9 :])
            except json.JSONDecodeError:
                continue
            k, t = d.get("key"), d.get("time_ms", 0)
            if k == "run_start":
                start = t
            elif k == "run_stop":
                stop = t
                samples = d["metadata"].get("samples_count")
                status = d["metadata"].get("status")
            elif k == "eval_accuracy":
                evals.append((d["metadata"].get("samples_count"), d.get("value")))
            elif k == "eval_start":
                pending = t
            elif k == "eval_stop" and pending:
                evaltime += (t - pending) / 1000.0
                pending = None
            elif k == "seed":
                seed = d.get("value")
    if start is None or stop is None:
        return None
    return {
        "file": os.path.basename(path),
        "ttt_s": (stop - start) / 1000.0,
        "start_ms": start,
        "samples": samples,
        "status": status,
        "evals": evals,
        "evaltime_s": evaltime,
        "seed": seed,
    }


def effective_score(raw, osamp, meta):
    """Apply RCP normalisation the way the official scorer does.

    If the olympic-mean samples-to-converge is below the RCP mean, the score is
    scaled back UP by RCP_mean/your_mean. Published results are already
    RCP-normalised, so any comparison against them must use this value, not the
    raw wallclock. Comparing raw-vs-published silently flatters a submission
    that converged early.
    """
    rm = meta.get("rcp_mean")
    if not rm or not osamp or osamp >= rm:
        return raw, 1.0
    f = rm / osamp
    return raw * f, f


def main():
    bench = sys.argv[1]
    meta = RCP.get(bench, {})
    runs = []
    for d in sys.argv[2:]:
        for f in sorted(glob.glob(os.path.join(d, "*.log"))):
            # mlperf_logging*: checker output. train.*.exp.log: Primus v26.x copies
            # the last trial's log there, which would count one run twice.
            b = os.path.basename(f)
            if b.startswith("mlperf_logging") or (b.startswith("train.") and b.endswith(".exp.log")):
                continue
            r = parse(f)
            if r:
                runs.append(r)

    if not runs:
        print("no complete runs found")
        return

    print(f"\n===== {bench} : {len(runs)} complete run(s) =====")
    print(f"{'run':<34}{'TTT min':>9}{'samples':>9}{'samp/s':>8}{'eval s':>8}{'final':>9}  status")
    for r in runs:
        thr = r["samples"] / r["ttt_s"] if r["samples"] else 0
        final = r["evals"][-1][1] if r["evals"] else float("nan")
        print(
            f"{r['file']:<34}{r['ttt_s']/60:>9.3f}{r['samples']:>9}{thr:>8.2f}"
            f"{r['evaltime_s']:>8.1f}{final:>9.4f}  {r['status']}"
        )

    good = [r for r in runs if r["status"] == "success"]
    times = sorted(r["ttt_s"] / 60 for r in good)
    samps = sorted(r["samples"] for r in good)

    print(f"\n  mean TTT (all {len(times)})      : {statistics.mean(times):.3f} min")
    if len(times) >= 3:
        oly = times[1:-1]
        print(f"  olympic mean (drop hi/lo) : {statistics.mean(oly):.3f} min   <-- MLPerf score method")
    if samps:
        print(f"  mean samples to converge  : {statistics.mean(samps):.0f}")
        if len(samps) >= 3:
            osamp = statistics.mean(samps[1:-1])
            print(f"  olympic mean samples      : {osamp:.0f}")
            if meta.get("rcp_mean"):
                rm, me = meta["rcp_mean"], meta["min_epochs"]
                print(f"  RCP mean / min_epochs     : {rm:.0f} / {me:.0f}")
                if osamp < me:
                    print("  RCP VERDICT: *** FAIL *** converging faster than min_epochs")
                elif osamp < rm:
                    f = rm / osamp
                    print(f"  RCP VERDICT: pass, but score scaled by {f:.4f} (no net gain)")
                else:
                    print(f"  RCP VERDICT: pass, scaling_factor = 1.0 "
                          f"({100*(osamp/rm-1):+.1f}% vs RCP mean)")

    if len(times) >= 3 and meta.get("published"):
        raw = statistics.mean(times[1:-1])
        osamp = statistics.mean(samps[1:-1]) if len(samps) >= 3 else None
        score, factor = effective_score(raw, osamp, meta)
        print("\n  vs published:")
        if meta.get("scale_note"):
            print(f"    !! {meta['scale_note']}")
        if factor != 1.0:
            print(f"    (raw {raw:.3f} min x RCP scaling {factor:.4f} = effective {score:.3f} min)")
        for name, val in meta["published"].items():
            print(f"    {name:<16}{val:>8.3f} min   ours {score:.3f} -> "
                  f"{'BEAT' if score < val else 'behind'} by {abs(100*(score/val-1)):.2f}%")

    n = meta.get("runs_required", 10)
    sliding_window(good, n, meta)
    print(f"\n  runs required for a valid score: {n}")


def sliding_window(good, n, meta):
    """Score every window of n consecutive runs (ordered by run_start).

    MLPerf allows running M>N runs and designating N *consecutive* runs, but
    section 'Scoring' requires choosing the window closest to the MEDIAN, and
    explicitly prohibits picking the fastest:
      "Running multiple iterations of N runs to try find the lowest one is
       against the spirit of MLPerf and is prohibited."
    Both are printed so the cost of compliance is visible, but only the median
    window may actually be submitted.
    """
    if len(good) <= n:
        return
    ordered = sorted(good, key=lambda r: r["start_ms"])
    windows = []
    for i in range(len(ordered) - n + 1):
        w = ordered[i : i + n]
        t = sorted(r["ttt_s"] / 60 for r in w)
        s = sorted(r["samples"] for r in w)
        windows.append(
            {
                "i": i,
                "score": statistics.mean(t[1:-1]),
                "samples": statistics.mean(s[1:-1]),
            }
        )

    print(f"\n  ---- sliding windows of {n} consecutive runs ----")
    print(f"  {'window':<12}{'olympic TTT':>13}{'olympic samples':>18}")
    for w in windows:
        print(f"  runs {w['i']+1}-{w['i']+n:<6}{w['score']:>13.3f}{w['samples']:>18.0f}")

    by_score = sorted(windows, key=lambda w: w["score"])
    median_w = by_score[(len(by_score) - 1) // 2]
    best_w = by_score[0]
    print(f"\n  COMPLIANT (median window) : runs {median_w['i']+1}-{median_w['i']+n}"
          f"  score {median_w['score']:.3f} min")
    print(f"  fastest window (NOT submittable): runs {best_w['i']+1}-{best_w['i']+n}"
          f"  score {best_w['score']:.3f} min")
    if meta.get("rcp_mean"):
        rm, me = meta["rcp_mean"], meta["min_epochs"]
        os_ = median_w["samples"]
        if os_ < me:
            print("  median-window RCP: *** FAIL *** below min_epochs")
        elif os_ < rm:
            print(f"  median-window RCP: pass, score scaled by {rm/os_:.4f}"
                  f" -> effective {median_w['score']*rm/os_:.3f} min")
        else:
            print("  median-window RCP: pass, scaling_factor = 1.0")
    eff, _ = effective_score(median_w["score"], median_w["samples"], meta)
    for name, val in (meta.get("published") or {}).items():
        print(f"    vs {name:<16}{val:>8.3f} -> "
              f"{'BEAT' if eff < val else 'behind'} by {abs(100*(eff/val-1)):.2f}%"
              f"   (effective {eff:.3f})")


if __name__ == "__main__":
    main()
