#!/usr/bin/env python3
"""Report llama2 convergence-class distribution per config.

The llama2 score is dominated by *where* runs land on the 384-sequence eval
grid, not by raw wallclock. This separates the two:

  - hit-rate at each convergence point (the thing to optimise)
  - TTT *within* a class (true throughput, free of convergence luck)
  - the resulting RCP-normalised effective score

Usage: llama2_convergence.py <logdir> [<logdir> ...]
"""
import collections
import glob
import json
import os
import statistics
import sys

RCP_MEAN = 3178.67
MIN_EPOCHS = 2960.63


def parse(path):
    start = stop = samples = status = None
    final_loss = None
    with open(path, errors="replace") as fh:
        for line in fh:
            i = line.find(":::MLLOG")
            if i < 0:
                continue
            try:
                d = json.loads(line[i + 9:])
            except json.JSONDecodeError:
                continue
            k = d.get("key")
            if k == "run_start":
                start = d["time_ms"]
            elif k == "run_stop":
                stop = d["time_ms"]
                samples = d["metadata"].get("samples_count")
                status = d["metadata"].get("status")
            elif k == "eval_accuracy":
                final_loss = d.get("value")
    if start is None or stop is None or status != "success":
        return None
    return samples, (stop - start) / 60000.0, final_loss


def effective(ttt, samples):
    return ttt * (RCP_MEAN / samples if samples < RCP_MEAN else 1.0)


def main():
    print(f"{'config':<18}{'n':>3}{'@2688':>7}{'@3072':>7}{'@3456':>7}"
          f"{'TTT@3072':>10}{'olympic':>9}{'eff':>8}  verdict")
    for d in sys.argv[1:]:
        # Skip Primus v26.x's train.*.exp.log: a copy of the last trial's log.
        files = [f for f in sorted(glob.glob(os.path.join(d, "*.log")))
                 if not (os.path.basename(f).startswith("train.") and f.endswith(".exp.log"))]
        runs = [r for r in (parse(f) for f in files) if r]
        if not runs:
            continue
        by = collections.defaultdict(list)
        for s, t, _ in runs:
            by[s].append(t)
        n = len(runs)
        cnt = {k: len(v) for k, v in by.items()}

        times = sorted(t for _, t, _ in runs)
        samps = sorted(s for s, _, _ in runs)
        if n >= 3:
            oly_t = statistics.mean(times[1:-1])
            oly_s = statistics.mean(samps[1:-1])
            eff = effective(oly_t, oly_s)
            verdict = "FAIL <min_epochs" if oly_s < MIN_EPOCHS else ""
        else:
            oly_t = oly_s = eff = float("nan")
            verdict = "(need >=3 runs)"

        t72 = f"{statistics.mean(by[3072]):.3f}" if 3072 in by else "-"
        print(f"{os.path.basename(d.rstrip('/')):<18}{n:>3}"
              f"{cnt.get(2688,0):>7}{cnt.get(3072,0):>7}{cnt.get(3456,0):>7}"
              f"{t72:>10}{oly_s:>9.0f}{eff:>8.3f}  {verdict}")

    print(f"\n  reference: RCP mean {RCP_MEAN:.0f}, min_epochs {MIN_EPOCHS:.0f}")
    print("  target: maximise @3072 share; effective score is the figure of merit")
    print("  AMD MI355X published 8.270 min (implies ~7.99 raw at 3072)")


if __name__ == "__main__":
    main()
