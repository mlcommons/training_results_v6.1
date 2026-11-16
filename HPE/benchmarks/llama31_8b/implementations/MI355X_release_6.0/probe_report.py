#!/usr/bin/env python3
"""Parse a probe run and report steady-state training throughput.

Throughput is measured between consecutive evaluations (eval_stop -> next
eval_start), which excludes process startup, synthetic warmup, JIT/autotune and
the eval forward passes themselves.
"""
import glob
import json
import os
import sys


def load_events(logdir):
    events = []
    for path in glob.glob(os.path.join(logdir, "*.log")) + glob.glob(
        os.path.join(logdir, "*.out")
    ):
        with open(path, errors="replace") as fh:
            for line in fh:
                idx = line.find(":::MLLOG")
                if idx < 0:
                    continue
                try:
                    events.append(json.loads(line[idx + 9 :]))
                except json.JSONDecodeError:
                    pass
    events.sort(key=lambda d: d.get("time_ms", 0))
    return events


def main():
    label, logdir = sys.argv[1], sys.argv[2]
    events = load_events(logdir)

    gbs = 32
    eval_interval = int(os.environ.get("PRIMUS_EVAL_INTERVAL", "100"))
    marks, evals, run = [], [], {}
    pending = None
    for d in events:
        key, t = d.get("key"), d.get("time_ms", 0)
        if key == "global_batch_size":
            gbs = d.get("value", gbs)
        elif key == "eval_start":
            pending = t
        elif key == "eval_stop" and pending is not None:
            evals.append(t - pending)
            marks.append((pending, t))
            pending = None
        elif key in ("run_start", "run_stop", "init_stop", "overall_throughput"):
            run[key] = d.get("value") if key == "overall_throughput" else t

    # steady-state windows: gap between one eval_stop and the next eval_start
    samples_per_window = eval_interval * gbs
    rates = []
    for (_, stop), (start, _) in zip(marks, marks[1:]):
        dt = (start - stop) / 1000.0
        if dt > 1.0:
            rates.append(samples_per_window / dt)

    print(f"\n===== PROBE RESULT: {label} =====")
    if not rates:
        print("  NO STEADY-STATE WINDOW FOUND (run failed or too short)")
        print(f"  events parsed: {len(events)}, evals: {len(evals)}")
        tail = os.path.join(logdir, "stdout.log")
        if os.path.exists(tail):
            print("  --- last 25 lines of stdout ---")
            for line in open(tail, errors="replace").readlines()[-25:]:
                print("   ", line.rstrip())
        sys.exit(1)

    best = max(rates)
    median = sorted(rates)[len(rates) // 2]
    print(f"  windows              : {len(rates)}  ({samples_per_window} samples each)")
    print(f"  steady-state samp/s  : {[f'{r:.2f}' for r in rates]}")
    print(f"  MEDIAN steady-state  : {median:.2f} samples/s")
    print(f"  BEST   steady-state  : {best:.2f} samples/s")
    if evals:
        print(f"  eval count/mean time : {len(evals)}  {sum(evals)/len(evals)/1000:.2f}s")
    if "run_start" in run and "init_stop" in run:
        print(f"  init -> run_start    : {(run['run_start']-run['init_stop'])/1000:.1f}s")
    if run.get("overall_throughput"):
        print(f"  overall_throughput   : {run['overall_throughput']}")

    # Project a scored run: 184320 samples, ~3.3% eval overhead at the real cadence.
    projected = 184320 / (median * (1 - 0.033)) / 60
    print(f"  => projected TTT @184320 samples: {projected:.2f} min")
    with open(os.path.join(logdir, "result.json"), "w") as fh:
        json.dump(
            {
                "label": label,
                "median_steady_samples_per_s": median,
                "best_steady_samples_per_s": best,
                "windows": rates,
                "projected_ttt_min": projected,
            },
            fh,
            indent=1,
        )


if __name__ == "__main__":
    main()
