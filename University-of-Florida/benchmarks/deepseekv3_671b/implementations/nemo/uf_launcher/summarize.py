#!/usr/bin/env python3
"""One line per run of a run_b300.sh / run_hpg.sub job.

Usage: summarize.py <LOGDIR> <DATESTAMP>
Reads <DATESTAMP>_<i>.log (sorted MLPerf lines) and compliance_<DATESTAMP>_<i>.out.
"""
import glob
import json
import os
import re
import sys

KEYS_61 = [
    "lowest_numerical_precision_in_linear", "lowest_numerical_precision_in_attn",
    "lowest_numerical_precision_in_comm", "tensor_parallelism", "pipeline_parallelism",
    "context_parallelism", "expert_parallelism", "micro_batch_size", "config_filename",
]

logdir, datestamp = sys.argv[1], sys.argv[2]
logs = sorted(glob.glob(os.path.join(logdir, f"{datestamp}_*.log")),
              key=lambda p: int(re.search(r"_(\d+)\.log$", p).group(1)) if re.search(r"_(\d+)\.log$", p) else 0)
logs = [p for p in logs if re.search(rf"{datestamp}_\d+\.log$", p)]
print(f"{'run':>4} {'status':9} {'minutes':>8} {'step s':>7} {'evals':>5} {'last eval':>10} {'seed':>6} {'9 keys':7} compliance")
times = []
for path in logs:
    idx = re.search(r"_(\d+)\.log$", path).group(1)
    events = []
    with open(path, errors="replace") as f:
        for line in f:
            if ":::MLLOG " in line:
                try:
                    events.append(json.loads(line.split(":::MLLOG ", 1)[1]))
                except json.JSONDecodeError:
                    pass
    start = next((e["time_ms"] for e in events if e.get("key") == "run_start"), None)
    stop = next((e for e in events if e.get("key") == "run_stop"), None)
    status = (stop or {}).get("metadata", {}).get("status", "none")
    minutes = (stop["time_ms"] - start) / 60000 if stop and start else None
    evals = [e["value"] for e in events if e.get("key") == "eval_accuracy"]
    # Median train step time (tracked_stats), without the first value (it includes the compile time)
    steps = [e["value"]["train_step_time"] for e in events if e.get("key") == "tracked_stats"
             and isinstance(e.get("value"), dict) and "train_step_time" in e["value"]][1:]
    step = sorted(steps)[len(steps) // 2] if steps else None
    seed = next((e["value"] for e in events if e.get("key") == "seed"), "-")
    keys_ok = all(sum(e.get("key") == k for e in events) == 1 for k in KEYS_61)
    comp = os.path.join(logdir, f"compliance_{datestamp}_{idx}.out")
    result = "-"
    if os.path.exists(comp):
        text = open(comp, errors="replace").read()
        result = "SUCCESS" if "SUCCESS" in text else "FAILED"
        warn = [ln.split("WARNING -", 1)[1].strip() for ln in text.splitlines() if "WARNING -" in ln]
        if result == "FAILED" and warn:
            result += ": " + warn[0][:70]
    if status == "success" and minutes is not None:
        times.append(minutes)
    print(f"{idx:>4} {status:9} {minutes if minutes is None else round(minutes, 2)!s:>8} "
          f"{'-' if step is None else round(step, 3)!s:>7} {len(evals):>5} "
          f"{(round(evals[-1], 4) if evals else '-')!s:>10} {seed!s:>6} {'OK' if keys_ok else 'BAD':7} {result}")
if times:
    times.sort()
    print(f"successful runs: {len(times)}, minutes min {times[0]:.2f} median {times[len(times) // 2]:.2f} max {times[-1]:.2f}")
