#!/usr/bin/env python3
"""Read-only input validation for the GB300 16-node Qwen397B launcher.

Reuse the prior asset audit and NVIDIA sparse fingerprint implementation.
The official manifest's Git blob is unchanged in the 2026-10-13 code drop:
26b77d4e15123623ce8ad139c4196a9c6d4c14d0.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rl_root", type=Path,
                        help="Shared root, normally /lustre/$USER/mlperf-rl")
    parser.add_argument("--report", type=Path,
                        help="Defaults to RL_ROOT/logs/assets_check_1013.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.rl_root.expanduser().resolve()
    data = root / "data"
    env = os.environ.copy()
    defaults = {
        "HF_CKPT_PATH": data / "qwen35-397b-a17b",
        "NEMO_GYM_SWE_SIF_DIR": data / "r2egym-sifs-aarch64" / "images",
        "NEMO_GYM_SWE_TRAIN_DATA_PATH": data / "benchmark-r2e-gym-easy" / "benchmark_r2e_gym_easy_train.jsonl",
        "NEMO_GYM_SWE_VALIDATION_DATA_PATH": data / "benchmark-r2e-gym-easy" / "benchmark_r2e_gym_easy_val.jsonl",
    }
    for key, path in defaults.items():
        if not env.get(key):
            env[key] = str(path)
        print(f"{key}={env[key]}", flush=True)
    report = args.report or root / "logs" / "assets_check_1013.json"
    checker = Path(__file__).resolve().parent / "assets" / "check_assets.py"
    return subprocess.call(
        [sys.executable, str(checker), "--require", "train", "--workers",
         str(args.workers), "--report", str(report)], env=env)


if __name__ == "__main__":
    sys.exit(main())
