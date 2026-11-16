#!/usr/bin/env python3
"""Source the production config and submit the experiment-based batch script."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess


CONFIG_NAME = "config_GB200_2560x4x12xtp2pp4ep32cp1_mxfp8_full_cg_mrc.sh"
REQUIRED_PATHS = (
    "CONT", "DATADIR", "LOAD_CHECKPOINTS_DIR", "MLPERF_TRAIN_DIR",
    "NODE_LOGDIR", "NODE_TMPDIR",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hostfile", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--account", default=os.environ.get("SBATCH_ACCOUNT"))
    parser.add_argument("--partition", default=os.environ.get("SBATCH_PARTITION"))
    parser.add_argument("--qos", default=os.environ.get("SBATCH_QOS"))
    parser.add_argument("--time", type=int, default=60, help="Allocation minutes (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="Source config and print; no Slurm calls")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        hostfile = args.hostfile.resolve(strict=True)
        hosts = [line.strip() for line in hostfile.read_text().splitlines() if line.strip()]
    except OSError as error:
        parser.error(str(error))
    if len(hosts) != 2560 or len(set(hosts)) != 2560:
        parser.error("--hostfile must contain exactly 2560 distinct hostnames")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host) for host in hosts):
        parser.error("Use one plain hostname per line, not comments or Slurm expressions")
    if not 0 <= args.seed <= 4294967295:
        parser.error("--seed must be an integer in [0, 4294967295]")
    if args.time < 60:
        parser.error("--time must allow at least the production 60-minute allocation")

    env = os.environ.copy()
    env.setdefault("LOGROOT", str(root / "results"))
    for key in REQUIRED_PATHS + ("LOGROOT", "MRC_MOUNTS_FILE"):
        value = env.get(key)
        if key in REQUIRED_PATHS and not value:
            parser.error(f"{key} is required")
        if value is not None and (
            not Path(value).is_absolute() or any(c in value for c in ",:\n\r")
        ):
            parser.error(f"{key} must be absolute and cannot contain mount delimiters")
    if any(c in str(root) for c in ",:\n\r"):
        parser.error("Run on Linux from an implementation path without mount delimiters")

    cmd = [
        "sbatch", "--parsable", "--nodes=2560", "--exclusive", "--mem=0",
        f"--time={args.time}", f"--nodelist={hostfile}",
        f"--chdir={root}", "--export=ALL",
        f"--output={env['LOGROOT']}/slurm-%j.out",
        f"--error={env['LOGROOT']}/slurm-%j.err",
    ]
    for option in ("account", "partition", "qos"):
        value = getattr(args, option)
        if value:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                parser.error(f"--{option} must be a single site identifier")
            cmd.append(f"--{option}={value}")
    cmd.extend(["--", str(root / "run.sub")])

    # Match the experiment launcher: source before sbatch so the original batch
    # script receives its training configuration and MRC mounts through the environment.
    script = "\n".join((
        "set -euo pipefail",
        f"source {shlex.quote(str(root / 'configs' / CONFIG_NAME))}",
        f"export SEED={args.seed}",
        f"export SEED_BASE={args.seed}",
        'export DATESTAMP="$(date +%y%m%d%H%M%S%N)"',
        'export LOGDIR="${LOGROOT}/${DATESTAMP}"',
        'printf "CONFIG_FILENAME=%s\\nLOGDIR=%s\\n" "$CONFIG_FILENAME" "$LOGDIR"',
    ))
    if args.dry_run:
        script += f"\nprintf '%s\\n' {shlex.quote(shlex.join(cmd))}\n"
    else:
        script += '\nmkdir -p -- "$LOGDIR"\n'
        script += "exec " + shlex.join(cmd) + "\n"
    return subprocess.run(["bash", "-c", script], env=env, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
