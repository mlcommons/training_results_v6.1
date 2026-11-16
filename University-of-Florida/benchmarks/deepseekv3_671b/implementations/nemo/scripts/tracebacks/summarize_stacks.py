#!/usr/bin/env python3
"""Group per-rank hang dumps by identical stack signature.

A 256-rank hang produces 256 dumps but usually only a handful of distinct stacks.
This collapses them into those distinct signatures with the rank set that shares
each one, so the interesting minority (the ranks that are somewhere different) is
immediately visible.

Addresses, thread/LWP/process ids, timestamps and thread run-state are normalized
away before hashing, since those differ per rank without changing the stack.

Usage:
  summarize_stacks.py <artifact_dir> --kind pyspy
  summarize_stacks.py <artifact_dir> --kind gdb --grep nccl
  summarize_stacks.py <artifact_dir> --kind pyspy --max-lines 40 --top 5
"""

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

RANK_RE = re.compile(r"rank(\d+|UNKNOWN)_")

NORMALIZERS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\(LWP \d+\)"), "(LWP N)"),
    (re.compile(r"\bThread \d+(\.\d+)?\b"), "Thread N"),
    (re.compile(r"\bProcess \d+\b"), "Process N"),
    (re.compile(r"\bpid[= ]\d+\b"), "pid=N"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ][\d:.+-]+"), "TIMESTAMP"),
    (re.compile(r"\((?:idle|active)\)"), "(STATE)"),
    (re.compile(r"[ \t]+$", re.MULTILINE), ""),
]


def normalize(text: str) -> str:
    for pattern, repl in NORMALIZERS:
        text = pattern.sub(repl, text)
    return text.strip()


def rank_of(path: Path):
    match = RANK_RE.search(path.name)
    if not match:
        return None
    return None if match.group(1) == "UNKNOWN" else int(match.group(1))


def compress_ranks(ranks) -> str:
    """Render a sorted rank list as compact ranges: 0-63,128,130-131."""
    known = sorted(r for r in ranks if r is not None)
    unknown = sum(1 for r in ranks if r is None)
    parts, start, prev = [], None, None
    for rank in known:
        if start is None:
            start = prev = rank
            continue
        if rank == prev + 1:
            prev = rank
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = rank
    if start is not None:
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    if unknown:
        parts.append(f"{unknown}xUNKNOWN")
    return ",".join(parts) if parts else "(none)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--kind",
        default="pyspy",
        choices=["pyspy", "pyspy-native", "gdb", "cuda-gdb", "meta"],
        help="which dump type to group (default: pyspy)",
    )
    parser.add_argument("--max-lines", type=int, default=60, help="stack lines to print per group")
    parser.add_argument("--top", type=int, default=0, help="only show the N largest groups (0 = all)")
    parser.add_argument("--grep", default=None, help="only show groups whose stack matches this (case-insensitive)")
    args = parser.parse_args()

    if not args.artifact_dir.is_dir():
        print(f"ERROR: not a directory: {args.artifact_dir}", file=sys.stderr)
        return 1

    files = sorted(args.artifact_dir.glob(f"*.{args.kind}"))
    if not files:
        print(f"ERROR: no *.{args.kind} files in {args.artifact_dir}", file=sys.stderr)
        return 1

    groups = defaultdict(list)
    representative = {}
    for path in files:
        try:
            raw = path.read_text(errors="replace")
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        digest = hashlib.sha1(normalize(raw).encode()).hexdigest()[:12]
        groups[digest].append(rank_of(path))
        representative.setdefault(digest, (path, raw))

    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    if args.grep:
        needle = args.grep.lower()
        ordered = [(d, r) for d, r in ordered if needle in representative[d][1].lower()]
        if not ordered:
            print(f"no groups matching {args.grep!r}")
            return 0

    shown = ordered[: args.top] if args.top else ordered
    print(f"{len(files)} {args.kind} dumps -> {len(ordered)} distinct signature(s)\n")

    for digest, ranks in shown:
        path, raw = representative[digest]
        print("=" * 78)
        print(f"signature {digest}  ranks={len(ranks)}  [{compress_ranks(ranks)}]")
        print(f"representative: {path.name}")
        print("=" * 78)
        lines = raw.splitlines()
        for line in lines[: args.max_lines]:
            print(line)
        if len(lines) > args.max_lines:
            print(f"... ({len(lines) - args.max_lines} more lines; --max-lines to see more)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
