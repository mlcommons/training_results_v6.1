#!/usr/bin/env python3
"""Correct non-performance submission identity emitted by the reference logger."""

import argparse
from pathlib import Path


REPLACEMENTS = {
    '"key": "submission_org", "value": "reference"':
        '"key": "submission_org", "value": "AMD"',
    '"key": "submission_platform", "value": "reference"':
        '"key": "submission_platform", "value": "MI355X"',
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()

    result_files = sorted(args.result_directory.glob("result_*.txt"))
    if len(result_files) != 10:
        raise RuntimeError(f"Expected 10 result logs, found {len(result_files)}")

    for result_file in result_files:
        text = result_file.read_text(encoding="utf-8")
        for old, new in REPLACEMENTS.items():
            if text.count(old) != 1:
                raise RuntimeError(
                    f"{result_file}: expected exactly one occurrence of {old}"
                )
            text = text.replace(old, new)
        result_file.write_text(text, encoding="utf-8", newline="")


if __name__ == "__main__":
    main()
