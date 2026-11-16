# Classification: Internal
"""Compare each result_<i>.txt with the raw log it was made from by redact_logs.sh.

For every pair it prints the line counts, whether the MLLOG lines are identical (same lines in the
same order), how many non-MLLOG lines differ, whether every differing line is the raw line with
/shared/<dir>/<dir> replaced by "[internal storage]", and how many /shared/ paths remain.
    python3 redaction_check.py <raw log dir> <results dir>
"""
import glob
import json
import os
import re
import sys

PATH = re.compile(r"/shared/[^/\s]+/[^/\s]+")


def lines(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="surrogateescape").split("\n")


def main(raw_dir, res_dir):
    raws = sorted(glob.glob(os.path.join(raw_dir, "*_[0-9][0-9].log")))
    for i, raw_path in enumerate(raws):
        res_path = os.path.join(res_dir, f"result_{i}.txt")
        raw, res = lines(raw_path), lines(res_path)
        same_count = len(raw) == len(res)
        mllog_raw = [l for l in raw if ":::MLLOG" in l]
        mllog_res = [l for l in res if ":::MLLOG" in l]
        changed = [(a, b) for a, b in zip(raw, res) if a != b] if same_count else []
        print(json.dumps({
            "result": os.path.basename(res_path),
            "raw_log": os.path.basename(raw_path),
            "lines_raw": len(raw),
            "lines_result": len(res),
            "mllog_lines": len(mllog_raw),
            "mllog_identical": mllog_raw == mllog_res,
            "changed_lines": len(changed),
            "changed_lines_mllog": sum(":::MLLOG" in a for a, _ in changed),
            "changes_are_path_redaction_only": all(PATH.sub("[internal storage]", a) == b for a, b in changed),
            "shared_paths_left_in_result": sum(len(PATH.findall(l)) for l in res),
        }))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
