#!/usr/bin/env python3
"""Read-only asset audit for the pinned Qwen3.5 MLPerf code drop.

Uses NVIDIA's bundled, unmodified sparse fingerprint implementation. This is
an input check, not a model execution test or full-file cryptographic audit.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = {
    "expected-mounts.csv": "ede8a63f6057271706383f8814dfbc9a15c3567339ba437f55830480f757741b",
    "mountcheck.py": "eca8f91120238bee61d8702582dded3ee41af9b9682b5bafa16dd56a0a985e0f",
}
ENV = {
    "HF_CKPT": "HF_CKPT_PATH",
    "SIF_DIR": "NEMO_GYM_SWE_SIF_DIR",
    "TRAIN_DATA": "NEMO_GYM_SWE_TRAIN_DATA_PATH",
    "VAL_DATA": "NEMO_GYM_SWE_VALIDATION_DATA_PATH",
}


def load_manifest():
    for name, expected in PINS.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundled {name} differs from the pinned original")
    from mountcheck import fingerprint_file
    with (HERE / "expected-mounts.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["type"] != "file" for r in rows):
        raise ValueError("Unexpected non-file row in pinned manifest")
    return rows, fingerprint_file


def check_file(row, path, fingerprint):
    result = {"key": row["key"], "relative_path": row["relative_path"],
              "path": str(path), "expected_bytes": int(row["num_bytes"])}
    try:
        if not path.is_file():
            return dict(result, status="MISSING")
        before = path.stat()
        result["actual_bytes"] = before.st_size
        if before.st_size != result["expected_bytes"]:
            return dict(result, status="SIZE_MISMATCH")
        actual = fingerprint(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            return dict(result, status="CHANGED_DURING_CHECK")
        if actual["sparse_sha256"] != row["sparse_sha256"]:
            return dict(result, status="HASH_MISMATCH",
                        expected_sparse_sha256=row["sparse_sha256"],
                        actual_sparse_sha256=actual["sparse_sha256"])
        return dict(result, status="OK")
    except OSError as e:
        return dict(result, status="READ_ERROR", detail=str(e))


def read_tasks(path, expected_count):
    ids, errors = [], []
    try:
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = row["responses_create_params"]["metadata"]["instance_id"]
                if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
                    raise ValueError(f"invalid instance_id on line {line_no}")
                ids.append(task_id)
    except (OSError, ValueError, KeyError, TypeError) as e:
        errors.append(str(e))
    if len(ids) != expected_count:
        errors.append(f"expected {expected_count} rows, found {len(ids)}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate instance_id values")
    return {"rows": len(ids), "unique_ids": len(set(ids)), "errors": errors}, ids


def audit(roots, rows, fingerprint, workers):
    jobs = [(row, roots[row["key"]] if row["relative_path"] == "."
             else roots[row["key"]] / row["relative_path"]) for row in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: check_file(*job, fingerprint), jobs))
    by_key = {}
    for result in results:
        by_key.setdefault(result["key"], []).append(result)
    counts = {k: dict(Counter(r["status"] for r in group)) for k, group in by_key.items()}
    all_ok = lambda key: bool(by_key.get(key)) and all(r["status"] == "OK" for r in by_key[key])

    train, train_ids = read_tasks(roots["TRAIN_DATA"], 700)
    val, val_ids = read_tasks(roots["VAL_DATA"], 251)
    overlap = sorted(set(train_ids) & set(val_ids))
    sif_rows = {r["relative_path"]: r for r in rows if r["key"] == "SIF_DIR"}
    sif_results = {r["relative_path"]: r for r in by_key["SIF_DIR"]}

    def task_sifs(ids):
        checked = []
        for task_id in sorted(set(ids)):
            name = task_id + ".sif"
            if name not in sif_rows:
                checked.append({"relative_path": name, "status": "NOT_IN_OFFICIAL_MANIFEST"})
                continue
            result = sif_results[name]
            # The shipped runtime tries sifroot/r2egym/id.sif BEFORE sifroot/id.sif.
            # Validate the file it would actually open if both layouts exist.
            nested = roots["SIF_DIR"] / "r2egym" / name
            if nested.exists():
                result = check_file(sif_rows[name], nested, fingerprint)
            checked.append(result)
        return {"required": len(checked),
                "counts": dict(Counter(r["status"] for r in checked)),
                "issues": [r for r in checked if r["status"] != "OK"]}

    train_sifs, val_sifs = task_sifs(train_ids), task_sifs(val_ids)
    index_errors = []
    try:
        index = json.loads((roots["HF_CKPT"] / "model.safetensors.index.json").read_text())
        if not isinstance(index, dict):
            raise ValueError("model index must be an object")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model index weight_map must be a nonempty object")
        if any(not isinstance(v, str) for v in weight_map.values()):
            raise ValueError("model index shard names must be strings")
        shards = set(weight_map.values())
        if len(shards) != 94:
            index_errors.append(f"index references {len(shards)} shards; expected 94")
        manifest_shards = {r["relative_path"] for r in rows if r["key"] == "HF_CKPT"
                           and r["relative_path"].endswith(".safetensors")}
        if shards != manifest_shards:
            index_errors.append("index shard names differ from official manifest")
    except (OSError, ValueError, KeyError, TypeError) as e:
        index_errors.append(str(e))

    model_ok = all_ok("HF_CKPT") and not index_errors
    jsonl_ok = all_ok("TRAIN_DATA") and all_ok("VAL_DATA") and not train["errors"] and not val["errors"] and not overlap
    eval_ok = bool(model_ok and jsonl_ok and val_sifs["required"] == 251 and not val_sifs["issues"])
    train_ok = bool(eval_ok and train_sifs["required"] == 700 and not train_sifs["issues"])
    full_ok = all(r["status"] == "OK" for r in results)
    try:
        extra_sifs = sorted(str(p.relative_to(roots["SIF_DIR"])) for p in roots["SIF_DIR"].rglob("*.sif")
                            if str(p.relative_to(roots["SIF_DIR"])) not in sif_rows)
    except OSError:
        extra_sifs = []
    return {
        "code_drop_commit": "18aa52988ba9edd2d0fffa70ec75328fc15d21e3",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Official sparse SHA256 plus file sizes; no full-file MD5 or GPU execution test",
        "roots": {k: str(v) for k, v in roots.items()},
        "official_manifest_counts": counts,
        "model_index_errors": index_errors,
        "train_jsonl": train, "val_jsonl": val, "train_val_overlap": overlap,
        "train_task_sifs": train_sifs, "val_task_sifs": val_sifs,
        "eval_inputs_ok": eval_ok, "train_inputs_ok": train_ok,
        "full_official_manifest_ok": full_ok,
        "extra_local_sifs": extra_sifs,
        "official_manifest_issues": [r for r in results if r["status"] != "OK"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require", choices=["eval", "train", "full"], default="eval",
                        help="Select the exit-code gate; all three statuses are always reported")
    args = parser.parse_args()
    try:
        if args.workers < 1:
            raise ValueError("workers must be positive")
        missing = [v for v in ENV.values() if not os.environ.get(v)]
        if missing:
            raise ValueError("Source site.sh first; missing: " + ", ".join(missing))
        roots = {k: Path(os.environ[v]).expanduser().resolve() for k, v in ENV.items()}
        rows, fingerprint = load_manifest()
        print("Checking official sizes and sparse hashes (read-only)...", flush=True)
        report = audit(roots, rows, fingerprint, args.workers)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        for key, counts in report["official_manifest_counts"].items():
            print(f"OFFICIAL_{key}: {json.dumps(counts, sort_keys=True)}")
        for split in ("train", "val"):
            data = report[split + "_jsonl"]
            sifs = report[split + "_task_sifs"]
            print(f"{split.upper()}_JSONL: rows={data['rows']} unique={data['unique_ids']} errors={data['errors']}")
            print(f"{split.upper()}_REQUIRED_SIFS: {json.dumps(sifs['counts'], sort_keys=True)}")
        for label, key in (("EVAL_INPUTS", "eval_inputs_ok"), ("TRAIN_INPUTS", "train_inputs_ok"),
                           ("FULL_OFFICIAL_MANIFEST", "full_official_manifest_ok")):
            print(f"{label}={'PASS' if report[key] else 'FAIL'}")
        print(f"EXTRA_LOCAL_SIFS={len(report['extra_local_sifs'])} (informational; not deleted)")
        for issue in report["official_manifest_issues"][:8]:
            print(f"EXAMPLE: {issue['status']} {issue['key']} {issue['relative_path']}")
        print("REPORT=" + str(args.report.resolve()))
        gate = {"eval": "eval_inputs_ok", "train": "train_inputs_ok", "full": "full_official_manifest_ok"}[args.require]
        passed = report[gate]
        if args.require == "full":
            passed = passed and report["train_inputs_ok"]
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, TypeError) as e:
        print("CHECK_ERROR: " + str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
