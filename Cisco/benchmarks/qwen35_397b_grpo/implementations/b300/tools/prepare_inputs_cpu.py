''                                                                        

                                                                             
                                                                              
                                                                              
                                                                             
   
import hashlib
import os
import json
from pathlib import Path

from mlperf_common.mountcheck import inspect, load_csv, save_csv

ROOT = Path(os.environ["QWEN_DATA_ROOT"])
OFFICIAL = Path(__file__).resolve().parents[1]
OUT = Path("/smoke/expected-mounts-x86_64.csv")
HF = ROOT / "huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/8472618112abcbd45acbcdc58436aff4233c23f7"
TASKS = ROOT / "qwen35_data/benchmark-r2e-gym-easy"
SIFS = ROOT / "qwen35_data/r2egym-sifs-x86_64"
inputs = [f"HF_CKPT:{HF}", f"TRAIN_DATA:{TASKS / 'benchmark_r2e_gym_easy_train.jsonl'}",
          f"VAL_DATA:{TASKS / 'benchmark_r2e_gym_easy_val.jsonl'}"]
original = [row for row in load_csv(OFFICIAL / "expected-mounts.csv") if row["key"] != "SIF_DIR"]
actual = inspect(inputs)
lookup = {(r["key"], r["type"], r["relative_path"]): r for r in actual}
for row in original:
    key = (row["key"], row["type"], row["relative_path"])
    assert key in lookup, f"Missing original expected input: {key}"
    for field in ("num_files", "num_bytes", "sparse_sha256"):
        assert lookup[key][field] == row[field], f"Original input mismatch: {key} {field}"
print("HF checkpoint and task files match NVIDIA's original sparse manifest", flush=True)

md5s = {}
for line in (SIFS / "r2egym-sifs-x86_64.md5").read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    if name.startswith("images/") and name.endswith(".sif"):
        md5s[name.removeprefix("images/")] = digest
present = {p.name for p in (SIFS / "images").iterdir() if p.is_file()}
assert present == set(md5s), "SIF file inventory differs from official x86 MD5 package listing"
for name in sorted(md5s)[:3]:
    digest = hashlib.md5()
    with (SIFS / "images" / name).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    assert digest.hexdigest() == md5s[name], f"SIF MD5 anchor mismatch: {name}"
print(f"Official x86 package inventory matches ({len(md5s)} SIFs); 3 whole-file MD5 anchors passed", flush=True)

rows = inspect([f"SIF_DIR:{SIFS / 'images'}"])
for row in rows:
    del row["full_path"]
save_csv(original + rows, OUT)
print(json.dumps({"status": "PASS", "manifest": str(OUT), "sif_files": len(md5s),
                  "scope": "Original HF/task sparse checks; x86 inventory + 3 full MD5 anchors + all sparse hashes",
                  "full_sif_md5_qualification": False}), flush=True)
