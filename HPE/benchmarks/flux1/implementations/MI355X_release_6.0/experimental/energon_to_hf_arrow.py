"""Rebuild the MLCommons Flux.1 preprocessed HF datasets (.arrow) from our
energon/webdataset copy, so AMD's Primus v26.7 recipe (which reads the HF
format with load_from_disk) can run without re-copying 2.3 TB from /nas.

Exact inverse of scripts/to_webdataset.py: that script stored every field's
raw bytes unchanged (.t5/.clip/.mean/.logvar.bytes, key and timestep in
.json) and named samples by global index, so sample i of the HF dataset is
member f"{i:08d}.*" of shard i // 10000. Rows are written in the original
file layout (contiguous shards, 100-row record batches, schema + HF metadata
copied from a reference file) so the output is byte-identical to the
MLCommons files, and every file is checked against MLCommons' official md5
list. Any mismatch is reported and the exit status is non-zero.

Usage (inside an image with pyarrow, e.g. rocm/amd-mlperf:flux1_training_6.0):
  python energon_to_hf_arrow.py --energon DIR/train --out DIR/cc12m_preprocessed \
      --ref REF.arrow --md5 flux-1-cc12m-preprocessed.md5 --rows 1099776 --files 4762 \
      [--only 0-15] [--workers 16]
"""

import argparse
import array
import hashlib
import io
import json
import os
import sys
import tarfile
from multiprocessing import Pool

import pyarrow as pa

SAMPLES_PER_SHARD = 10000
BATCH_ROWS = 100
A = None  # args, set in each worker


def layout(rows, files):
    q, r = divmod(rows, files)
    sizes = [q + 1 if j < r else q for j in range(files)]
    starts = [0] * files
    for j in range(1, files):
        starts[j] = starts[j - 1] + sizes[j - 1]
    return starts, sizes


class Shards:
    def __init__(self, root):
        self.root, self.cache = root, {}

    def member(self, i, ext):
        k = i // SAMPLES_PER_SHARD
        if k not in self.cache:
            if len(self.cache) >= 2:
                old = min(self.cache)
                self.cache.pop(old)[0].close()
            tf = tarfile.open(os.path.join(self.root, f"shard_{k:06d}.tar"))
            self.cache[k] = (tf, {m.name: m for m in tf.getmembers()})
        tf, members = self.cache[k]
        return tf.extractfile(members[f"{i:08d}.{ext}"]).read()


def init(args):
    global A, SHARDS, SCHEMA, STARTS, SIZES
    A = args
    SHARDS = Shards(args.energon)
    with pa.OSFile(args.ref, "rb") as f:
        SCHEMA = pa.ipc.open_stream(f).schema
    STARTS, SIZES = layout(args.rows, args.files)


def column(values, typ, spill=b""):
    # The MLCommons files were written by an Arrow version that padded the
    # string column's data buffer to a multiple of 64 bytes by reading on into
    # its source buffer, i.e. into the keys of the following samples. Values
    # are unchanged (offsets bound every key); only the padding bytes differ.
    if typ != pa.string():
        return pa.array(values, type=typ)
    raw = [v.encode("utf-8") for v in values]
    offsets, pos = [0], 0
    for r in raw:
        pos += len(r)
        offsets.append(pos)
    data = b"".join(raw)
    if spill:
        pad = -len(data) % 64
        data += (spill + b"\0" * pad)[:pad]
    offs = array.array("i", offsets).tobytes()
    return pa.Array.from_buffers(typ, len(raw), [None, pa.py_buffer(offs), pa.py_buffer(data)])


def build(j):
    name = f"data-{j:05d}-of-{A.files:05d}.arrow"
    path = os.path.join(A.out, name)
    want = A.md5s.get(name)
    if os.path.exists(path) and want:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        if h.hexdigest() == want:
            return name, "kept", want
    has_ts = "timestep" in SCHEMA.names
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, SCHEMA) as w:
        start, n = STARTS[j], SIZES[j]
        for b0 in range(start, start + n, BATCH_ROWS):
            cols = {c: [] for c in SCHEMA.names}
            for i in range(b0, min(b0 + BATCH_ROWS, start + n)):
                meta = json.loads(SHARDS.member(i, "json"))
                cols["__key__"].append(meta["key"])
                cols["t5_encodings"].append(SHARDS.member(i, "t5.bytes"))
                cols["clip_encodings"].append(SHARDS.member(i, "clip.bytes"))
                cols["mean"].append(SHARDS.member(i, "mean.bytes"))
                cols["logvar"].append(SHARDS.member(i, "logvar.bytes"))
                if has_ts:
                    cols["timestep"].append(meta["timestep"])
            end = min(b0 + BATCH_ROWS, start + n)
            spill = b""
            i = end
            while len(spill) < 64 and i < start + n:
                spill += json.loads(SHARDS.member(i, "json"))["key"].encode("utf-8")
                i += 1
            arrays = [column(cols[c], SCHEMA.field(c).type, spill) for c in SCHEMA.names]
            w.write_batch(pa.record_batch(arrays, schema=SCHEMA))
    data = sink.getvalue()
    got = hashlib.md5(data).hexdigest()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return name, "built", got


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--energon", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--md5", required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--files", type=int, required=True)
    p.add_argument("--only", default="")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    args.md5s = {}
    for line in open(args.md5):
        h, n = line.split()
        args.md5s[n] = h
    os.makedirs(args.out, exist_ok=True)
    if args.only:
        a, b = (int(x) for x in args.only.split("-"))
        todo = list(range(a, b + 1))
    else:
        todo = list(range(args.files))
    bad = done = 0
    with Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        for name, how, got in pool.imap(build, todo, chunksize=max(1, SAMPLES_PER_SHARD // (args.rows // args.files) // 2)):
            done += 1
            if got != args.md5s.get(name):
                bad += 1
                print(f"MD5 MISMATCH {name}: {got} vs official {args.md5s.get(name)}", flush=True)
            if done % 100 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} files, {bad} mismatches", flush=True)
    print(f"RESULT {done} files, {bad} md5 mismatches vs official list")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
