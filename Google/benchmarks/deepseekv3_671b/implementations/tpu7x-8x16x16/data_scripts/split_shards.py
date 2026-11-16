r"""Splits the 2048 train shards of `c4/en:<version>` into 16384 shards.

Every train file is cut into `--pieces` (default 8) consecutive byte
ranges of whole TFRecord records, so with the default 2048 input files there
are 16384 output files of 2783 or 2784 records each.

The split is purely at the byte level. The TFRecord framing (length, masked
CRC32C of the length, payload, masked CRC32C of the payload) is never decoded
or re-encoded, so every record is copied exactly once, unchanged and in its
original order: input file `j` becomes output files `j * pieces + k`, in order.

The output is a complete dataset root: besides the 16384 train shards it holds
the rewritten `dataset_info.json` / `train_info.json`, and a copy of everything
else in the input (the validation shards and metadata of `c4/en/<version>`, and
all other versions such as the eval set), so MaxText can use it as
`dataset_path` for both the train and the eval dataset.

Example:

  python split_shards.py --src_dir=$HOME/tmp/dsv3_stage \
      --dst_dir=$HOME/tmp/dsv3_reshard_16384 --num_workers=32

Needs nothing beyond the Python standard library.
"""

import argparse
from collections.abc import Sequence
from concurrent import futures
import json
import logging
import os
import shutil
import struct
import time

_SPLIT = "train"
# Bytes around each record's payload: uint64 length + uint32 length CRC + uint32
# payload CRC.
_FRAMING_BYTES = 8 + 4 + 4


def _shard_name(shard: int, num_shards: int) -> str:
  return f"c4-{_SPLIT}.tfrecord-{shard:05d}-of-{num_shards:05d}"


def _record_offsets(data: bytes, path: str) -> list[int]:
  """Returns the start offset of every record in `data`, plus `len(data)`."""
  offsets = []
  position = 0
  while position < len(data):
    (length,) = struct.unpack_from("<Q", data, position)
    offsets.append(position)
    position += _FRAMING_BYTES + length
  if position != len(data):
    raise ValueError(f"{path}: truncated TFRecord ({position} vs {len(data)})")
  offsets.append(len(data))
  return offsets


def _split_file(
    src_path: str, shard: int, pieces: int, num_out: int, out_dir: str
) -> tuple[int, list[int]]:
  """Splits one TFRecord file into `pieces` files; returns its record counts.

  Output file `shard * pieces + k` gets `n // pieces` records, plus one for the
  first `n % pieces` pieces.

  Returns:
    `(shard, [records in each of the `pieces` output files])`.
  """
  with open(src_path, "rb") as handle:
    data = handle.read()
  offsets = _record_offsets(data, src_path)
  view = memoryview(data)
  base, extra = divmod(len(offsets) - 1, pieces)
  counts = []
  start = 0
  for piece in range(pieces):
    count = base + (1 if piece < extra else 0)
    path = os.path.join(out_dir, _shard_name(shard * pieces + piece, num_out))
    with open(path + ".tmp", "wb") as out:
      out.write(view[offsets[start] : offsets[start + count]])
    os.replace(path + ".tmp", path)
    counts.append(count)
    start += count
  return shard, counts


def _read_json(path: str):
  with open(path, encoding="utf-8") as handle:
    return json.load(handle)


def _write_json(path: str, value) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
  )
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
      "--src_dir", required=True,
      help="TFDS root of the 2048-shard dataset (the output of build_dataset.py).",
  )
  parser.add_argument(
      "--dst_dir", required=True,
      help="Local TFDS root to write the resharded dataset to (must be new).",
  )
  parser.add_argument(
      "--version", default="3.0.5", help="TFDS version holding the train split."
  )
  parser.add_argument(
      "--pieces", type=int, default=8, help="Output files per input file."
  )
  parser.add_argument(
      "--num_workers", type=int, default=min(32, os.cpu_count() or 1),
      help="Processes splitting files in parallel (each holds one ~22 MB file).",
  )
  args = parser.parse_args(argv)
  if "://" in args.dst_dir or "://" in args.src_dir:
    parser.error("--src_dir/--dst_dir must be local paths")
  started = time.monotonic()

  src_dir = os.path.join(os.path.abspath(args.src_dir), "c4", "en", args.version)
  dst_dir = os.path.join(os.path.abspath(args.dst_dir), "c4", "en", args.version)
  if os.path.exists(dst_dir):
    parser.error(f"{dst_dir} already exists")

  train_info = _read_json(os.path.join(src_dir, f"{_SPLIT}_info.json"))
  expected = [int(n) for n in train_info["shard_lengths"]]
  num_in = len(expected)
  num_out = num_in * args.pieces
  os.makedirs(dst_dir)

  lengths = [0] * num_out
  with futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
    pending = [
        pool.submit(
            _split_file,
            os.path.join(src_dir, _shard_name(shard, num_in)),
            shard, args.pieces, num_out, dst_dir,
        )
        for shard in range(num_in)
    ]
    for done, future in enumerate(futures.as_completed(pending), 1):
      shard, counts = future.result()
      if sum(counts) != expected[shard]:
        raise ValueError(
            f"input shard {shard}: {sum(counts)} records, expected"
            f" {expected[shard]}"
        )
      lengths[shard * args.pieces : (shard + 1) * args.pieces] = counts
      if done % max(1, num_in // 20) == 0 or done == num_in:
        logging.info(
            "%d/%d input shards split (%.0f s)",
            done, num_in, time.monotonic() - started,
        )
  if sum(lengths) != sum(expected):
    raise ValueError("record count changed by the split")
  shard_lengths = [str(n) for n in lengths]

  # Metadata: only the train split's shard lengths change (numBytes is the same,
  # the bytes are).
  info = _read_json(os.path.join(src_dir, "dataset_info.json"))
  for split in info["splits"]:
    if split["name"] == _SPLIT:
      split["shardLengths"] = shard_lengths
  info["description"] = (
      info.get("description", "")
      + f" | train resharded {num_in}->{num_out} files (byte-level split)"
  )
  _write_json(os.path.join(dst_dir, "dataset_info.json"), info)
  train_info["shard_lengths"] = shard_lengths
  _write_json(os.path.join(dst_dir, f"{_SPLIT}_info.json"), train_info)

  # Everything else is carried over verbatim: validation shards, features.json,
  # validation_info.json, ... and the other versions (the eval set).
  rewritten = {"dataset_info.json", f"{_SPLIT}_info.json"}
  train_prefix = f"c4-{_SPLIT}.tfrecord-"
  for name in sorted(os.listdir(src_dir)):
    if name not in rewritten and not name.startswith(train_prefix):
      shutil.copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))
  en_src = os.path.dirname(src_dir)
  for version in sorted(os.listdir(en_src)):
    if version != args.version:
      shutil.copytree(
          os.path.join(en_src, version),
          os.path.join(os.path.dirname(dst_dir), version),
      )
      logging.info("copied version %s unchanged", version)

  logging.info(
      "wrote %d train shards (%d records) to %s in %.0f s",
      num_out, sum(lengths), dst_dir, time.monotonic() - started,
  )


if __name__ == "__main__":
  main()
