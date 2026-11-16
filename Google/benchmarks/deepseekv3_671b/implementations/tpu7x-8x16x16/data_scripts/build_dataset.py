r"""Rebuilds the MLPerf DeepSeek-V3 C4 TFDS datasets from the MLCommons bundle.

A standalone port (only `numpy` and `tensorflow` are needed) of the builder that
produced the C4 datasets read by the DeepSeek-V3 submissions. It reads the Llama
3.1 8B preprocessed C4 corpus that MLCommons publishes as Megatron `.bin` /
`.idx` files and writes TFDS `tfrecord` shards that are byte-identical to the
reference datasets (see README.md).

Three stages, selected with `--stages`:

  train, validation
      A 1:1 repackaging of the Megatron corpus into `c4/en/<version>`: one
      TFRecord (a `tf.train.Example` with a single int32 `ids` feature) per
      Megatron document, split contiguously across `--num_shards` shards.

  eval
      The MLPerf evaluation set `c4/en/<eval_version>`: the
      first `--eval_num_sequences` packed sequences of the validation stream,
      replicated across `--eval_num_shards` shards of `--eval_sequences_per_shard`
      sequences each (shard count is a multiple of 128: 128 shards x 8 sequences
      cover the 1024 eval sequences once; 2048 supports up to 2048 eval
      data-loading hosts). Each shard holds ONE record (see
      `replicate_across_shards` and README.md for why).

Example:

  python build_dataset.py \
      --src_dir=$HOME/tmp/mlperf_c4 --dst_dir=$HOME/tmp/dsv3_stage \
      --stages=train,validation,eval --num_workers=32

Output is always written to the local filesystem under
`<dst_dir>/c4/en/<version>/`; upload it yourself afterwards.

Only `numpy` is needed to read the corpus. `tensorflow` (CPU is enough) is
imported inside the writer processes, so the shard bytes are produced by the
same `tf.io.TFRecordWriter` / `tf.train.Example` code that wrote the datasets of
the submitted runs.
"""

import argparse
from collections.abc import Callable, Iterable, Sequence
from concurrent import futures
import dataclasses
import hashlib
import json
import logging
import multiprocessing
import os
import struct
import time

import numpy as np

# TensorFlow is only imported inside worker processes; keep its C++ logging quiet.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DATASET = "c4"
CONFIG = "en"
STAGES = ("train", "validation", "eval")

# TFDS's `naming.DEFAULT_FILENAME_TEMPLATE`.
FILEPATH_TEMPLATE = "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}"

# Verbatim content of the `features.json` of the datasets used by the submitted runs.
FEATURES_JSON = {
    "pythonClassName": (
        "tensorflow_datasets.core.features.features_dict.FeaturesDict"
    ),
    "featuresDict": {
        "features": {
            "ids": {
                "pythonClassName": (
                    "tensorflow_datasets.core.features.sequence_feature.Sequence"
                ),
                "sequence": {
                    "feature": {
                        "pythonClassName": (
                            "tensorflow_datasets.core.features.tensor_feature.Tensor"
                        ),
                        "tensor": {
                            "shape": {},
                            "dtype": "int32",
                            "encoding": "none",
                        },
                    },
                    "length": "-1",
                },
            }
        }
    },
}

DESCRIPTION = (
    "Pre-tokenized C4 dataset with verified authentic token boundaries and zero"
    " empty records"
)
CONFIG_DESCRIPTION = (
    "English tokenized C4 for MLPerf (100% full dataset, 0% empty records)"
)
EVAL_DESCRIPTION = (
    "Pre-packed MLPerf DeepSeek-V3 evaluation set derived from the MLCommons C4"
    " validation split"
)


# ----------------------------------------------------------------------------
# Megatron `.bin` / `.idx` reader.
# ----------------------------------------------------------------------------

_MAGIC = b"MMIDIDX\x00\x00"
_HEADER_BYTES = 34

# Megatron's dtype code -> numpy dtype (integer token types only). The MLPerf C4
# corpus uses code 4 (int32): the Llama 3.1 vocabulary (128256) needs > 16 bits.
_DTYPE_CODES = {
    1: np.dtype("u1"),
    2: np.dtype("i1"),
    3: np.dtype("<i2"),
    4: np.dtype("<i4"),
    5: np.dtype("<i8"),
    8: np.dtype("<u2"),
}


class Corpus:
  """Read-only view of a Megatron-LM `MMIDIDX` `.bin` / `.idx` pair.

  The `.idx` file is a 34-byte header followed by three flat little-endian
  arrays (N = number of sequences; for this corpus one sequence == one
  document)::

      offset 0          b"MMIDIDX\x00\x00"        magic (9 bytes)
      offset 9          version      uint64       (== 1)
      offset 17         dtype code   uint8        (4 == int32)
      offset 18         N            uint64       number of sequences
      offset 26         D            uint64       number of doc_idx entries (N+1)
      offset 34         sizes        int32[N]     tokens per sequence
      offset 34 + 4N    pointers     int64[N]     byte offset into the `.bin`
      offset 34 + 12N   doc_idx      int64[D]     sequence index where each
                                                  document starts (0..N)

  The `.bin` file is the bare concatenation of every sequence's tokens (no
  separators, no end-of-document markers), `sizes[i]` tokens of `dtype` each.
  """

  def __init__(self, stem: str):
    """Opens `<stem>.bin` and `<stem>.idx`."""
    self.bin_path = stem + ".bin"
    self.idx_path = stem + ".idx"
    with open(self.idx_path, "rb") as handle:
      header = handle.read(_HEADER_BYTES)
    if len(header) != _HEADER_BYTES or header[:9] != _MAGIC:
      raise ValueError(f"{self.idx_path}: not a Megatron MMIDIDX index")
    (version,) = struct.unpack_from("<Q", header, 9)
    (code,) = struct.unpack_from("<B", header, 17)
    (num_docs,) = struct.unpack_from("<Q", header, 18)
    (num_doc_idx,) = struct.unpack_from("<Q", header, 26)
    if version != 1:
      raise ValueError(f"{self.idx_path}: unsupported index version {version}")
    if code not in _DTYPE_CODES:
      raise ValueError(f"{self.idx_path}: unsupported dtype code {code}")
    expected_size = _HEADER_BYTES + 12 * num_docs + 8 * num_doc_idx
    if os.path.getsize(self.idx_path) != expected_size:
      raise ValueError(
          f"{self.idx_path}: size {os.path.getsize(self.idx_path)} != expected"
          f" {expected_size} for {num_docs} sequences"
      )
    self.num_docs = int(num_docs)
    self.dtype = _DTYPE_CODES[code]
    # Lazily paged, so opening the 45.6M-document train index is instant.
    self.sizes = np.memmap(
        self.idx_path, "<i4", "r", _HEADER_BYTES, (self.num_docs,)
    )
    self.pointers = np.memmap(
        self.idx_path, "<i8", "r", _HEADER_BYTES + 4 * self.num_docs,
        (self.num_docs,),
    )
    self.doc_idx = np.memmap(
        self.idx_path, "<i8", "r", _HEADER_BYTES + 12 * self.num_docs,
        (int(num_doc_idx),),
    )

  def check(self, chunk: int = 1 << 22) -> None:
    """Verifies the invariants `read_documents` and the TFDS metadata rely on.

    Raises:
      ValueError: if the `.bin` is truncated, documents are not stored
        back-to-back, a document is empty (the submitted runs have
        `empty_records: 0`), or a Megatron document is not exactly one sequence.
    """
    itemsize = self.dtype.itemsize
    if len(self.doc_idx) != self.num_docs + 1:
      raise ValueError(f"{self.idx_path}: doc_idx has {len(self.doc_idx)} entries")
    end = 0  # Byte offset where the next sequence must start.
    for first in range(0, self.num_docs, chunk):
      last = min(first + chunk, self.num_docs)
      sizes = np.asarray(self.sizes[first:last], dtype=np.int64)
      if sizes.min() <= 0:
        raise ValueError(f"{self.idx_path}: empty document in [{first}, {last})")
      starts = end + (np.cumsum(sizes) - sizes) * itemsize
      if not np.array_equal(self.pointers[first:last], starts):
        raise ValueError(f"{self.idx_path}: sequences not contiguous in [{first}, {last})")
      if not np.array_equal(self.doc_idx[first:last], np.arange(first, last)):
        raise ValueError(f"{self.idx_path}: documents are not single sequences")
      end += int(sizes.sum()) * itemsize
    if self.doc_idx[self.num_docs] != self.num_docs:
      raise ValueError(f"{self.idx_path}: documents are not single sequences")
    if os.path.getsize(self.bin_path) != end:
      raise ValueError(
          f"{self.bin_path}: size {os.path.getsize(self.bin_path)} != {end}"
          " bytes described by the index (truncated download?)"
      )

  def read_documents(self, first: int, count: int) -> list[np.ndarray]:
    """Returns `count` consecutive documents starting at document `first`."""
    sizes = np.asarray(self.sizes[first : first + count], dtype=np.int64)
    total = int(sizes.sum())
    flat = np.fromfile(
        self.bin_path, self.dtype, total, offset=int(self.pointers[first])
    )
    if flat.size != total:
      raise EOFError(f"{self.bin_path}: wanted {total} tokens, got {flat.size}")
    return np.split(flat, np.cumsum(sizes[:-1]))


# ----------------------------------------------------------------------------
# Eval packing.
# ----------------------------------------------------------------------------


def pack_sequences(
    corpus: Corpus, num_sequences: int, sequence_length: int, eos_id: int
) -> np.ndarray:
  """Returns the first `num_sequences` packed sequences of the corpus.

  Matches MaxText's `c4_mlperf` stream chunking: documents are concatenated
  with `eos_id` (128001) after each document and the stream is cut every 4096
  tokens.

  Returns:
    An `int32` array of shape `[num_sequences, sequence_length]`.
  """
  needed = num_sequences * sequence_length
  # Each document contributes len(doc) + 1 tokens once its EOS is appended.
  cumulative = np.cumsum(np.asarray(corpus.sizes, dtype=np.int64) + 1)
  if cumulative[-1] < needed:
    raise ValueError(
        f"{corpus.idx_path}: stream holds {cumulative[-1]} tokens but {needed}"
        " were requested; lower --eval_num_sequences"
    )
  num_docs = int(np.searchsorted(cumulative, needed, side="left")) + 1
  stream = np.empty(int(cumulative[num_docs - 1]), dtype=np.int32)
  cursor = 0
  for document in corpus.read_documents(0, num_docs):
    stream[cursor : cursor + len(document)] = document
    cursor += len(document)
    stream[cursor] = eos_id
    cursor += 1
  return stream[:needed].reshape(num_sequences, sequence_length)


def replicate_across_shards(
    sequences: np.ndarray,
    num_shards: int,
    sequences_per_shard: int,
    reserve_eos_slot: bool = False,
) -> list[np.ndarray]:
  """Lays the packed sequences out as one flat record per shard.

  MaxText's `c4_mlperf` loader re-runs the packing on whatever it reads: it
  appends one EOS per *record*, flattens, and cuts every `sequence_length`
  tokens, dropping the remainder. Storing finished sequences as individual
  records would therefore misalign every chunk after the first, so each shard
  carries its sequences concatenated end to end in a single record. The one EOS
  MaxText appends then falls into the discarded remainder and the chunks it
  produces are bit-identical to the sequences stored here.

  Shard `s` gets sequences `(s * sequences_per_shard + j) % len(sequences)` for
  `j in range(sequences_per_shard)`, so every sequence appears the same number
  of times and the mean loss equals the mean over the distinct sequences.

  Args:
    sequences: `[num_unique, sequence_length]` from `pack_sequences`.
    num_shards: shard count is a multiple of 128 (128 shards x 8 sequences cover
      the 1024 eval sequences once; 2048 supports up to 2048 eval data-loading
      hosts).
    sequences_per_shard: sequences each host evaluates.
    reserve_eos_slot: drop the final token of each shard so that the EOS MaxText
      appends completes the last sequence instead of being discarded. Costs one
      token per shard but keeps the chunking aligned when a host reads more than
      one shard (hosts != num_shards).

  Returns:
    One `int32` token array per shard.
  """
  num_unique, sequence_length = sequences.shape
  total = num_shards * sequences_per_shard
  if total % num_unique:
    raise ValueError(
        f"{num_shards} shards x {sequences_per_shard} sequences = {total} is"
        f" not a whole number of repeats of {num_unique} distinct sequences"
    )
  shards = []
  for shard in range(num_shards):
    indices = [
        (shard * sequences_per_shard + j) % num_unique
        for j in range(sequences_per_shard)
    ]
    tokens = sequences[indices].reshape(sequences_per_shard * sequence_length)
    shards.append(tokens[:-1] if reserve_eos_slot else tokens)
  return shards


# ----------------------------------------------------------------------------
# TFDS shard + metadata writing.
# ----------------------------------------------------------------------------


def shard_filename(split: str, shard: int, num_shards: int) -> str:
  """Renders one shard's filename, e.g. `c4-train.tfrecord-00007-of-02048`."""
  return f"{DATASET}-{split}.tfrecord-{shard:05d}-of-{num_shards:05d}"


def plan_shard_lengths(num_records: int, num_shards: int) -> list[int]:
  """Splits `num_records` contiguously across `num_shards`, TFDS-style.

  The first `num_records % num_shards` shards get one extra record. This
  reproduces the `shardLengths` of the submitted runs exactly (train: 1699 x
  22270 then 349 x 22269; validation: 1093 x 45 then 955 x 44).
  """
  if num_shards <= 0:
    raise ValueError(f"num_shards must be positive, got {num_shards}")
  base, remainder = divmod(num_records, num_shards)
  return [base + (1 if i < remainder else 0) for i in range(num_shards)]


def _serialize_ids(tf, ids: np.ndarray) -> bytes:
  """Encodes one document as a serialized `tf.train.Example`."""
  feature = tf.train.Feature(
      int64_list=tf.train.Int64List(value=np.asarray(ids, dtype=np.int64))
  )
  example = tf.train.Example(
      features=tf.train.Features(feature={"ids": feature})
  )
  return example.SerializeToString()


def _write_tfrecord(path: str, documents: Iterable[np.ndarray]) -> int:
  """Writes `documents` as one uncompressed TFRecord file; returns its size."""
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

  tmp_path = path + ".tmp"
  with tf.io.TFRecordWriter(tmp_path) as writer:
    for ids in documents:
      writer.write(_serialize_ids(tf, ids))
  os.replace(tmp_path, path)
  return os.path.getsize(path)


def _md5(path: str) -> str:
  digest = hashlib.md5(usedforsecurity=False)
  with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ShardStat:
  """What a worker reports about the shard it wrote."""

  shard: int
  num_bytes: int
  md5: str  # Only computed for the (tiny) eval shards.


@dataclasses.dataclass(frozen=True)
class _DocumentJob:
  """A contiguous run of Megatron documents to write as one shard."""

  stem: str
  out_path: str
  shard: int
  first_doc: int
  num_docs: int


@dataclasses.dataclass(frozen=True)
class _PackedJob:
  """A shard whose single record is a precomputed token array."""

  out_path: str
  shard: int
  tokens: np.ndarray


def _write_document_shard(job: _DocumentJob) -> ShardStat:
  documents = Corpus(job.stem).read_documents(job.first_doc, job.num_docs)
  return ShardStat(job.shard, _write_tfrecord(job.out_path, documents), "")


def _write_packed_shard(job: _PackedJob) -> ShardStat:
  num_bytes = _write_tfrecord(job.out_path, [job.tokens])
  return ShardStat(job.shard, num_bytes, _md5(job.out_path))


def _run_jobs(
    label: str,
    jobs: Sequence[_DocumentJob | _PackedJob],
    worker: Callable[..., ShardStat],
    num_workers: int,
) -> list[ShardStat]:
  """Runs `worker` over `jobs` and returns the stats sorted by shard."""
  total = len(jobs)
  started = time.monotonic()
  stats: list[ShardStat] = []

  def record(stat: ShardStat) -> None:
    stats.append(stat)
    done = len(stats)
    if done % max(1, total // 20) == 0 or done == total:
      logging.info(
          "%s: %d/%d shards written (%.0f s)",
          label, done, total, time.monotonic() - started,
      )

  if num_workers <= 1:
    for job in jobs:
      record(worker(job))
  else:
    # `spawn`, not `fork`: workers import TensorFlow, which is not fork-safe.
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(
        max_workers=min(num_workers, total), mp_context=context
    ) as pool:
      try:
        pending = [pool.submit(worker, job) for job in jobs]
        for future in futures.as_completed(pending):
          record(future.result())
      except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
  return sorted(stats, key=lambda s: s.shard)


@dataclasses.dataclass(frozen=True)
class SplitInfo:
  """Everything the metadata files need to know about one split."""

  name: str
  shard_lengths: Sequence[int]  # Records per shard, in shard order.
  num_bytes: int  # Sum of the on-disk sizes of the shard files.


def _split_info_json_legacy(split: SplitInfo) -> str:
  """`<split>_info.json` as shipped with the submitted runs' `c4/en/3.0.5`."""
  lengths = ", ".join(f'"{n}"' for n in split.shard_lengths)
  return (
      "{\n"
      f'  "records_written": {sum(split.shard_lengths)},\n'
      '  "empty_records": 0,\n'
      f'  "num_bytes": "{split.num_bytes}",\n'
      f'  "shard_lengths": [{lengths}]\n'
      "}\n"
  )


def _split_info_json_compact(split: SplitInfo) -> str:
  """`<split>_info.json` as shipped with the submitted runs' `c4/en/3.0.9`."""
  return json.dumps({
      "records_written": sum(split.shard_lengths),
      "empty_records": 0,
      "num_bytes": str(split.num_bytes),
      "shard_lengths": [str(n) for n in split.shard_lengths],
  })


def _write_text(directory: str, name: str, text: str) -> None:
  with open(os.path.join(directory, name), "wb") as handle:
    handle.write(text.encode("utf-8"))


def write_metadata(
    out_dir: str,
    version: str,
    description: str,
    config_description: str,
    splits: Sequence[SplitInfo],
    legacy_split_info: bool,
) -> None:
  """Writes `features.json`, `dataset_info.json` and the `<split>_info.json`.

  Only `dataset_info.json` and `features.json` are read by TFDS. The per-split
  info files mirror what the submitted runs use; `c4/en/3.0.5` predates this
  builder and formats them differently from `c4/en/3.0.9`, which
  `legacy_split_info` selects, so that every file is byte-identical to the
  datasets of the submitted runs.
  """
  os.makedirs(out_dir, exist_ok=True)
  _write_text(out_dir, "features.json", json.dumps(FEATURES_JSON, indent=4))
  split_entries = []
  for split in splits:
    split_entries.append({
        "filepathTemplate": FILEPATH_TEMPLATE,
        "name": split.name,
        "numBytes": str(split.num_bytes),
        "shardLengths": [str(n) for n in split.shard_lengths],
    })
    info = (
        _split_info_json_legacy(split)
        if legacy_split_info
        else _split_info_json_compact(split)
    )
    _write_text(out_dir, f"{split.name}_info.json", info)
  _write_text(
      out_dir,
      "dataset_info.json",
      json.dumps(
          {
              "configDescription": config_description,
              "configName": CONFIG,
              "description": description,
              "fileFormat": "tfrecord",
              "name": DATASET,
              "splits": split_entries,
              "version": version,
          },
          indent=2,
      ),
  )


# ----------------------------------------------------------------------------
# Stages.
# ----------------------------------------------------------------------------


def _version_dir(dst_dir: str, version: str) -> str:
  return os.path.join(dst_dir, DATASET, CONFIG, version)


def _build_document_split(
    args: argparse.Namespace, split: str, stem: str, out_dir: str
) -> SplitInfo:
  """Repackages one Megatron corpus 1:1 into `args.num_shards` shards."""
  corpus = Corpus(os.path.join(args.src_dir, stem))
  corpus.check()
  lengths = plan_shard_lengths(corpus.num_docs, args.num_shards)
  logging.info(
      "%s: %d documents -> %d shards (%d..%d records each)",
      split, corpus.num_docs, args.num_shards, min(lengths), max(lengths),
  )
  first_docs = np.cumsum([0] + lengths[:-1]).tolist()
  jobs = [
      _DocumentJob(
          stem=os.path.join(args.src_dir, stem),
          out_path=os.path.join(
              out_dir, shard_filename(split, shard, args.num_shards)
          ),
          shard=shard,
          first_doc=first_docs[shard],
          num_docs=lengths[shard],
      )
      for shard in range(args.num_shards)
  ]
  stats = _run_jobs(split, jobs, _write_document_shard, args.num_workers)
  return SplitInfo(split, lengths, sum(s.num_bytes for s in stats))


def _build_eval(args: argparse.Namespace, out_dir: str) -> SplitInfo:
  """Writes the packed, replicated eval shards plus `_build_stats_validation.json`."""
  corpus = Corpus(os.path.join(args.src_dir, args.validation_stem))
  corpus.check()
  sequences = pack_sequences(
      corpus, args.eval_num_sequences, args.sequence_length, args.eos_id
  )
  payloads = replicate_across_shards(
      sequences,
      args.eval_num_shards,
      args.eval_sequences_per_shard,
      args.reserve_eos_slot,
  )
  logging.info(
      "eval: %d distinct sequences replicated %dx across %d shards of %d",
      len(sequences),
      args.eval_num_shards * args.eval_sequences_per_shard // len(sequences),
      args.eval_num_shards,
      args.eval_sequences_per_shard,
  )
  jobs = [
      _PackedJob(
          out_path=os.path.join(
              out_dir, shard_filename("validation", shard, len(payloads))
          ),
          shard=shard,
          tokens=tokens,
      )
      for shard, tokens in enumerate(payloads)
  ]
  stats = _run_jobs("eval", jobs, _write_packed_shard, args.num_workers)
  # The reference datasets ship this by-product of their distributed build
  # (per-shard size and md5); the line order there is arbitrary, here it is
  # sorted by shard.
  _write_text(
      out_dir,
      "_build_stats_validation.json",
      "".join(
          json.dumps({
              "split": "validation",
              "shard": s.shard,
              "num_bytes": s.num_bytes,
              "md5": s.md5,
          })
          + "\n"
          for s in stats
      ),
  )
  return SplitInfo(
      "validation", [1] * len(payloads), sum(s.num_bytes for s in stats)
  )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
      "--src_dir", required=True,
      help="Directory holding the MLCommons `.bin`/`.idx` files.",
  )
  parser.add_argument(
      "--dst_dir", required=True,
      help="Local TFDS `data_dir` root; shards go to <dst_dir>/c4/en/<version>/.",
  )
  parser.add_argument(
      "--stages", default=",".join(STAGES),
      help="Comma-separated subset of train,validation,eval to build.",
  )
  parser.add_argument(
      "--train_stem", default="c4-train.en_6_text_document",
      help="Basename of the train `.bin`/`.idx` pair.",
  )
  parser.add_argument(
      "--validation_stem", default="c4-validation-91205-samples.en_text_document",
      help="Basename of the validation `.bin`/`.idx` pair.",
  )
  parser.add_argument(
      "--num_shards", type=int, default=2048,
      help="Shards per split for the train/validation stages.",
  )
  parser.add_argument(
      "--version", default="3.0.5",
      help="TFDS version directory for the train/validation stages.",
  )
  parser.add_argument(
      "--eval_version", default="3.0.9",
      help="TFDS version directory for the packed eval stage.",
  )
  parser.add_argument(
      "--eval_num_sequences", type=int, default=1024,
      help="Distinct packed sequences to evaluate on (MLPerf: 1024).",
  )
  parser.add_argument(
      "--eval_num_shards", type=int, default=2048,
      help=(
          "Eval shards; shard count is a multiple of 128 (128 shards x 8"
          " sequences cover the 1024 eval sequences once; 2048 supports up to"
          " 2048 eval data-loading hosts)."
      ),
  )
  parser.add_argument(
      "--eval_sequences_per_shard", type=int, default=8,
      help="Packed sequences each host evaluates (eval pdb x devices).",
  )
  parser.add_argument(
      "--sequence_length", type=int, default=4096,
      help="Packed sequence length (MaxText `max_target_length`).",
  )
  parser.add_argument(
      "--eos_id", type=int, default=128001,
      help="End-of-document token appended between documents when packing.",
  )
  parser.add_argument(
      "--reserve_eos_slot", action="store_true",
      help=(
          "Store one token less per eval shard so the EOS that MaxText appends"
          " lands inside the last sequence instead of in the discarded"
          " remainder. Costs one token per shard but keeps chunking aligned"
          " when hosts != eval_num_shards. Not used for the submitted runs."
      ),
  )
  parser.add_argument(
      "--num_workers", type=int, default=min(32, os.cpu_count() or 1),
      help=(
          "Writer processes (each imports TensorFlow, ~0.5 GB RSS). 1 runs"
          " inline."
      ),
  )
  args = parser.parse_args(argv)
  args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
  unknown = sorted(set(args.stages) - set(STAGES))
  if unknown:
    parser.error(f"unknown stages: {unknown}; choose from {list(STAGES)}")
  if "://" in args.dst_dir:
    parser.error("--dst_dir must be a local path; upload the result afterwards")
  args.src_dir = os.path.abspath(args.src_dir)
  args.dst_dir = os.path.abspath(args.dst_dir)
  return args


def main(argv: Sequence[str] | None = None) -> None:
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
  )
  args = _parse_args(argv)
  started = time.monotonic()

  document_splits = [
      (split, stem)
      for split, stem in (
          ("train", args.train_stem),
          ("validation", args.validation_stem),
      )
      if split in args.stages
  ]
  if document_splits:
    out_dir = _version_dir(args.dst_dir, args.version)
    os.makedirs(out_dir, exist_ok=True)
    split_infos = [
        _build_document_split(args, split, stem, out_dir)
        for split, stem in document_splits
    ]
    write_metadata(
        out_dir,
        args.version,
        DESCRIPTION,
        CONFIG_DESCRIPTION,
        split_infos,
        legacy_split_info=True,
    )
    logging.info("wrote %s", out_dir)

  if "eval" in args.stages:
    out_dir = _version_dir(args.dst_dir, args.eval_version)
    os.makedirs(out_dir, exist_ok=True)
    write_metadata(
        out_dir,
        args.eval_version,
        EVAL_DESCRIPTION,
        EVAL_DESCRIPTION,
        [_build_eval(args, out_dir)],
        legacy_split_info=False,
    )
    logging.info("wrote %s", out_dir)

  logging.info("done in %.0f s", time.monotonic() - started)


if __name__ == "__main__":
  main()
