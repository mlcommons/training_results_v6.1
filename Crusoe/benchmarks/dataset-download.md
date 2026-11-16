# Datasets: how each one got onto the cluster

All five benchmarks' datasets were downloaded and prepared using the scripts shipped inside each benchmark's own NGC container, run once, then reused across every scale/seed for that benchmark. Nothing here is submitter-authored preprocessing logic — only the destination paths and the container invocation are ours.

| Benchmark | Dataset | Size on disk | Location |
|---|---|---|---|
| llama31_8b | C4 (preprocessed, 8B-tokenizer variant) | ~84 GB | `/data/mlperf/llama31_8b/data` |
| llama2_70b_lora | GovReport | small (fine-tuning set) | `/data/mlperf/llama2_70b_lora/data/gov_report` + model checkpoint under `/data/mlperf/llama2_70b_lora/model` |
| gpt_oss_20b | C4 (reuses the llama31_8b preprocessed set) | shared with above | `/data/mlperf/llama31_8b/data`, `DATA_FOLDER=8b` |
| flux1 | Energon-converted image/text shards (CC12M-derived) | large (multi-hundred-GB) | `/data/mlperf/flux1/data/energon/{train,val,empty_encodings}` |
| dlrmv4 | Yambda-5B (Yandex Music, real listening events) | 114 GB processed | `/data/mlperf/dlrmv4/data` |

## llama31_8b — C4

Downloaded via the container's own dataset script into `/data/mlperf/llama31_8b/data`. Verified against the container's own `llama-3-1-8b-preprocessed-c4-dataset.md5` — all files OK. Layout:

```
8b/
  LICENSE.txt  NOTICE.txt
  c4-train.en_6_text_document.{bin,idx}
  c4-validation-91205-samples.en_text_document.{bin,idx}
  tokenizer/
```

## llama2_70b_lora — GovReport

Small fine-tuning dataset plus a starting checkpoint (LoRA fine-tunes an already-trained 70B model). Checkpoint layout verified against the container's own `expected-mounts.csv`:

```
model/iter_0000000/{__0_0.distcp,__0_1.distcp,common.pt,metadata.json,tokenizer/...}
```

## gpt_oss_20b — reuses the llama31_8b C4 set

No separate download — `DATADIR` points at the same `llama31_8b/data` directory, `DATA_FOLDER=8b` selects the same preprocessed shard.

## flux1 — Energon-converted CC12M-derived shards

Prepared via the container's `energon_prepare.py`. Final layout:

```
data/energon/
  train/   (110 shards)
  val/     (3 shards)
  empty_encodings/
```

## dlrmv4 — Yambda-5B

Real, publicly-released Yandex Music dataset (Apache-2.0, HuggingFace `yandex/yambda`): 4.79 billion listening events from 1M users, 9.39M tracks. Downloaded and preprocessed via the container's own `generative_recommenders.dlrm_v4.preprocess_public_data` module:

```bash
export DATADIR=/data/mlperf/dlrmv4/data
export HF_HOME=/data/mlperf/dlrmv4/hf_cache   # kept off $HOME - the raw download briefly needs 200GB+ here
docker run --rm --user $(id -u):$(id -g) --network=host \
  -v /data/mlperf/dlrmv4/data:/data/mlperf/dlrmv4/data \
  -v /data/mlperf/dlrmv4/hf_cache:/data/mlperf/dlrmv4/hf_cache \
  -e DATADIR=/data/mlperf/dlrmv4/data -e HF_HOME=/data/mlperf/dlrmv4/hf_cache -e HOME=/tmp \
  nvcr.io/nvdlfwea/mlperftv61/dlrmv4-arm:20260921 \
  bash /workspace/dlrmv4/download_dataset.sh
```

Final layout (verified against the container's own `verify_dataset.sh` — all 7 expected files present; its bundled md5 file ships with placeholder hashes since this is a new benchmark, so verification is layout-only):

```
raw/5b/multi_event.parquet
shared_metadata/{artist_item_mapping,album_item_mapping,embeddings}.parquet
processed_5b/{train_sessions,test_events,session_index}.parquet
processed_5b/item_popularity.npy, processed_5b/split_meta.json
processed_5b/hstu_cache/   (built on first training run, ~114GB, reused across all NEXP/seeds)
```

The `hstu_cache/` is a one-time, expensive build (flattens all 4.79B events into mmap-able arrays, ~150-250GB peak RAM). It is scale-independent — build it once at any scale and every later run at any GPU count reuses it directly. Building it concurrently with a large multi-node job's own model initialization caused a reproducible OOM (see main README, "Known issues"); pre-building it via a small job first avoids that entirely.
