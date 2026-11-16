# MLPerf DeepSeek-V3: rebuilding the C4 datasets

Standalone scripts that rebuild, from the public MLCommons files, the two C4
datasets the DeepSeek-V3 MLPerf Training submission reads through TFDS. They
need only Python 3.10+, `numpy` and `tensorflow` (CPU is enough).

| TFDS dataset | split | content | shards |
|---|---|---|---|
| `c4/en:<version>` | `train` | the MLCommons train corpus (45,608,611 documents) | 16384 (2048 before `split_shards.py`) |
| `c4/en:<eval_version>` | `validation` | the MLPerf eval set: first 1024 packed 4096-sequences of the validation corpus | 2048 |

`<version>` and `<eval_version>` are set by `build_dataset.py --version` and
`--eval_version`; the defaults match `dataset_name` and `eval_dataset_name` in
`config.yml`. Every record is a `tf.train.Example` with a single `ids` feature
(a `tf.train.Int64List` of token ids, decoded as `int32` per `features.json`).

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | MLCommons `.bin`/`.idx` to the TFDS datasets above. |
| `split_shards.py` | Reshards the 2048 train shards to 16384 at record boundaries. |
| `requirements.txt` | Python dependencies. |

## Quick start

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt

DATA=/path/with/200GB/free
# 1. download the corpus (see below), then
python build_dataset.py --src_dir=$DATA/mlperf_c4 --dst_dir=$DATA/dsv3_c4 --stages=train,eval --num_workers=32
# 2. split train shards to the 16384-shard layout used by the submission recipes
python split_shards.py  --src_dir=$DATA/dsv3_c4 --dst_dir=$DATA/dsv3_c4_16384 --num_workers=32
```

The submission recipes point MaxText at the result with:

```
dataset_type=c4_mlperf dataset_path=$DATA/dsv3_c4_16384/
dataset_name=c4/en:<version> eval_dataset_name=c4/en:<eval_version>
train_data_columns=[ids] eval_data_columns=[ids]
tokenize_train_data=False tokenize_eval_data=False
```

`dataset_path` may also be a `gs://` copy of the output.

## Step 0: download the MLCommons corpus

The corpus is MLCommons' public "Llama 3.1 8B preprocessed C4 dataset". Only
the four `c4-*` files are needed (about 85 GB).

```bash
mkdir -p $DATA/mlperf_c4
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d $DATA/mlperf_c4 \
    https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
```

or fetch the files directly and verify them:

```bash
BASE=https://training.mlcommons-storage.org/llama3_1/datasets/c4/llama3_1_8b
cd $DATA/mlperf_c4
for f in c4-validation-91205-samples.en_text_document.{bin,idx} \
         c4-train.en_6_text_document.{bin,idx}; do
  curl -fL -C - -O $BASE/$f
done
curl -s https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.md5 \
  | md5sum -c --ignore-missing -
```

| File | md5 |
|---|---|
| `c4-train.en_6_text_document.bin` | `c781dc1a0eeb5a0322ad3b528e25027e` |
| `c4-train.en_6_text_document.idx` | `78b6262d170c7c9839253fdb4f944057` |
| `c4-validation-91205-samples.en_text_document.bin` | `85944abe86263a9a12f283b34af10a2b` |
| `c4-validation-91205-samples.en_text_document.idx` | `32d3afd6f81d320ca88c6675c27cf26f` |

## Step 1: build

```bash
python build_dataset.py --src_dir=$DATA/mlperf_c4 --dst_dir=$DATA/dsv3_c4 --stages=train,eval --num_workers=32
```

Writes the two datasets under `<dst_dir>/c4/en/<version>/` and
`<dst_dir>/c4/en/<eval_version>/` (TFRecord shards plus TFDS metadata). The
train split has one record per document, in corpus order, split contiguously
over 2048 shards.

## Step 2: 16384 train shards

The submission recipes read the train split with `train_shard_in_read=True`,
which assigns files round-robin to data-loading hosts, so they use a copy with
8x more, 8x smaller shards:

```bash
python split_shards.py --src_dir=$DATA/dsv3_c4 --dst_dir=$DATA/dsv3_c4_16384 --num_workers=32
```