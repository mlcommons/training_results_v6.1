"""Evaluation computes only the positions the validation loss reads.

Each validation row is one 8192-token document whose loss covers only its summary (about 1,070 tokens, ending on
average at position ~5,700). With causal attention, positions after a row's last loss token cannot change any logit
the loss reads, so they are cut: the row is kept up to that token, rounded up to a multiple of
TRAINLOOM_EVAL_TRIM_BUCKET (default 512) so only a few distinct shapes reach the kernels. Rows are then served longest
first, so the 8 ranks of one eval step get rows of similar length; an eval step lasts as long as its slowest rank.
Padding slots of the last partial step (index < 0, all-zero loss mask) shrink to one bucket.

TE's Triton cast/transpose and RMSNorm kernels are autotuned on the token count, so every trimmed length would be
a first call that benchmarks all configs inside the eval forward; that produced NaN in the last layers on random
ranks (a sweep of 480 new lengths: 30 non-finite events with the autotuning, none without). Here a token count those
kernels have not seen reuses the config already tuned for the same row width (the training shapes, tuned in the
untimed warmup), or the kernel's first config, and never benchmarks. The cast is elementwise, so its output does not
depend on the config; the training shapes keep their tuned configs.

The validation loss is unchanged as a real-valued function: the cut positions are dead computation, and with
`sample_weight: constant` the loss is the mean of per-row means, which does not depend on the order of rows.

Loaded through TRAINLOOM_PY_PATCHES. Patches MegatronGPTSFTModel.build_train_valid_test_datasets to wrap the
validation dataset.
"""

import os

import numpy as np
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTPackedDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from transformer_engine.pytorch.triton_kernels import cast_transpose as _ct
from transformer_engine.pytorch.triton_kernels import rmsnorm as _rms

BUCKET = int(os.environ.get("TRAINLOOM_EVAL_TRIM_BUCKET", "512"))


def kept_tokens(input_ids, loss_mask, bucket=BUCKET):
    """Tokens of a single-document row to keep: the model sees tokens[:n-1] and predicts tokens[1:n]; the loss reads
    loss_mask[:len-1] (the collate drops the last position), so n - 1 must cover its last nonzero entry."""
    n = len(input_ids)
    hits = np.flatnonzero(np.asarray(loss_mask[: n - 1]))
    needed = hits[-1] + 1 if len(hits) else 1
    return min(n, -(-needed // bucket) * bucket + 1)


class TrimmedEvalDataset(GPTSFTPackedDataset):
    """A GPTSFTPackedDataset whose rows are cut after their last loss token and served longest first."""

    def trim(self):
        rows = [self.indexed_dataset[i] for i in range(len(self.indexed_dataset))]
        assert all(list(r["seq_start_id"]) == [0] for r in rows), "eval_trim expects one document per row"
        self._kept = np.array([kept_tokens(r["input_ids"], r["loss_mask"]) for r in rows])
        self._order = np.argsort(-self._kept, kind="stable")
        assert self.samples_mapping is None

    def __getitem__(self, idx):
        if idx < 0:
            item, n = super().__getitem__(idx), BUCKET + 1
        else:
            row = int(self._order[idx])
            item, n = super().__getitem__(row), int(self._kept[row])
        return {"input_ids": item["input_ids"][:n], "seq_boundaries": [0, n], "loss_mask": item["loss_mask"][:n]}


def _build_with_trim(self, stage):
    _build(self, stage)
    ds = self._validation_ds
    for d in ds if isinstance(ds, list) else [ds]:
        assert type(d) is GPTSFTPackedDataset, type(d)
        d.__class__ = TrimmedEvalDataset
        d.trim()


_build = MegatronGPTSFTModel.build_train_valid_test_datasets
MegatronGPTSFTModel.build_train_valid_test_datasets = _build_with_trim


def _never_benchmark_new_rows(tuner):
    """A key unseen by this triton.Autotuner (its first entry is the row count) takes the config tuned for the
    same remaining key (row width, dtypes), else the first config, instead of benchmarking. Mirrors the key that
    Autotuner.run builds (Triton 3.4)."""
    run = tuner.run

    def run_no_new_benchmark(*args, **kwargs):
        named = {k: v for k, v in {**dict(zip(tuner.arg_names, args)), **kwargs}.items() if k in tuner.arg_names}
        key = [named[k] for k in tuner.keys if k in named]
        key += [str(v.dtype) for v in named.values() if hasattr(v, "dtype")]
        key = tuple(key)
        if len(tuner.configs) > 1 and key not in tuner.cache:
            same_width = [c for k, c in tuner.cache.items() if k[1:] == key[1:]]
            tuner.cache[key] = same_width[0] if same_width else tuner.configs[0]
        return run(*args, **kwargs)

    tuner.run = run_no_new_benchmark


for _tuner in (_ct._cast_transpose_triton, _ct._transpose_triton_dbias, _ct._reduce_bias_triton, _rms._rmsnorm_fwd_triton):
    _never_benchmark_new_rows(_tuner)
