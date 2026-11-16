"""eval_trim against the stock validation rows, through NeMo's own collate_fn (CPU, no GPU needed).

Run inside the training image with the data mounted at /data:
    cd /workspace/trainloom_patches && python check_eval_trim.py
For every validation row: the trimmed batch's tokens, labels and loss mask equal the stock batch's on the kept
positions, the stock loss mask is all zero on the cut positions, and the per-row loss token count is unchanged. Also
checks that every row is served exactly once and prints the per-step eval cost estimate.
"""

import sys
from types import SimpleNamespace

import numpy as np

import eval_trim as et


def dataset(cls):
    d = cls.__new__(cls)
    d.indexed_dataset = np.load("/data/validation.npy", allow_pickle=True)
    d.samples_mapping = None
    d.tokenizer = SimpleNamespace(eos_id=2)
    d.answer_only_loss, d.pad_to_max_length, d.max_seq_length, d.pad_seq_length_to_mult = True, False, 8192, 16
    d.return_cu_seqlen, d.pad_cu_seqlens, d.ceil_to_power_2 = False, False, False
    return d


def main():
    stock, trimmed = dataset(et.GPTSFTPackedDataset), dataset(et.TrimmedEvalDataset)
    trimmed.trim()
    n = len(stock.indexed_dataset)
    assert sorted(int(trimmed._order[i]) for i in range(n)) == list(range(n)), "rows not served exactly once"
    bad = 0
    for i in range(n):
        row = int(trimmed._order[i])
        a, b = stock.collate_fn([stock[row]]), trimmed.collate_fn([trimmed[i]])
        k = b["tokens"].shape[1]
        same = all(np.array_equal(a[x][:, :k], b[x][:, :k]) for x in ("tokens", "labels", "position_ids"))
        same &= np.array_equal(a["loss_mask"][:, :k], b["loss_mask"])
        same &= int(a["loss_mask"][:, k:].sum()) == 0
        bad += not same
    pad = trimmed.collate_fn([trimmed[-1]])
    assert int(pad["loss_mask"].sum()) == 0 and pad["tokens"].shape[1] == et.BUCKET
    lengths = np.array([trimmed.collate_fn([trimmed[i]])["tokens"].shape[1] for i in range(n)])
    cost = 0.81 * lengths / 8192 + 0.19 * (lengths / 8192) ** 2
    steps = [cost[i:i + 8].max() for i in range(0, n, 8)]
    print(f"rows {n}: {n - bad} identical on every position the loss reads, {bad} different")
    print(f"distinct lengths {sorted(set(lengths.tolist()))}")
    print(f"estimated eval time vs untrimmed: {np.sum(steps) / len(steps):.3f}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
