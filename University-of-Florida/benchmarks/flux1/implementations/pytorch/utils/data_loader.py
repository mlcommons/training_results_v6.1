# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebDataset-based data loader for pre-encoded Flux.1 latents.

Reads tar files directly using Python tarfile module (no webdataset dependency).
Provides train and validation DataLoader factories with the full MLPerf Flux
data processing pipeline.
"""

import functools
import io
import json
import math
import os
import struct
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# Index the validation sampler emits for padding rows. __getitem__ answers it with a
# zero sample flagged valid=False, so every rank can run the same number of full
# batches without dropping or duplicating a real sample.
PAD_INDEX = -1
_T5_SHAPE = (256, 4096)
_CLIP_SHAPE = (768,)
_LATENT_SHAPE = (16, 32, 32)


def deserialize_numpy_array(data: bytes) -> torch.Tensor:
    """Deserialize uint16 numpy bytes to bfloat16 tensor."""
    buffer = io.BytesIO(data)
    uint16_data = np.load(buffer)
    return torch.from_numpy(uint16_data).view(torch.bfloat16)


class TarShardDataset(Dataset):
    """Dataset that reads pre-encoded samples from tar shards.

    Each sample contains t5, clip, mean, logvar tensors and json metadata.
    Samples are loaded lazily from tar files.

    Args:
        shard_paths: List of paths to tar files.
        empty_t5: Empty T5 encoding for CFG dropout (shape [256, 4096]).
        empty_clip: Empty CLIP encoding for CFG dropout (shape [768]).
        is_validation: If True, use pre-assigned timesteps from metadata.
        cfg_dropout_prob: Probability of replacing embeddings with empty ones.
        vae_scale: VAE normalization scale factor.
        vae_shift: VAE normalization shift factor.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        shard_paths: list[str],
        empty_t5: torch.Tensor,
        empty_clip: torch.Tensor,
        is_validation: bool = False,
        cfg_dropout_prob: float = 0.1,
        vae_scale: float = 0.3611,
        vae_shift: float = 0.1159,
        seed: int = 42,
        samples_per_shard: Optional[int] = None,
    ):
        self.shard_paths = sorted(shard_paths)
        self.empty_t5 = empty_t5  # [256, 4096]
        self.empty_clip = empty_clip  # [768]
        self.is_validation = is_validation
        self.cfg_dropout_prob = cfg_dropout_prob
        self.vae_scale = vae_scale
        self.vae_shift = vae_shift
        self.seed = seed

        # Build index: list of (shard_idx, sample_idx) pairs.
        # Fast approach: probe first shard to get exact count and bytes-per-sample,
        # then estimate other shards from file size. Avoids scanning all 110 tars.
        self.index = []
        self._shard_sample_counts = []

        # Nominal shard size: the number of samples in a "full" shard.
        # Used for computing global sample keys (shard N starts at N * nominal).
        self._nominal_shard_size = samples_per_shard if samples_per_shard else 10000

        if samples_per_shard is not None:
            # All shards same count (caller knows the format)
            for shard_idx in range(len(self.shard_paths)):
                self._shard_sample_counts.append(samples_per_shard)
                for i in range(samples_per_shard):
                    self.index.append((shard_idx, i))
        else:
            # Get sample count from .tar.idx (entries - 1 sentinel)
            for shard_idx, path in enumerate(self.shard_paths):
                idx_path = path + ".idx"
                idx_size = os.path.getsize(idx_path)
                count = idx_size // 8 - 1  # last entry is sentinel
                if shard_idx == 0:
                    self._nominal_shard_size = count
                self._shard_sample_counts.append(count)
                for i in range(count):
                    self.index.append((shard_idx, i))

        # Cache: raw file handles and idx offsets per shard (set in worker_init_fn)
        self._file_handles: dict[int, io.BufferedReader] = {}
        self._shard_offsets: dict[int, list[int]] = {}

    def _get_file_and_offsets(
        self, shard_idx: int,
    ) -> tuple[io.BufferedReader, list[int]]:
        if shard_idx not in self._file_handles:
            self._file_handles[shard_idx] = open(
                self.shard_paths[shard_idx], "rb",
            )
            idx_path = self.shard_paths[shard_idx] + ".idx"
            with open(idx_path, "rb") as f:
                data = f.read()
            n = len(data) // 8
            self._shard_offsets[shard_idx] = list(
                struct.unpack(f"<{n}q", data),
            )
        return self._file_handles[shard_idx], self._shard_offsets[shard_idx]

    @staticmethod
    def _read_tar_entry(fh: io.BufferedReader) -> bytes:
        """Read one tar entry (header + data) at the current file position."""
        header = fh.read(512)
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        content = fh.read(info.size)
        # Skip padding to next 512-byte boundary
        remainder = info.size % 512
        if remainder:
            fh.seek(512 - remainder, 1)
        return content

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx == PAD_INDEX:
            return {
                "t5": torch.zeros(_T5_SHAPE),
                "clip": torch.zeros(_CLIP_SHAPE),
                "mean": torch.zeros(_LATENT_SHAPE),
                "logvar": torch.zeros(_LATENT_SHAPE),
                "timestep": torch.tensor(0, dtype=torch.int32),
                "valid": torch.tensor(False),
            }
        shard_idx, sample_idx = self.index[idx]
        fh, offsets = self._get_file_and_offsets(shard_idx)

        # Seek to sample offset and read 5 consecutive tar entries
        fh.seek(offsets[sample_idx])
        meta = json.loads(self._read_tar_entry(fh))           # .json
        t5 = deserialize_numpy_array(self._read_tar_entry(fh))     # .t5.bytes
        clip = deserialize_numpy_array(self._read_tar_entry(fh))   # .clip.bytes
        mean = deserialize_numpy_array(self._read_tar_entry(fh))   # .mean.bytes
        logvar = deserialize_numpy_array(self._read_tar_entry(fh)) # .logvar.bytes

        timestep = meta.get("timestep", -1)

        return {
            "t5": t5.float(),
            "clip": clip.float(),
            "mean": mean.float(),
            "logvar": logvar.float(),
            "timestep": torch.tensor(timestep, dtype=torch.int32),
            "valid": torch.tensor(True),
        }


def _worker_init_fn(worker_id: int) -> None:
    """Reset file handles in each worker process."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._file_handles = {}
        dataset._shard_offsets = {}


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack VAE latents to patch tokens.

    [B, 16, 32, 32] -> [B, 256, 64]
    2x2 patches: 16 channels * 4 spatial positions = 64 features per patch.
    """
    B, C, H, W = latents.shape
    pH, pW = 2, 2
    nH, nW = H // pH, W // pW
    # [B, C, nH, pH, nW, pW] -> [B, nH, nW, C, pH, pW] -> [B, nH*nW, C*pH*pW]
    latents = latents.reshape(B, C, nH, pH, nW, pW)
    latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(B, nH * nW, C * pH * pW)
    return latents


def unpack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Unpack patch tokens back to spatial format.

    [B, 256, 64] -> [B, 16, 32, 32]
    """
    B, S, D = latents.shape
    C = 16
    pH, pW = 2, 2
    nH = nW = int(math.sqrt(S))
    latents = latents.reshape(B, nH, nW, C, pH, pW)
    latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(B, C, nH * pH, nW * pW)
    return latents


def make_img_ids(batch_size: int, seq_len: int = 256) -> torch.Tensor:
    """Create positional IDs for image tokens.

    Returns [B, seq_len, 3] with (time_id=0, height_id, width_id).
    For 32x32 -> 16x16 patches: positions are (0, h, w) for h,w in [0,16).
    """
    h = w = int(math.sqrt(seq_len))
    img_ids = torch.zeros(batch_size, seq_len, 3)
    row_ids = torch.arange(h).unsqueeze(1).expand(h, w).reshape(-1).float()
    col_ids = torch.arange(w).unsqueeze(0).expand(h, w).reshape(-1).float()
    img_ids[:, :, 1] = row_ids.unsqueeze(0)
    img_ids[:, :, 2] = col_ids.unsqueeze(0)
    return img_ids


def make_txt_ids(batch_size: int, seq_len: int = 256) -> torch.Tensor:
    """Create positional IDs for text tokens. All zeros [B, seq_len, 3]."""
    return torch.zeros(batch_size, seq_len, 3)


def collate_and_process(
    batch: list[dict],
    empty_t5: torch.Tensor,
    empty_clip: torch.Tensor,
    is_validation: bool,
    cfg_dropout_prob: float,
    vae_scale: float,
    vae_shift: float,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    """Collate batch and apply full data processing pipeline.

    Returns dict with keys: img, img_ids, txt, txt_ids, timesteps, y, target
    """
    B = len(batch)
    mean = torch.stack([b["mean"] for b in batch])  # [B, 16, 32, 32]
    logvar = torch.stack([b["logvar"] for b in batch])  # [B, 16, 32, 32]
    t5 = torch.stack([b["t5"] for b in batch])  # [B, 256, 4096]
    clip = torch.stack([b["clip"] for b in batch])  # [B, 768]
    # 1.0 for real samples, 0.0 for PAD_INDEX rows (validation padding).
    valid = torch.stack([b["valid"] for b in batch]).float()  # [B]

    # Step 1: VAE reparameterization
    noise_vae = torch.randn_like(mean, generator=generator)
    latents = mean + torch.exp(0.5 * logvar) * noise_vae  # [B, 16, 32, 32]

    # Step 2: VAE normalization
    latents = vae_scale * (latents - vae_shift)

    # Step 3: Sample timesteps
    if is_validation:
        timestep_ints = torch.tensor(
            [
                b["timestep"].item()
                if isinstance(b["timestep"], torch.Tensor)
                else b["timestep"]
                for b in batch
            ],
            dtype=torch.float32,
        )
        timesteps = timestep_ints / 8.0
    else:
        timesteps = torch.rand(B, generator=generator)

    # Step 4: Create noisy input and target
    noise = torch.randn_like(latents, generator=generator)
    t_expand = timesteps.view(B, 1, 1, 1)
    noisy = (1.0 - t_expand) * latents + t_expand * noise
    target = noise - latents  # [B, 16, 32, 32]

    # Step 5: Pack latents to patches
    img = pack_latents(noisy)  # [B, 256, 64]
    target_packed = pack_latents(target)  # [B, 256, 64]

    # Step 6: Create positional encodings
    img_ids = make_img_ids(B)  # [B, 256, 3]
    txt_ids = make_txt_ids(B)  # [B, 256, 3]

    # Step 7: Classifier-free guidance dropout (training only)
    if not is_validation and cfg_dropout_prob > 0:
        drop_mask = torch.rand(B, generator=generator) < cfg_dropout_prob
        for i in range(B):
            if drop_mask[i]:
                t5[i] = empty_t5
                clip[i] = empty_clip

    return {
        "img": img.bfloat16(),
        "img_ids": img_ids.bfloat16(),
        "txt": t5.bfloat16(),
        "txt_ids": txt_ids.bfloat16(),
        "timesteps": timesteps.bfloat16(),
        "y": clip.bfloat16(),
        "target": target_packed.bfloat16(),
        "valid": valid,
    }


def process_batch_tensors(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    t5: torch.Tensor,
    clip: torch.Tensor,
    timestep_ints: torch.Tensor,
    empty_t5: torch.Tensor,
    empty_clip: torch.Tensor,
    is_validation: bool,
    cfg_dropout_prob: float,
    vae_scale: float,
    vae_shift: float,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    """Process a collated batch of tensors through the full pipeline.

    This function takes already-batched tensors (from DataLoader default collate)
    and applies VAE reparam, normalization, noise, packing, and CFG dropout.

    Args:
        mean: [B, 16, 32, 32] float32
        logvar: [B, 16, 32, 32] float32
        t5: [B, 256, 4096] float32
        clip: [B, 768] float32
        timestep_ints: [B] int32, -1 for training (random), 0-7 for val
        empty_t5: [256, 4096] float32
        empty_clip: [768] float32
    """
    B = mean.shape[0]

    # VAE reparameterization
    noise_vae = torch.randn_like(mean, generator=generator)
    latents = mean + torch.exp(0.5 * logvar) * noise_vae

    # VAE normalization
    latents = vae_scale * (latents - vae_shift)

    # Timesteps
    if is_validation:
        timesteps = timestep_ints.float() / 8.0
    else:
        timesteps = torch.rand(B, generator=generator)

    # Noisy input and target
    noise = torch.randn_like(latents, generator=generator)
    t_expand = timesteps.view(B, 1, 1, 1)
    noisy = (1.0 - t_expand) * latents + t_expand * noise
    target = noise - latents

    # Pack to patches
    img = pack_latents(noisy)
    target_packed = pack_latents(target)

    # Positional encodings
    img_ids = make_img_ids(B)
    txt_ids = make_txt_ids(B)

    # CFG dropout (training only)
    if not is_validation and cfg_dropout_prob > 0:
        drop_mask = torch.rand(B, generator=generator) < cfg_dropout_prob
        for i in range(B):
            if drop_mask[i]:
                t5[i] = empty_t5
                clip[i] = empty_clip

    return {
        "img": img.bfloat16(),
        "img_ids": img_ids.bfloat16(),
        "txt": t5.bfloat16(),
        "txt_ids": txt_ids.bfloat16(),
        "timesteps": timesteps.bfloat16(),
        "y": clip.bfloat16(),
        "target": target_packed.bfloat16(),
    }


def load_empty_encodings(
    dataset_root: str = "/dataset",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load empty T5 and CLIP encodings for CFG dropout.

    Returns (empty_t5 [256, 4096], empty_clip [768]) as float32.
    """
    t5_empty = np.load(os.path.join(dataset_root, "empty_encodings", "t5_empty.npy"))
    clip_empty = np.load(
        os.path.join(dataset_root, "empty_encodings", "clip_empty.npy")
    )
    # These are float16, shape [1, 256, 4096] and [1, 768] -- squeeze batch dim
    empty_t5 = torch.from_numpy(t5_empty.astype(np.float32)).squeeze(0)  # [256, 4096]
    empty_clip = torch.from_numpy(clip_empty.astype(np.float32)).squeeze(0)  # [768]
    return empty_t5, empty_clip


def get_shard_paths(split: str, dataset_root: str = "/dataset") -> list[str]:
    """Get sorted list of tar shard paths for a given split."""
    split_dir = os.path.join(dataset_root, split)
    paths = sorted(
        str(p) for p in Path(split_dir).glob("shard_*.tar")
    )
    return paths


def build_train_loader(
    dataset_root: str = "/dataset",
    micro_batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    cfg_dropout_prob: float = 0.1,
    vae_scale: float = 0.3611,
    vae_shift: float = 0.1159,
    sampler_seed: Optional[int] = None,
) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """Build training DataLoader.

    ``seed`` feeds the per-rank RNG (noise, timesteps, CFG mask) and may differ per
    rank. ``sampler_seed`` drives the DistributedSampler permutation and must be
    identical on every rank, otherwise the per-rank shards are not a partition of
    the dataset. Defaults to ``seed`` for single-process use.

    Returns (dataloader, empty_t5, empty_clip).
    """
    if sampler_seed is None:
        sampler_seed = seed
    empty_t5, empty_clip = load_empty_encodings(dataset_root)
    shard_paths = get_shard_paths("train", dataset_root)

    dataset = TarShardDataset(
        shard_paths=shard_paths,
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=False,
        cfg_dropout_prob=cfg_dropout_prob,
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        seed=seed,
    )

    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset, shuffle=True, seed=sampler_seed, drop_last=True
        )

    # process_batch_tensors runs in worker processes via collate_fn so
    # the main process receives fully-processed bf16 tensors that
    # DataLoader can pin in-place; per-step H2D then issues async via
    # non_blocking=True and overlaps with the graph replay. Avoids the
    # main-thread / ThreadPoolExecutor GIL contention path.
    collate_fn = functools.partial(
        collate_and_process,
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=False,
        cfg_dropout_prob=cfg_dropout_prob,
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        generator=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    return loader, empty_t5, empty_clip


class _ContiguousShardSampler(torch.utils.data.Sampler):
    """Contiguous per-rank chunk, padded to a rank-uniform number of full batches.

    The split matches HuggingFace ``Dataset.shard(contiguous=True)``: the first
    ``total % world_size`` ranks each get one extra sample and every sample is
    yielded exactly once across ranks. The chunk is then padded with PAD_INDEX up
    to ``n_batches * batch_size`` where
    ``n_batches = ceil(ceil(total / world_size) / batch_size)`` is the same on
    every rank. The eval forward contains collectives, so all ranks must run the
    same number of batches, and no real sample may be dropped to get there.
    """

    def __init__(self, total: int, rank: int, world_size: int, batch_size: int):
        div, mod = divmod(total, world_size)
        self._start = div * rank + min(rank, mod)
        self._end = self._start + div + (1 if rank < mod else 0)
        n_batches = math.ceil(math.ceil(total / world_size) / batch_size)
        self._padded_len = n_batches * batch_size
        assert self._padded_len >= self._end - self._start

    @property
    def num_valid(self) -> int:
        return self._end - self._start

    def __iter__(self):
        real = list(range(self._start, self._end))
        return iter(real + [PAD_INDEX] * (self._padded_len - len(real)))

    def __len__(self):
        return self._padded_len


def build_val_loader(
    dataset_root: str = "/dataset",
    micro_batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    vae_scale: float = 0.3611,
    vae_shift: float = 0.1159,
) -> DataLoader:
    """Build validation DataLoader."""
    empty_t5, empty_clip = load_empty_encodings(dataset_root)
    shard_paths = get_shard_paths("val", dataset_root)

    dataset = TarShardDataset(
        shard_paths=shard_paths,
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=True,
        cfg_dropout_prob=0.0,
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        seed=seed,
    )

    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    sampler = _ContiguousShardSampler(
        len(dataset), rank, world_size, micro_batch_size
    )

    collate_fn = functools.partial(
        collate_and_process,
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=True,
        cfg_dropout_prob=0.0,
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        generator=None,
    )

    # Every batch is full by construction (the sampler pads), so nothing is
    # dropped: the padded rows carry valid=0 and are masked out of the loss.
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    return loader


if __name__ == "__main__":
    # Quick test: load a few samples and verify shapes
    print("Loading empty encodings...")
    empty_t5, empty_clip = load_empty_encodings()
    print(f"  empty_t5: {empty_t5.shape}, empty_clip: {empty_clip.shape}")

    print("\nLoading train dataset...")
    shard_paths = get_shard_paths("train")
    print(f"  Found {len(shard_paths)} train shards")

    dataset = TarShardDataset(
        shard_paths=shard_paths[:1],  # Just first shard for speed
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=False,
        samples_per_shard=10000,
    )
    print(f"  Dataset size (1 shard): {len(dataset)}")

    sample = dataset[0]
    print(f"  t5: {sample['t5'].shape}")
    print(f"  clip: {sample['clip'].shape}")
    print(f"  mean: {sample['mean'].shape}")
    print(f"  logvar: {sample['logvar'].shape}")
    print(f"  timestep: {sample['timestep']} (type: {type(sample['timestep'])})")

    print("\nProcessing a mini-batch of 4 using process_batch_tensors...")
    batch = [dataset[i] for i in range(4)]
    # Simulate DataLoader default collation
    collated = {
        k: torch.stack([b[k] for b in batch]) for k in batch[0]
    }
    gen = torch.Generator().manual_seed(42)
    processed = process_batch_tensors(
        mean=collated["mean"],
        logvar=collated["logvar"],
        t5=collated["t5"],
        clip=collated["clip"],
        timestep_ints=collated["timestep"],
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=False,
        cfg_dropout_prob=0.1,
        vae_scale=0.3611,
        vae_shift=0.1159,
        generator=gen,
    )
    for k, v in processed.items():
        print(f"  {k}: {v.shape}, {v.dtype}")

    print("\nLoading val dataset...")
    val_paths = get_shard_paths("val")
    print(f"  Found {len(val_paths)} val shards")
    val_dataset = TarShardDataset(
        shard_paths=val_paths[:1],
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=True,
        samples_per_shard=10000,
    )
    val_sample = val_dataset[0]
    print(f"  Val timestep: {val_sample['timestep']}")

    val_batch = [val_dataset[i] for i in range(4)]
    val_collated = {
        k: torch.stack([b[k] for b in val_batch]) for k in val_batch[0]
    }
    val_processed = process_batch_tensors(
        mean=val_collated["mean"],
        logvar=val_collated["logvar"],
        t5=val_collated["t5"],
        clip=val_collated["clip"],
        timestep_ints=val_collated["timestep"],
        empty_t5=empty_t5,
        empty_clip=empty_clip,
        is_validation=True,
        cfg_dropout_prob=0.0,
        vae_scale=0.3611,
        vae_shift=0.1159,
    )
    for k, v in val_processed.items():
        print(f"  {k}: {v.shape}, {v.dtype}")

    print("\nData loader test PASSED")
