"""Training batches read from shards."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from . import features, shards
from .moves import MOVE_FIELDS
from .scale import fit_scale, win_probability

__all__ = [
    "SCALE_ROWS",
    "Batch",
    "DataConfig",
    "ShardBatches",
    "fit_scale_on_shards",
]

#: Centipawn labels the value scale is fitted on, where the fit settles.
SCALE_ROWS = 400_000


@dataclass(frozen=True, slots=True)
class Batch:
    """One batch: the typed arrays each head reads, and the labels.

    `facts` is one array per fact, `(b, …)` in the type the fact was declared
    with. `moves` is one array per field of the `move` group, `(b, m)`, padded
    with zeros past a row's legal move count and masked by `move_mask`. `best`
    is the labelled move's index and `value` its win probability.
    """

    facts: dict[str, torch.Tensor]
    moves: dict[str, torch.Tensor]
    move_mask: torch.Tensor
    best: torch.Tensor
    value: torch.Tensor

    def __len__(self) -> int:
        return int(self.value.shape[0])

    def to(self, device: torch.device) -> Batch:
        """The same batch on `device`."""
        return Batch(
            facts={name: array.to(device, non_blocking=True) for name, array in self.facts.items()},
            moves={name: array.to(device, non_blocking=True) for name, array in self.moves.items()},
            move_mask=self.move_mask.to(device, non_blocking=True),
            best=self.best.to(device, non_blocking=True),
            value=self.value.to(device, non_blocking=True),
        )


@dataclass(frozen=True)
class DataConfig:
    """Where the shards are, which facts are read, and how they are batched."""

    shards: Path
    #: The fact arrays the batch carries; `None` is every array the shards hold.
    features: tuple[str, ...] | None = None
    batch_size: int = 256
    seed: int = 0

    def available(self) -> list[str]:
        """The fact arrays the shards carry, read from the first one's manifest."""
        paths = shards.shard_paths(self.shards, "train") or shards.shard_paths(self.shards, "holdout")
        if not paths:
            raise ValueError(f"no shards in {self.shards}")
        return shards.held_names(paths[0])

    def selection(self) -> list[str]:
        """The fact arrays a batch carries.

        Raises `ValueError` for a selected array the shards do not carry.
        """
        held = self.available()
        if self.features is None:
            return held
        missing = [name for name in self.features if name not in set(held)]
        if missing:
            raise ValueError(f"the shards carry no {', '.join(missing)}")
        return list(self.features)


class ShardBatches(IterableDataset[Batch]):
    """Batches of one split's shards.

    Iterating reads every shard of the split once. Shuffling permutes the
    shards and the rows within a shard, so a shard is the shuffling window; the
    build's `--shard-rows` sets how wide that is.
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        scale: float,
        split: str = "train",
        shuffle: bool = True,
    ) -> None:
        self.config = config
        self.scale = scale
        self.split = split
        self.shuffle = shuffle
        self.paths = shards.shard_paths(config.shards, split)
        if not self.paths:
            raise ValueError(f"no {split} shards in {config.shards}")
        self.names = config.selection()

    def __iter__(self) -> Iterator[Batch]:
        seed = self.config.seed if self.split == "train" else self.config.seed + 1
        paths = list(self.paths)
        rng = np.random.default_rng(seed)
        if self.shuffle:
            random.Random(seed).shuffle(paths)
        for path in paths:
            shard = shards.load(path)
            order = np.arange(len(shard))
            if self.shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), self.config.batch_size):
                yield self._batch(shard, order[start : start + self.config.batch_size])

    def _batch(self, shard: shards.Shard, rows: NDArray[np.int64]) -> Batch:
        """The named rows of `shard` collated."""
        starts = shard.cuts[rows].astype(np.int64)
        counts = shard.cuts[rows + 1].astype(np.int64) - starts
        most = int(counts.max())
        # One flat index per padded slot, clamped to the row's own moves; the
        # mask is what tells the padding from a move.
        slots = np.arange(most)
        mask = slots < counts[:, None]
        picked = starts[:, None] + np.where(mask, slots, 0)
        return Batch(
            facts={name: features.as_tensor(shard.facts[name][rows]) for name in self.names},
            moves={field: features.as_tensor(_padded(shard.moves[field], picked, mask)) for field in MOVE_FIELDS},
            move_mask=torch.from_numpy(mask),
            best=torch.from_numpy(shard.best[rows].astype(np.int64)),
            value=torch.from_numpy(win_probability(shard.cp[rows], shard.mate[rows], self.scale)),
        )


def _padded(values: np.ndarray, picked: NDArray[np.int64], mask: NDArray[np.bool_]) -> np.ndarray:
    """The values `picked` names, of the same type, zero where `mask` is false."""
    out = np.zeros(picked.shape, dtype=values.dtype)
    np.copyto(out, values[picked], where=mask)
    return out


def fit_scale_on_shards(directory: Path, *, rows: int = SCALE_ROWS) -> float:
    """The logistic value scale fitted on up to `rows` held-out centipawn labels.

    Only the held-out shards count, so the fit never sees a training label.
    """
    labels: list[NDArray[np.int32]] = []
    taken = 0
    for path in shards.shard_paths(directory, "holdout"):
        shard = shards.load(path)
        labels.append(shard.cp[shard.mate == 0])
        taken += int(labels[-1].shape[0])
        if taken >= rows:
            break
    if not labels:
        raise ValueError(f"no held-out centipawn labels in {directory}")
    return fit_scale(np.concatenate(labels)[:rows])
