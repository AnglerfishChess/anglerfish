"""Training batches streamed from the Lichess evaluation dump."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import esca
import numpy as np
import torch
from esca.lichess import Batch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from .moves import move_index
from .scale import fit_scale, win_probability

__all__ = [
    "SCALE_ROWS",
    "Counts",
    "DataConfig",
    "EvalBatches",
    "Sample",
    "Split",
    "collate",
    "fit_scale_on_dump",
    "holdout_keys",
    "position_key",
    "resolve_groups",
]

#: Which side of the deterministic record-index split a row belongs to.
Split = Literal["train", "holdout"]

#: Centipawn labels the value scale is fitted on, where the fit settles.
SCALE_ROWS = 400_000


def resolve_groups(groups: tuple[str, ...] | None) -> tuple[str, ...]:
    """The named schema groups, or every group of the v0 schema."""
    if groups is None:
        return tuple(esca.SCHEMA.group_names)
    unknown = [name for name in groups if name not in esca.SCHEMA.group_names]
    if unknown:
        raise ValueError(f"not schema groups: {', '.join(unknown)}")
    return tuple(groups)


@dataclass(frozen=True)
class DataConfig:
    """Where the rows come from and how they are split, shuffled and cut."""

    dump: Path
    #: Schema groups to encode, in schema order; `None` is every group.
    groups: tuple[str, ...] | None = None
    #: A record contributes a row only if some evaluation reaches this depth.
    min_depth: int = 20
    #: One record index in this many goes to the held-out split.
    holdout_every: int = 64
    #: Rows per collated training batch.
    batch_size: int = 256
    #: Rows held for shuffling before one is handed out.
    shuffle_buffer: int = 4096
    #: Rows the dump reader gathers and encodes at a time.
    read_batch: int = 4096
    #: Cap on record indices read from the dump, both splits together.
    max_rows: int | None = None
    seed: int = 0

    @property
    def group_list(self) -> tuple[str, ...]:
        return resolve_groups(self.groups)

    @property
    def width(self) -> int:
        """Values per position row under this group selection."""
        return esca.SCHEMA.width_of(list(self.group_list))


@dataclass(frozen=True, slots=True)
class Sample:
    """One position: its features, its legal moves' features and its labels."""

    features: NDArray[np.float32]
    moves: NDArray[np.float32]
    best: int
    value: float


def _dump_batches(config: DataConfig) -> Iterator[tuple[int, Batch]]:
    """Each dump batch with the record index of its first row."""
    index = 0
    stream = esca.lichess.batches(
        config.dump,
        batch_size=config.read_batch,
        min_depth=config.min_depth,
        groups=list(config.group_list),
    )
    for batch in stream:
        if config.max_rows is not None and index >= config.max_rows:
            return
        yield index, batch
        index += len(batch)


def _wanted(index: int, config: DataConfig, split: Split) -> bool:
    holdout = index % config.holdout_every == 0
    return holdout if split == "holdout" else not holdout


def _batch_stop(start: int, size: int, config: DataConfig) -> int:
    """How many rows of a batch starting at `start` are within `max_rows`."""
    if config.max_rows is None:
        return size
    return max(0, min(size, config.max_rows - start))


def position_key(fen: str) -> str:
    """The position a FEN names, without its clocks.

    Placement, side to move, castling rights and en-passant square, joined by
    spaces. Two FENs share a key exactly when they are the same position at
    possibly different move counts.
    """
    return " ".join(fen.split(" ", 4)[:4])


def holdout_keys(config: DataConfig) -> frozenset[str]:
    """The `position_key`s of every held-out candidate of the dump."""
    keys: set[str] = set()
    for start, batch in _dump_batches(config):
        for row in range(_batch_stop(start, len(batch), config)):
            if _wanted(start + row, config, "holdout"):
                keys.add(position_key(batch.fens[row]))
    return frozenset(keys)


def _shuffled(samples: Iterator[Sample], size: int, rng: random.Random) -> Iterator[Sample]:
    if size <= 1:
        yield from samples
        return
    buffer: list[Sample] = []
    for sample in samples:
        if len(buffer) < size:
            buffer.append(sample)
            continue
        slot = rng.randrange(size)
        yield buffer[slot]
        buffer[slot] = sample
    rng.shuffle(buffer)
    yield from buffer


def collate(samples: list[Sample]) -> dict[str, torch.Tensor]:
    """One batch of samples as padded tensors.

    Keys are `features` (b, w), `moves` (b, m, `esca.MOVE_WIDTH`), `move_mask`
    (b, m) of bool, `best` (b,) of int64 and `value` (b,). Move rows beyond a
    position's legal move count are zero and masked out.
    """
    if not samples:
        raise ValueError("a batch holds at least one sample")
    most = max(sample.moves.shape[0] for sample in samples)
    moves = np.zeros((len(samples), most, esca.MOVE_WIDTH), dtype=np.float32)
    mask = np.zeros((len(samples), most), dtype=bool)
    for row, sample in enumerate(samples):
        count = sample.moves.shape[0]
        moves[row, :count] = sample.moves
        mask[row, :count] = True
    return {
        "features": torch.from_numpy(np.stack([sample.features for sample in samples])),
        "moves": torch.from_numpy(moves),
        "move_mask": torch.from_numpy(mask),
        "best": torch.tensor([sample.best for sample in samples], dtype=torch.int64),
        "value": torch.tensor([sample.value for sample in samples], dtype=torch.float32),
    }


@dataclass
class Counts:
    """How many rows the dump offered, and what became of them.

    `read` counts the split's candidates by index. `duplicate`, `leaked` and
    `unmatched` are the ones dropped and `kept` the ones handed out; they add
    up to `read` over a stream read to its end.
    """

    read: int = 0
    kept: int = 0
    #: The labelled best move was not among the legal moves.
    unmatched: int = 0
    #: The position was already held out under an earlier index.
    duplicate: int = 0
    #: A training candidate whose position is in the held-out set.
    leaked: int = 0


class EvalBatches(IterableDataset[dict[str, torch.Tensor]]):
    """Collated batches of one split of an evaluation dump.

    Iterating restarts at the head of the dump; which split a record is a
    candidate for is a function of its index in the dump reader's output, so it
    is the same for every pass over the same file at the same `min_depth`.

    The held-out split keeps a candidate only the first time its `position_key`
    is seen; the training split drops every candidate whose key is held out.
    That key set is `holdout`, or one read from the dump on first use.

    A row whose labelled best move is not among the position's legal moves is
    dropped too. `counts` says how many rows went each way.
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        scale: float,
        split: Split = "train",
        shuffle: bool = True,
        holdout: frozenset[str] | None = None,
    ) -> None:
        self.config = config
        self.scale = scale
        self.split = split
        self.shuffle = shuffle
        self.counts = Counts()
        self._holdout = holdout

    def holdout(self) -> frozenset[str]:
        """The keys of the held-out positions, read from the dump when needed."""
        if self._holdout is None:
            self._holdout = holdout_keys(self.config)
        return self._holdout

    def _candidates(self, start: int, batch: Batch, seen: set[str]) -> list[int]:
        """The batch rows this split takes, purity filters applied."""
        config = self.config
        held = self.holdout() if self.split == "train" else None
        chosen: list[int] = []
        for row in range(_batch_stop(start, len(batch), config)):
            if not _wanted(start + row, config, self.split):
                continue
            self.counts.read += 1
            key = position_key(batch.fens[row])
            if held is not None:
                if key in held:
                    self.counts.leaked += 1
                    continue
            elif key in seen:
                self.counts.duplicate += 1
                continue
            else:
                seen.add(key)
            chosen.append(row)
        return chosen

    def samples(self) -> Iterator[Sample]:
        """The split's rows, one position at a time, in dump order."""
        config = self.config
        seen: set[str] = set()
        for start, batch in _dump_batches(config):
            chosen = self._candidates(start, batch, seen)
            if not chosen:
                continue
            values = win_probability(batch.cp, batch.mate, self.scale)
            moves, move_features, cuts = esca.encode_moves([batch.fens[row] for row in chosen])
            for slot, row in enumerate(chosen):
                best = move_index(moves[slot], batch.best_moves[row])
                if best is None:
                    self.counts.unmatched += 1
                    continue
                self.counts.kept += 1
                yield Sample(
                    features=batch.features[row].copy(),
                    moves=move_features[cuts[slot] : cuts[slot + 1]],
                    best=best,
                    value=float(values[row]),
                )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(self.config.seed if self.split == "train" else self.config.seed + 1)
        source = self.samples()
        stream = _shuffled(source, self.config.shuffle_buffer, rng) if self.shuffle else source
        pending: list[Sample] = []
        for sample in stream:
            pending.append(sample)
            if len(pending) == self.config.batch_size:
                yield collate(pending)
                pending = []
        if pending:
            yield collate(pending)


def fit_scale_on_dump(config: DataConfig, *, rows: int = SCALE_ROWS) -> float:
    """The logistic scale fitted on up to `rows` centipawn labels.

    Only held-out candidates count, so the fit never sees a training label.
    """
    labels: list[NDArray[np.float32]] = []
    taken = 0
    for start, batch in _dump_batches(config):
        index = start + np.arange(len(batch))
        keep = (index % config.holdout_every == 0) & (batch.mate == 0.0)
        labels.append(batch.cp[keep])
        taken += int(keep.sum())
        if taken >= rows:
            break
    if not labels:
        raise ValueError(f"no rows at depth {config.min_depth} in {config.dump}")
    return fit_scale(np.concatenate(labels)[:rows])
