"""Turning the Lichess evaluation dump into shards.

`python -m pyanglerfish.build --dump … --out …`
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import esca
import esca.tensors as tensors
import numpy as np

from . import dump, features, moves, shards
from .moves import move_index

__all__ = ["BuildConfig", "Counts", "build", "holdout_keys", "main", "position_key"]


@dataclass(frozen=True)
class BuildConfig:
    """Which rows of a dump become shards, and how they are cut."""

    dump: Path
    out: Path
    #: The facts groups the shards carry; `None` is `features.DEFAULT_GROUPS`.
    groups: tuple[str, ...] | None = None
    #: A record contributes a row only if some evaluation reaches this depth.
    min_depth: int = 20
    #: One record index in this many goes to the held-out split.
    holdout_every: int = 64
    #: Rows per shard, and so the width of the training shuffle.
    shard_rows: int = 8192
    #: Cap on record indices read from the dump, both splits together.
    max_rows: int | None = None

    @property
    def group_list(self) -> tuple[str, ...]:
        return features.DEFAULT_GROUPS if self.groups is None else self.groups

    @property
    def names(self) -> list[str]:
        """The fact arrays a shard carries under this group selection."""
        return features.names_of(self.group_list)


@dataclass
class Counts:
    """How many rows the dump offered a split, and what became of them.

    `read` counts the split's candidates by index; `duplicate`, `leaked` and
    `unmatched` are the ones dropped and `kept` the ones written.
    """

    read: int = 0
    kept: int = 0
    #: The labelled best move was not among the legal moves.
    unmatched: int = 0
    #: The position was already held out under an earlier index.
    duplicate: int = 0
    #: A training candidate whose position is in the held-out set.
    leaked: int = 0
    #: The dump's placement is one no game can reach.
    unreachable: int = 0


@dataclass
class Report:
    """What one build wrote."""

    train: Counts = field(default_factory=Counts)
    holdout: Counts = field(default_factory=Counts)
    shards: list[Path] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"train rows read {self.train.read}, kept {self.train.kept}, "
            f"held out elsewhere {self.train.leaked}, best move not legal {self.train.unmatched}; "
            f"held-out rows read {self.holdout.read}, kept {self.holdout.kept}, "
            f"repeated {self.holdout.duplicate}, unmatched {self.holdout.unmatched}; "
            f"unreachable placements {self.train.unreachable + self.holdout.unreachable}; "
            f"{len(self.shards)} shards"
        )


def position_key(fen: str) -> str:
    """The position a FEN names, without its clocks.

    Placement, side to move, castling rights and en-passant square, joined by
    spaces. Two FENs share a key exactly when they are the same position at
    possibly different move counts.
    """
    return " ".join(fen.split(" ", 4)[:4])


def _rows(config: BuildConfig) -> Iterator[tuple[int, dump.Row]]:
    """The dump's rows within `max_rows`, each with its record index."""
    for index, row in enumerate(dump.read(config.dump, min_depth=config.min_depth)):
        if config.max_rows is not None and index >= config.max_rows:
            return
        yield index, row


def holdout_keys(config: BuildConfig) -> frozenset[str]:
    """The `position_key`s of every held-out candidate of the dump."""
    return frozenset(position_key(row.fen) for index, row in _rows(config) if index % config.holdout_every == 0)


class _Writer:
    """Rows gathered into shards of one split, written as they fill up."""

    def __init__(self, config: BuildConfig, split: str, counts: Counts) -> None:
        self.config = config
        self.split = split
        self.counts = counts
        self.written: list[Path] = []
        self._positions: list[esca.Position] = []
        self._annotated: list[esca.AnnotatedMove] = []
        self._cuts: list[int] = [0]
        self._labels: list[tuple[int, int, int]] = []

    def add(self, row: dump.Row, position: esca.Position) -> None:
        """One row, dropped and counted where its labelled move is not legal."""
        annotated = position.annotated_moves()
        try:
            target = position.parse_uci(row.best)
        except (esca.MoveParseError, esca.IllegalMove):
            self.counts.unmatched += 1
            return
        best = move_index([one.move for one in annotated], target)
        if best is None:
            self.counts.unmatched += 1
            return
        self.counts.kept += 1
        self._positions.append(position)
        self._annotated.extend(annotated)
        self._cuts.append(len(self._annotated))
        self._labels.append((row.cp, row.mate, best))
        if len(self._positions) >= self.config.shard_rows:
            self.flush()

    def flush(self) -> None:
        """Writes what has been gathered, if anything."""
        if not self._positions:
            return
        labels = np.array(self._labels, dtype=np.int64)
        shard = shards.Shard(
            facts=tensors.facts(self._positions, groups=list(self.config.group_list)),
            moves=moves.arrays(self._annotated),
            cuts=np.array(self._cuts, dtype=np.uint32),
            cp=labels[:, 0].astype(np.int32),
            mate=labels[:, 1].astype(np.int16),
            best=labels[:, 2].astype(np.uint16),
        )
        path = self.config.out / f"{self.split}-{len(self.written):05d}.safetensors"
        shards.save(shard, path)
        self.written.append(path)
        self._positions = []
        self._annotated = []
        self._cuts = [0]
        self._labels = []


def build(config: BuildConfig, *, log: bool = True) -> Report:
    """Writes the shards of both splits, and says what went into them.

    Rows are split by their index in the dump: one index in `holdout_every` is
    a held-out candidate, the rest are training candidates. A held-out
    candidate is kept only the first time its `position_key` is seen, and a
    training candidate whose key is held out is dropped, so no position reaches
    both splits. That needs every held-out key before the first training row,
    so the dump is read through once for the keys and once for the shards.
    """
    held = holdout_keys(config)
    if log:
        print(f"held-out positions {len(held)}")
    report = Report()
    writers = {
        "train": _Writer(config, "train", report.train),
        "holdout": _Writer(config, "holdout", report.holdout),
    }
    seen: set[str] = set()
    for index, row in _rows(config):
        split = "holdout" if index % config.holdout_every == 0 else "train"
        writer = writers[split]
        writer.counts.read += 1
        key = position_key(row.fen)
        if split == "train":
            if key in held:
                writer.counts.leaked += 1
                continue
        elif key in seen:
            writer.counts.duplicate += 1
            continue
        else:
            seen.add(key)
        try:
            position = esca.Position.from_fen(row.fen)
        except esca.FenError:
            writer.counts.unreachable += 1
            continue
        writer.add(row, position)
    for writer in writers.values():
        writer.flush()
        report.shards.extend(writer.written)
    if log:
        print(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Turn the Lichess evaluation dump into shards.")
    parser.add_argument("--dump", type=Path, required=True, help="the Lichess evaluation dump")
    parser.add_argument("--out", type=Path, required=True, help="the directory the shards go in")
    parser.add_argument("--groups", help="comma-separated facts groups; default is every group but maps")
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--holdout-every", type=int, default=64, help="one record index in this many is held out")
    parser.add_argument("--shard-rows", type=int, default=8192, help="rows per shard")
    parser.add_argument("--max-rows", type=int, help="stop after this many dump rows, both splits together")
    return parser


def main(argv: list[str] | None = None) -> None:
    """The `python -m pyanglerfish.build` entry point."""
    args = _parser().parse_args(argv)
    build(
        BuildConfig(
            dump=args.dump,
            out=args.out,
            groups=tuple(args.groups.split(",")) if args.groups else None,
            min_depth=args.min_depth,
            holdout_every=args.holdout_every,
            shard_rows=args.shard_rows,
            max_rows=args.max_rows,
        )
    )


if __name__ == "__main__":
    main()
