"""Batches over the synthetic dump: shapes, the split, and dropped rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import esca
import numpy as np
import pytest
import torch

from pyanglerfish import DataConfig, EvalBatches
from pyanglerfish import data as data_module

SCALE = 200.0


def config(dump: Path, **overrides: Any) -> DataConfig:
    base = DataConfig(dump=dump, min_depth=0, holdout_every=4, batch_size=4, shuffle_buffer=0, read_batch=8)
    return replace(base, **overrides)


def rows_of(dataset: EvalBatches) -> list[dict[str, torch.Tensor]]:
    return list(iter(dataset))


def test_batch_shapes_and_targets(sample_dump: Path) -> None:
    settings = config(sample_dump)
    dataset = EvalBatches(settings, scale=SCALE, split="train", shuffle=False)
    batches = rows_of(dataset)
    assert batches
    for batch in batches:
        rows, most = batch["move_mask"].shape
        assert batch["features"].shape == (rows, settings.width)
        assert batch["features"].dtype == torch.float32
        assert batch["moves"].shape == (rows, most, esca.MOVE_WIDTH)
        assert batch["value"].shape == (rows,)
        assert torch.all((batch["value"] >= 0.0) & (batch["value"] <= 1.0))
        assert torch.all(batch["move_mask"].any(dim=1))
        assert torch.all(batch["best"] >= 0)
        assert torch.all(batch["best"] < most)
        assert torch.all(batch["move_mask"].gather(1, batch["best"].unsqueeze(1)))
        # Padding rows are zero.
        assert torch.all(batch["moves"][~batch["move_mask"]] == 0.0)
    assert dataset.counts.kept == sum(int(batch["value"].shape[0]) for batch in batches)


def test_the_splits_partition_the_dump(sample_dump: Path) -> None:
    settings = config(sample_dump)
    train = EvalBatches(settings, scale=SCALE, split="train", shuffle=False)
    holdout = EvalBatches(settings, scale=SCALE, split="holdout", shuffle=False)
    rows_of(train)
    rows_of(holdout)
    assert train.counts.read > 0
    assert holdout.counts.read > 0
    every = settings.holdout_every
    assert holdout.counts.read == (train.counts.read + holdout.counts.read + every - 1) // every


def test_the_split_is_stable_across_passes(sample_dump: Path) -> None:
    settings = config(sample_dump)
    dataset = EvalBatches(settings, scale=SCALE, split="holdout", shuffle=False)
    first = torch.cat([batch["value"] for batch in rows_of(dataset)])
    second = torch.cat([batch["value"] for batch in rows_of(dataset)])
    assert torch.equal(first, second)


def test_a_group_selection_narrows_the_row(sample_dump: Path) -> None:
    settings = config(sample_dump, groups=("state", "material"))
    assert settings.width == 29 + 26
    batch = next(iter(EvalBatches(settings, scale=SCALE, shuffle=False)))
    assert batch["features"].shape[1] == settings.width


def test_unmatched_rows_are_dropped_and_counted(sample_dump: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def every_other(moves: object, target: object) -> int | None:
        seen.append(1)
        return None if len(seen) % 2 == 0 else 0

    monkeypatch.setattr(data_module, "move_index", every_other)
    dataset = EvalBatches(config(sample_dump), scale=SCALE, split="train", shuffle=False)
    rows = sum(int(batch["value"].shape[0]) for batch in rows_of(dataset))
    assert dataset.counts.unmatched == len(seen) // 2
    assert dataset.counts.kept == rows == dataset.counts.read - dataset.counts.unmatched


def test_max_rows_stops_the_stream(sample_dump: Path) -> None:
    settings = config(sample_dump, max_rows=4, read_batch=2)
    train = EvalBatches(settings, scale=SCALE, split="train", shuffle=False)
    holdout = EvalBatches(settings, scale=SCALE, split="holdout", shuffle=False)
    rows_of(train)
    rows_of(holdout)
    assert train.counts.read + holdout.counts.read == 4


def test_the_shuffle_buffer_keeps_every_row(sample_dump: Path) -> None:
    settings = config(sample_dump, shuffle_buffer=5, seed=3)
    ordered = EvalBatches(settings, scale=SCALE, split="train", shuffle=False)
    shuffled = EvalBatches(settings, scale=SCALE, split="train", shuffle=True)
    plain = torch.cat([batch["value"] for batch in rows_of(ordered)])
    mixed = torch.cat([batch["value"] for batch in rows_of(shuffled)])
    assert torch.equal(plain.sort().values, mixed.sort().values)


def test_the_fitted_scale_is_positive(sample_dump: Path) -> None:
    settings = config(sample_dump, holdout_every=1)
    assert data_module.fit_scale_on_dump(settings, rows=64) > 0.0


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
AFTER_NC6 = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"


class StubBatch:
    """What `EvalBatches` reads out of a dump batch, over given FENs."""

    def __init__(self, fens: list[str], width: int) -> None:
        self.fens = fens
        self.features = np.zeros((len(fens), width), dtype=np.float32)
        self.cp = np.full(len(fens), 25.0, dtype=np.float32)
        self.mate = np.zeros(len(fens), dtype=np.float32)
        self.best_moves = [esca.encode_moves(fen)[0][0] for fen in fens]

    def __len__(self) -> int:
        return len(self.fens)


@pytest.fixture
def stub_dump(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A dump of six rows, three of them the same position at other clocks."""
    fens = [
        START,
        AFTER_E4,
        START,
        AFTER_NC6,
        START.replace(" 0 1", " 4 7"),
        START.replace(" 0 1", " 6 9"),
    ]

    def batches(settings: DataConfig) -> Any:
        yield 0, StubBatch(fens, settings.width)

    monkeypatch.setattr(data_module, "_dump_batches", batches)
    return fens


def test_a_position_key_drops_the_clocks(stub_dump: list[str]) -> None:
    assert data_module.position_key(START) == data_module.position_key(stub_dump[4])
    assert data_module.position_key(START) != data_module.position_key(AFTER_E4)
    assert data_module.position_key(START) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def test_the_held_out_slice_keeps_one_row_per_position(sample_dump: Path, stub_dump: list[str]) -> None:
    settings = config(sample_dump, holdout_every=2)
    dataset = EvalBatches(settings, scale=SCALE, split="holdout", shuffle=False)
    kept = list(dataset.samples())
    assert len(kept) == 1
    assert dataset.counts.read == 3
    assert dataset.counts.duplicate == 2
    assert data_module.holdout_keys(settings) == {data_module.position_key(START)}


def test_a_training_row_of_a_held_out_position_is_dropped(sample_dump: Path, stub_dump: list[str]) -> None:
    settings = config(sample_dump, holdout_every=2)
    dataset = EvalBatches(settings, scale=SCALE, split="train", shuffle=False)
    kept = list(dataset.samples())
    assert [len(sample.moves) for sample in kept] == [20, 30]
    assert dataset.counts.read == 3
    assert dataset.counts.leaked == 1
    assert dataset.counts.kept == 2


def test_a_samples_move_rows_are_its_own(sample_dump: Path, stub_dump: list[str]) -> None:
    settings = config(sample_dump, holdout_every=2)
    dataset = EvalBatches(settings, scale=SCALE, split="train", shuffle=False)
    for sample, fen in zip(dataset.samples(), (AFTER_E4, AFTER_NC6), strict=True):
        assert np.array_equal(sample.moves, esca.encode_moves(fen)[1])
