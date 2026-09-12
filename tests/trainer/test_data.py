"""Batches over the sample's shards: types, padding and the split."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import esca.tensors as tensors
import pytest
import torch

from pyanglerfish import DataConfig, ShardBatches, shards
from pyanglerfish.data import fit_scale_on_shards
from pyanglerfish.features import TORCH_DTYPES
from pyanglerfish.moves import MOVE_DTYPES, MOVE_FIELDS

SCALE = 200.0


def config(directory: Path, **overrides: Any) -> DataConfig:
    return replace(DataConfig(shards=directory, batch_size=4), **overrides)


def batches(settings: DataConfig, split: str = "train") -> list[Any]:
    return list(ShardBatches(settings, scale=SCALE, split=split, shuffle=False))


def test_a_batch_is_typed_as_escas_layout_says(sample_shards: Path) -> None:
    known = {entry["name"]: entry for entry in tensors.layout()}
    for batch in batches(config(sample_shards)):
        rows = len(batch)
        for name, array in batch.facts.items():
            assert array.shape == (rows, *known[name]["shape"]), name
            assert array.dtype == TORCH_DTYPES[known[name]["dtype"]], name
        most = batch.move_mask.shape[1]
        for field in MOVE_FIELDS:
            assert batch.moves[field].shape == (rows, most), field
            assert batch.moves[field].dtype == TORCH_DTYPES[MOVE_DTYPES[field]], field


def test_a_batchs_labels_name_a_legal_move(sample_shards: Path) -> None:
    for batch in batches(config(sample_shards)):
        assert torch.all((batch.value >= 0.0) & (batch.value <= 1.0))
        assert batch.value.dtype == torch.float32
        assert torch.all(batch.move_mask.any(dim=1))
        assert torch.all(batch.best >= 0)
        assert torch.all(batch.best < batch.move_mask.shape[1])
        assert torch.all(batch.move_mask.gather(1, batch.best.unsqueeze(1)))


def test_padding_is_zero(sample_shards: Path) -> None:
    for batch in batches(config(sample_shards)):
        padding = ~batch.move_mask
        for field in MOVE_FIELDS:
            assert torch.all(batch.moves[field][padding] == 0), field


def test_the_splits_partition_the_dump(sample_shards: Path) -> None:
    settings = config(sample_shards)
    train = sum(len(batch) for batch in batches(settings, "train"))
    holdout = sum(len(batch) for batch in batches(settings, "holdout"))
    assert (train, holdout) == (6, 6)


def test_the_split_is_stable_across_passes(sample_shards: Path) -> None:
    dataset = ShardBatches(config(sample_shards), scale=SCALE, split="holdout", shuffle=False)
    first = torch.cat([batch.value for batch in dataset])
    second = torch.cat([batch.value for batch in dataset])
    assert torch.equal(first, second)


def test_shuffling_keeps_every_row(sample_shards: Path) -> None:
    settings = config(sample_shards, batch_size=2, seed=3)
    plain = torch.cat([batch.value for batch in batches(settings)])
    shuffled = ShardBatches(settings, scale=SCALE, split="train", shuffle=True)
    mixed = torch.cat([batch.value for batch in shuffled])
    assert torch.equal(plain.sort().values, mixed.sort().values)


def test_a_selection_narrows_the_batch(sample_shards: Path) -> None:
    settings = config(sample_shards, features=("state.in_check", "material.pawns"))
    batch = next(iter(ShardBatches(settings, scale=SCALE, shuffle=False)))
    assert list(batch.facts) == ["state.in_check", "material.pawns"]
    assert batch.facts["material.pawns"].shape == (4, 2)


def test_an_array_the_shards_lack_is_refused(sample_shards: Path) -> None:
    settings = config(sample_shards, features=("maps.hanging",))
    with pytest.raises(ValueError, match=r"maps\.hanging"):
        ShardBatches(settings, scale=SCALE)


def test_the_selection_defaults_to_what_the_shards_carry(sample_shards: Path) -> None:
    settings = config(sample_shards)
    held = shards.load(shards.shard_paths(sample_shards, "train")[0])
    assert settings.selection() == held.names


def test_the_fitted_scale_is_positive(sample_shards: Path) -> None:
    assert fit_scale_on_shards(sample_shards, rows=64) > 0.0
