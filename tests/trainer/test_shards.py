"""What a shard carries, and what a loader refuses."""

from __future__ import annotations

import json
from pathlib import Path

import esca
import esca.tensors as tensors
import numpy as np
import pytest

from pyanglerfish import shards
from pyanglerfish.build import BuildConfig, build
from pyanglerfish.features import DEFAULT_GROUPS
from pyanglerfish.moves import MOVE_DTYPES, MOVE_FIELDS


def test_a_build_writes_both_splits(sample_shards: Path) -> None:
    train = shards.shard_paths(sample_shards, "train")
    holdout = shards.shard_paths(sample_shards, "holdout")
    assert [path.name for path in train] == ["train-00000.safetensors", "train-00001.safetensors"]
    assert len(holdout) == 2
    assert sum(len(shards.load(path)) for path in train + holdout) == 12


def test_a_shard_holds_the_layout_esca_declares(sample_shards: Path) -> None:
    shard = shards.load(shards.shard_paths(sample_shards, "train")[0])
    wanted = {entry["name"]: entry for entry in tensors.layout() if entry["group"] in DEFAULT_GROUPS}
    assert set(shard.facts) == set(wanted)
    for name, array in shard.facts.items():
        assert array.shape == (len(shard), *wanted[name]["shape"]), name
        assert array.dtype == np.dtype(wanted[name]["dtype"]), name
    for field in MOVE_FIELDS:
        assert shard.moves[field].dtype == np.dtype(MOVE_DTYPES[field]), field
        assert shard.moves[field].shape == (int(shard.cuts[-1]),), field


def test_a_shard_round_trips(sample_shards: Path, tmp_path: Path) -> None:
    original = shards.load(shards.shard_paths(sample_shards, "train")[0])
    path = tmp_path / "again.safetensors"
    shards.save(original, path)
    again = shards.load(path)
    assert len(again) == len(original)
    for name, array in original.facts.items():
        assert np.array_equal(again.facts[name], array), name
    for field, array in original.moves.items():
        assert np.array_equal(again.moves[field], array), field
    assert np.array_equal(again.cuts, original.cuts)
    assert np.array_equal(again.cp, original.cp)
    assert np.array_equal(again.mate, original.mate)
    assert np.array_equal(again.best, original.best)


def test_the_manifest_names_the_layout(sample_shards: Path) -> None:
    path = shards.shard_paths(sample_shards, "train")[0]
    written = json.loads((path.parent / (path.name + shards.MANIFEST_SUFFIX)).read_text())
    assert written["esca"] == esca.__version__
    assert written["form"] == "packed"
    assert written["count"] == 4
    assert written["facts"][0] == {"name": "placement.units", "dtype": "bool", "shape": [2, 64]}
    assert {entry["name"] for entry in written["moves"]} == set(MOVE_FIELDS)


@pytest.mark.parametrize(
    ("damage", "complaint"),
    [
        ({"esca": "0.0.1"}, "another layout"),
        ({"form": "expanded"}, "another layout"),
        ({"facts": [{"name": "state.in_check", "dtype": "uint8", "shape": []}]}, "another layout"),
        ({"count": 3}, "holds 4 rows"),
    ],
)
def test_a_shard_the_manifest_does_not_describe_is_refused(
    sample_shards: Path, tmp_path: Path, damage: dict[str, object], complaint: str
) -> None:
    source = shards.shard_paths(sample_shards, "train")[0]
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    manifest = json.loads((source.parent / (source.name + shards.MANIFEST_SUFFIX)).read_text())
    manifest.update(damage)
    (tmp_path / (path.name + shards.MANIFEST_SUFFIX)).write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=complaint):
        shards.load(path)


def test_a_group_selection_narrows_the_shard(sample_dump: Path, tmp_path: Path) -> None:
    config = BuildConfig(
        dump=sample_dump, out=tmp_path, groups=("state", "material"), min_depth=0, holdout_every=2, shard_rows=4
    )
    build(config, log=False)
    shard = shards.load(shards.shard_paths(tmp_path, "train")[0])
    assert set(shard.facts) == set(tensors.names(["state", "material"]))
