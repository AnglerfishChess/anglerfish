"""What a checkpoint carries, and what a loader refuses."""

from __future__ import annotations

from pathlib import Path

import esca
import pytest
import torch

from pyanglerfish import NetConfig, TwoHeadNet, layout_hash
from pyanglerfish.train import load_checkpoint, save_checkpoint

FEATURES = ("state.in_check", "material.pawns")


def net() -> TwoHeadNet:
    torch.manual_seed(1)
    return TwoHeadNet(NetConfig(features=FEATURES, trunk=(8,), embedding=4, policy_hidden=3))


def test_round_trip_keeps_the_manifest_and_the_weights(tmp_path: Path) -> None:
    path = tmp_path / "net.pt"
    original = net()
    save_checkpoint(path, original, scale=173.5, step=42)

    loaded, manifest = load_checkpoint(path)
    assert manifest["esca"] == esca.__version__
    assert manifest["features"] == list(FEATURES)
    assert manifest["layout_hash"] == layout_hash(FEATURES)
    assert manifest["value_scale"] == pytest.approx(173.5)
    assert manifest["step"] == 42
    assert loaded.config == original.config
    for before, after in zip(original.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(before, after)


def test_another_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "net.pt"
    save_checkpoint(path, net(), scale=100.0, step=1)
    stored = torch.load(path, map_location="cpu", weights_only=False)
    stored["layout_hash"] = "0" * 64
    torch.save(stored, path)

    with pytest.raises(ValueError, match=r"0{64}"):
        load_checkpoint(path)


def test_the_hash_follows_the_selection() -> None:
    assert layout_hash(FEATURES) != layout_hash(tuple(reversed(FEATURES)))
    assert layout_hash(FEATURES) != layout_hash([*FEATURES, "pawns.passed"])
    assert layout_hash(FEATURES) == layout_hash(list(FEATURES))
