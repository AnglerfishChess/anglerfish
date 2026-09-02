"""What a checkpoint carries, and what a loader refuses."""

from __future__ import annotations

from pathlib import Path

import esca
import pytest
import torch

from pyanglerfish import NetConfig, TwoHeadNet
from pyanglerfish.train import load_checkpoint, save_checkpoint

GROUPS = ("state", "material")


def net() -> TwoHeadNet:
    torch.manual_seed(1)
    return TwoHeadNet(NetConfig(input_width=55, trunk=(8,), embedding=4, policy_hidden=3))


def test_round_trip_keeps_the_manifest_and_the_weights(tmp_path: Path) -> None:
    path = tmp_path / "net.pt"
    original = net()
    save_checkpoint(path, original, groups=GROUPS, scale=173.5, step=42)

    loaded, manifest = load_checkpoint(path)
    assert manifest["schema_id"] == esca.SCHEMA_ID
    assert manifest["schema_semver"] == esca.SCHEMA_V0.semver
    assert manifest["groups"] == list(GROUPS)
    assert manifest["value_scale"] == pytest.approx(173.5)
    assert manifest["step"] == 42
    assert loaded.config == original.config
    for before, after in zip(original.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(before, after)


def test_a_foreign_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "net.pt"
    save_checkpoint(path, net(), groups=GROUPS, scale=100.0, step=1)
    stored = torch.load(path, map_location="cpu", weights_only=False)
    stored["schema_id"] = "0" * 32
    torch.save(stored, path)

    with pytest.raises(ValueError, match=r"0{32}"):
        load_checkpoint(path)
