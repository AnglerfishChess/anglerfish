"""A few steps of the real loop over the sample's shards."""

from __future__ import annotations

from pathlib import Path

import torch

from pyanglerfish import DataConfig, NetConfig, TwoHeadNet
from pyanglerfish.train import TrainConfig, load_checkpoint, main, pick_device, train

FEATURES = ("state.in_check", "material.pawns", "pawns.passed")


def settings(directory: Path) -> DataConfig:
    return DataConfig(shards=directory, features=FEATURES, batch_size=2)


def test_a_short_run_reports_metrics_and_writes_a_checkpoint(sample_shards: Path, tmp_path: Path) -> None:
    data = settings(sample_shards)
    net = TwoHeadNet(NetConfig(features=FEATURES, trunk=(16,), embedding=8, policy_hidden=4))
    checkpoint = tmp_path / "net.pt"
    metrics = train(
        net,
        data,
        TrainConfig(steps=6, eval_every=3, eval_batches=2, log_every=100),
        scale=200.0,
        device=torch.device("cpu"),
        checkpoint=checkpoint,
        log=False,
    )
    assert metrics.rows > 0
    assert 0.0 <= metrics.value_mae <= 1.0
    assert 0.0 <= metrics.policy_top1 <= metrics.policy_top3 <= 1.0
    assert metrics.value_rmse >= metrics.value_mae

    loaded, manifest = load_checkpoint(checkpoint)
    assert manifest["step"] == 6
    assert manifest["features"] == list(FEATURES)
    assert loaded.config == net.config


def test_the_cli_trains_and_resumes(sample_shards: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "cli.pt"
    argv = [
        "--shards", str(sample_shards),
        "--features", ",".join(FEATURES),
        "--batch-size", "2",
        "--steps", "4",
        "--eval-every", "4",
        "--eval-batches", "1",
        "--log-every", "100",
        "--trunk", "16",
        "--embedding", "8",
        "--policy-hidden", "4",
        "--scale", "200",
        "--checkpoint", str(checkpoint),
        "--device", "cpu",
    ]  # fmt: skip
    main(argv)
    assert checkpoint.exists()
    main([*argv, "--resume", str(checkpoint)])
    _, manifest = load_checkpoint(checkpoint)
    assert manifest["value_scale"] == 200.0


def test_the_device_is_picked() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in {"cpu", "cuda"}
