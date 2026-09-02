"""Training the two-head net on the Lichess evaluation dump."""

from __future__ import annotations

import argparse
import itertools
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import esca
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import SCALE_ROWS, DataConfig, EvalBatches, fit_scale_on_dump, holdout_keys, resolve_groups
from .model import NetConfig, TwoHeadNet

__all__ = [
    "Metrics",
    "TrainConfig",
    "evaluate",
    "load_checkpoint",
    "main",
    "pick_device",
    "save_checkpoint",
    "train",
]


@dataclass(frozen=True)
class TrainConfig:
    """The optimiser, the schedule and how often the held-out slice is read."""

    steps: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    #: Weight of the policy cross-entropy against the value cross-entropy.
    policy_weight: float = 1.0
    #: `cosine` decays the learning rate to zero over `steps`; `constant` holds it.
    schedule: str = "cosine"
    eval_every: int = 200
    eval_batches: int = 8
    log_every: int = 20
    seed: int = 0


@dataclass(frozen=True)
class Metrics:
    """What one pass over the held-out slice says about the net."""

    rows: int
    value_loss: float
    policy_loss: float
    value_mae: float
    value_rmse: float
    sign_accuracy: float
    policy_top1: float
    policy_top3: float

    def __str__(self) -> str:
        return (
            f"rows {self.rows} "
            f"value loss {self.value_loss:.4f} mae {self.value_mae:.4f} rmse {self.value_rmse:.4f} "
            f"sign {self.sign_accuracy:.3f} | "
            f"policy loss {self.policy_loss:.4f} top1 {self.policy_top1:.3f} top3 {self.policy_top3:.3f}"
        )


def pick_device(name: str = "auto") -> torch.device:
    """The device to train on; `auto` takes CUDA where there is one."""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _losses(
    net: TwoHeadNet,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, policy = net(batch["features"], batch["moves"], batch["move_mask"])
    value_loss = F.binary_cross_entropy_with_logits(value, batch["value"])
    policy_loss = F.cross_entropy(policy, batch["best"])
    return value, policy, torch.stack((value_loss, policy_loss))


@torch.no_grad()
def evaluate(net: TwoHeadNet, batches: list[dict[str, torch.Tensor]], device: torch.device) -> Metrics:
    """The metrics of `docs/features.md` §7 over the given batches."""
    was_training = net.training
    net.eval()
    rows = 0
    value_loss = policy_loss = 0.0
    absolute = squared = signs = top1 = top3 = 0.0
    for held in batches:
        batch = _to(held, device)
        value, policy, losses = _losses(net, batch)
        count = batch["value"].shape[0]
        rows += count
        value_loss += float(losses[0]) * count
        policy_loss += float(losses[1]) * count
        predicted = torch.sigmoid(value)
        error = predicted - batch["value"]
        absolute += float(error.abs().sum())
        squared += float((error * error).sum())
        signs += float((torch.sign(predicted - 0.5) == torch.sign(batch["value"] - 0.5)).sum())
        wanted = batch["best"]
        ranked = policy.topk(min(3, policy.shape[1]), dim=1).indices
        top1 += float((ranked[:, 0] == wanted).sum())
        top3 += float((ranked == wanted.unsqueeze(1)).any(dim=1).sum())
    if was_training:
        net.train()
    if rows == 0:
        raise ValueError("nothing to evaluate")
    return Metrics(
        rows=rows,
        value_loss=value_loss / rows,
        policy_loss=policy_loss / rows,
        value_mae=absolute / rows,
        value_rmse=math.sqrt(squared / rows),
        sign_accuracy=signs / rows,
        policy_top1=top1 / rows,
        policy_top3=top3 / rows,
    )


def save_checkpoint(
    path: Path,
    net: TwoHeadNet,
    *,
    groups: tuple[str, ...],
    scale: float,
    step: int,
) -> None:
    """Writes the weights together with the manifest a loader checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_id": esca.SCHEMA_ID,
            "schema_semver": esca.SCHEMA_V0.semver,
            "groups": list(groups),
            "value_scale": scale,
            "net": net.config.as_dict(),
            "step": step,
            "state_dict": net.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, *, device: torch.device | str = "cpu") -> tuple[TwoHeadNet, dict[str, Any]]:
    """The net a checkpoint holds, and its manifest.

    Raises `ValueError` when the checkpoint was trained against a different
    feature schema than the installed `esca` emits.
    """
    manifest: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    stored = manifest.get("schema_id")
    if stored != esca.SCHEMA_ID:
        raise ValueError(f"checkpoint schema {stored} is not the installed schema {esca.SCHEMA_ID}")
    net = TwoHeadNet(NetConfig.from_dict(manifest["net"]))
    net.load_state_dict(manifest["state_dict"])
    net.to(device)
    return net, manifest


def _endless(dataset: EvalBatches) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        empty = True
        for batch in dataset:
            empty = False
            yield batch
        if empty:
            raise RuntimeError("the training split yielded no rows")


def _schedule(optimiser: torch.optim.Optimizer, config: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
    if config.schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(config.steps, 1))
    if config.schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimiser, lambda _step: 1.0)
    raise ValueError(f"unknown schedule: {config.schedule}")


def train(
    net: TwoHeadNet,
    data: DataConfig,
    config: TrainConfig,
    *,
    scale: float,
    device: torch.device,
    checkpoint: Path | None = None,
    log: bool = True,
) -> Metrics:
    """Runs `config.steps` optimiser steps and returns the last held-out metrics."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    net.to(device)
    net.train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = _schedule(optimiser, config)

    held_out = holdout_keys(data)
    if log:
        print(f"held-out positions {len(held_out)}")
    training = EvalBatches(data, scale=scale, split="train", holdout=held_out)
    holdout = EvalBatches(data, scale=scale, split="holdout", shuffle=False)
    held = list(itertools.islice(iter(holdout), config.eval_batches))
    if not held:
        raise ValueError("the held-out split is empty; lower --holdout-every or raise --max-rows")

    metrics = evaluate(net, held, device)
    if log:
        print(f"step 0: {metrics}")

    running = torch.zeros(2)
    seen = 0
    for step, raw in enumerate(itertools.islice(_endless(training), config.steps), start=1):
        batch = _to(raw, device)
        _, _, losses = _losses(net, batch)
        loss = losses[0] + config.policy_weight * losses[1]
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimiser.step()
        scheduler.step()
        running += losses.detach().cpu()
        seen += 1
        if log and step % config.log_every == 0:
            mean = running / seen
            print(
                f"step {step}: value {float(mean[0]):.4f} policy {float(mean[1]):.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )
            running = torch.zeros(2)
            seen = 0
        if step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate(net, held, device)
            if log:
                print(f"step {step}: held out: {metrics}")
            if checkpoint is not None:
                save_checkpoint(checkpoint, net, groups=data.group_list, scale=scale, step=step)
    if log:
        counts = training.counts
        print(
            f"train rows read {counts.read}, kept {counts.kept}, "
            f"held out elsewhere {counts.leaked}, best move not legal {counts.unmatched}; "
            f"held-out rows read {holdout.counts.read}, kept {holdout.counts.kept}, "
            f"repeated {holdout.counts.duplicate}, unmatched {holdout.counts.unmatched}"
        )
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Anglerfish two-head net.")
    parser.add_argument("--dump", type=Path, required=True, help="the Lichess evaluation dump")
    parser.add_argument("--groups", help="comma-separated schema groups; default is every group")
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--holdout-every", type=int, default=64, help="one record index in this many is held out")
    parser.add_argument("--max-rows", type=int, help="stop after this many dump rows, both splits together")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--read-batch", type=int, default=4096, help="rows the dump reader encodes at a time")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--trunk", default="1024,512", help="comma-separated trunk hidden widths")
    parser.add_argument("--embedding", type=int, default=256)
    parser.add_argument("--policy-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--scale", type=float, help="the logistic value scale; fitted when absent")
    parser.add_argument("--scale-rows", type=int, default=SCALE_ROWS, help="held-out labels the scale is fitted on")
    parser.add_argument("--checkpoint", type=Path, help="where to write the checkpoint")
    parser.add_argument("--resume", type=Path, help="a checkpoint to continue from")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    """The `python -m pyanglerfish.train` entry point."""
    args = _parser().parse_args(argv)
    groups = resolve_groups(tuple(args.groups.split(",")) if args.groups else None)
    data = DataConfig(
        dump=args.dump,
        groups=groups,
        min_depth=args.min_depth,
        holdout_every=args.holdout_every,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        read_batch=args.read_batch,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    config = TrainConfig(
        steps=args.steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        policy_weight=args.policy_weight,
        schedule=args.schedule,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        log_every=args.log_every,
        seed=args.seed,
    )
    device = pick_device(args.device)

    if args.resume is not None:
        net, manifest = load_checkpoint(args.resume, device=device)
        if tuple(manifest["groups"]) != groups:
            raise ValueError(f"checkpoint groups {manifest['groups']} are not {list(groups)}")
        scale = float(manifest["value_scale"])
        print(f"resumed from {args.resume} at step {manifest['step']}, value scale {scale:.1f}")
    else:
        scale = args.scale if args.scale is not None else fit_scale_on_dump(data, rows=args.scale_rows)
        print(f"value scale {scale:.1f}")
        net = TwoHeadNet(
            NetConfig(
                input_width=data.width,
                move_width=esca.MOVE_WIDTH,
                trunk=tuple(int(width) for width in args.trunk.split(",") if width),
                embedding=args.embedding,
                policy_hidden=args.policy_hidden,
                dropout=args.dropout,
            )
        )
    print(f"device {device}, groups {','.join(groups)}, input width {data.width}")
    train(net, data, config, scale=scale, device=device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
