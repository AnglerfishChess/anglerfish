"""The two-head net: one value logit and one score per legal move."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

__all__ = ["NetConfig", "TwoHeadNet"]


@dataclass(frozen=True)
class NetConfig:
    """The net's shape. `input_width` must match the encoded group selection."""

    input_width: int
    move_width: int = 24
    #: Trunk hidden widths, before the embedding layer.
    trunk: tuple[int, ...] = (1024, 512)
    #: Width of the position embedding both heads read.
    embedding: int = 256
    #: Hidden width of the per-move scorer.
    policy_hidden: int = 128
    dropout: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(values: Mapping[str, Any]) -> NetConfig:
        """The config a mapping from `as_dict` describes."""
        return NetConfig(
            input_width=int(values["input_width"]),
            move_width=int(values["move_width"]),
            trunk=tuple(int(width) for width in values["trunk"]),
            embedding=int(values["embedding"]),
            policy_hidden=int(values["policy_hidden"]),
            dropout=float(values["dropout"]),
        )


class TwoHeadNet(nn.Module):
    """A value head and a policy head over a shared position embedding.

    `forward` takes the position features `(b, input_width)`, the per-move
    features `(b, m, move_width)` and a boolean legality mask `(b, m)`, and
    returns the value logit `(b,)` and the move scores `(b, m)`, the illegal
    entries of which are -inf. Every row must carry at least one legal move.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.input_width
        for size in (*config.trunk, config.embedding):
            layers.append(nn.Linear(width, size))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            width = size
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Linear(config.embedding, 1)
        # A single linear over the embedding joined with a move's features,
        # summed instead of concatenated so the join is never materialised.
        self.policy_position = nn.Linear(config.embedding, config.policy_hidden)
        self.policy_move = nn.Linear(config.move_width, config.policy_hidden, bias=False)
        self.policy_score = nn.Linear(config.policy_hidden, 1)

    def forward(
        self,
        features: torch.Tensor,
        moves: torch.Tensor,
        move_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.trunk(features)
        value = self.value(embedding).squeeze(-1)
        joined = self.policy_position(embedding).unsqueeze(1) + self.policy_move(moves)
        policy = self.policy_score(torch.relu(joined)).squeeze(-1)
        return value, policy.masked_fill(~move_mask, float("-inf"))
