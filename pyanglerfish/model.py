"""The two-head net: one value logit and one score per legal move."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from .features import default_names, width
from .moves import MOVE_FIELDS

__all__ = ["NetConfig", "TwoHeadNet", "join"]


def join(arrays: Mapping[str, torch.Tensor], names: Sequence[str], lead: int, dtype: torch.dtype) -> torch.Tensor:
    """The named arrays cast to `dtype` and joined along one trailing axis.

    The first `lead` axes are kept — the batch, and the move for a per-move
    array — and everything past them is flattened.
    """
    return torch.cat([arrays[name].reshape(*arrays[name].shape[:lead], -1).to(dtype) for name in names], dim=-1)


@dataclass(frozen=True)
class NetConfig:
    """The net's shape and the arrays it reads.

    `features` names the fact arrays the trunk takes, `group.field` as
    `docs/features.md` lists them, in the order they are joined; the default is
    every array of `pyanglerfish.features.DEFAULT_GROUPS`, which is every group
    but `maps`, whose arrays repeat those of `attacks` and `threats`.
    """

    features: tuple[str, ...] = field(default_factory=lambda: tuple(default_names()))
    #: The fields of the `move` group the policy head takes.
    move_fields: tuple[str, ...] = MOVE_FIELDS
    #: Trunk hidden widths, before the embedding layer.
    trunk: tuple[int, ...] = (1024, 512)
    #: Width of the position embedding both heads read.
    embedding: int = 256
    #: Hidden width of the per-move scorer.
    policy_hidden: int = 128
    dropout: float = 0.0

    @property
    def input_width(self) -> int:
        """Values one position carries over `features`."""
        return width(self.features)

    @property
    def move_width(self) -> int:
        """Values one move carries over `move_fields`."""
        return len(self.move_fields)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(values: Mapping[str, Any]) -> NetConfig:
        """The config a mapping from `as_dict` describes."""
        return NetConfig(
            features=tuple(str(name) for name in values["features"]),
            move_fields=tuple(str(name) for name in values["move_fields"]),
            trunk=tuple(int(size) for size in values["trunk"]),
            embedding=int(values["embedding"]),
            policy_hidden=int(values["policy_hidden"]),
            dropout=float(values["dropout"]),
        )


class TwoHeadNet(nn.Module):
    """A value head and a policy head over a shared position embedding.

    `forward` takes the fact arrays of `config.features`, each `(b, …)` in the
    type esca declared it with, the arrays of `config.move_fields`, each
    `(b, m)`, and a boolean legality mask `(b, m)`. The first layer casts each
    array to the net's own dtype, wherever it is held, and joins them; nothing
    is scaled. It returns the value logit `(b,)` and the move scores `(b, m)`,
    the illegal entries of which are -inf. Every row must carry at least one
    legal move.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        previous = config.input_width
        for size in (*config.trunk, config.embedding):
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            previous = size
        self.trunk = nn.Sequential(*layers)
        self.value = nn.Linear(config.embedding, 1)
        # A single linear over the embedding joined with a move's features,
        # summed instead of concatenated so the join is never materialised.
        self.policy_position = nn.Linear(config.embedding, config.policy_hidden)
        self.policy_move = nn.Linear(config.move_width, config.policy_hidden, bias=False)
        self.policy_score = nn.Linear(config.policy_hidden, 1)

    @property
    def dtype(self) -> torch.dtype:
        """The type the net computes in, and casts its input to."""
        return self.value.weight.dtype

    def forward(
        self,
        facts: Mapping[str, torch.Tensor],
        moves: Mapping[str, torch.Tensor],
        move_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.dtype
        embedding = self.trunk(join(facts, self.config.features, 1, dtype))
        value = self.value(embedding).squeeze(-1)
        joined = self.policy_position(embedding).unsqueeze(1) + self.policy_move(
            join(moves, self.config.move_fields, 2, dtype)
        )
        policy = self.policy_score(torch.relu(joined)).squeeze(-1)
        return value, policy.masked_fill(~move_mask, float("-inf"))
