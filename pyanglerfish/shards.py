"""A shard: the facts, the moves and the labels of many positions, on disk.

One safetensors file holds the facts in esca's packed layout, the move arrays
of every position's legal moves end to end, and the labels; the manifest beside
it names the layout the file was written under, and a load refuses a file
written under another.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import esca
import esca.tensors as tensors
import numpy as np

from . import moves as move_facts

__all__ = ["MANIFEST_SUFFIX", "Shard", "held_names", "load", "manifest", "save", "shard_paths"]

#: What a shard's manifest is called: the shard's name plus this.
MANIFEST_SUFFIX = ".manifest.json"

_LABELS: dict[str, str] = {"cp": "int32", "mate": "int16", "best": "uint16"}


@dataclass(frozen=True, slots=True)
class Shard:
    """The rows of one shard, the facts in the expanded layout.

    `facts` is one array per fact, the batch axis first. `moves` is one array
    per field of the `move` group over every row's legal moves end to end, and
    `cuts` says where each row's moves start: row `i` holds
    `cuts[i]:cuts[i + 1]`. `best` indexes a row's own moves, and `cp` and
    `mate` are the side-relative labels.
    """

    facts: dict[str, np.ndarray]
    moves: dict[str, np.ndarray]
    cuts: np.ndarray
    cp: np.ndarray
    mate: np.ndarray
    best: np.ndarray

    def __len__(self) -> int:
        return int(self.cuts.shape[0]) - 1

    @property
    def names(self) -> list[str]:
        """The fact arrays this shard carries, in catalogue order."""
        return [entry["name"] for entry in tensors.layout() if entry["name"] in self.facts]


def manifest(names: Sequence[str], count: int) -> dict[str, Any]:
    """What a shard of `count` rows over the fact arrays `names` is written as."""
    return {
        "esca": esca.__version__,
        "form": "packed",
        "count": count,
        "facts": [
            {"name": entry["name"], "dtype": entry["dtype"], "shape": list(entry["shape"])}
            for entry in tensors.layout()
            if entry["name"] in set(names)
        ],
        "moves": [{"name": field, "dtype": move_facts.MOVE_DTYPES[field]} for field in move_facts.MOVE_FIELDS],
        "labels": dict(_LABELS),
    }


def save(shard: Shard, path: Path) -> None:
    """Writes `shard` to `path`, and its manifest beside it."""
    arrays: dict[str, np.ndarray] = dict(tensors.pack(shard.facts))
    for field, array in shard.moves.items():
        arrays[f"move.{field}"] = array
    arrays["move.cuts"] = shard.cuts
    arrays["label.cp"] = shard.cp
    arrays["label.mate"] = shard.mate
    arrays["label.best"] = shard.best
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors.save(arrays, path)
    _manifest_path(path).write_text(json.dumps(manifest(shard.names, len(shard)), indent=1) + "\n")


def load(path: Path) -> Shard:
    """The shard at `path`, its facts expanded.

    Raises `ValueError` when the manifest beside it names a layout other than
    the one the installed esca and this trainer write.
    """
    written: dict[str, Any] = json.loads(_manifest_path(path).read_text())
    names = [entry["name"] for entry in written.get("facts", [])]
    wanted = manifest(names, written.get("count", 0))
    if written != wanted:
        raise ValueError(f"{path.name} was written under another layout, by esca {written.get('esca')}")

    stored = tensors.load(path)
    count = int(wanted["count"])
    held = int(stored["label.cp"].shape[0])
    if held != count:
        raise ValueError(f"{path.name} holds {held} rows, not the {count} its manifest names")
    facts = tensors.expand({name: stored[name] for name in names}, count)
    return Shard(
        facts=facts,
        moves={field: stored[f"move.{field}"] for field in move_facts.MOVE_FIELDS},
        cuts=stored["move.cuts"],
        cp=stored["label.cp"],
        mate=stored["label.mate"],
        best=stored["label.best"],
    )


def shard_paths(directory: Path, split: str) -> list[Path]:
    """The shards of `split` in `directory`, in the order they were written."""
    return sorted(directory.glob(f"{split}-*.safetensors"))


def held_names(path: Path) -> list[str]:
    """The fact arrays the shard at `path` carries, from its manifest alone."""
    written: dict[str, Any] = json.loads(_manifest_path(path).read_text())
    return [entry["name"] for entry in written["facts"]]


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + MANIFEST_SUFFIX)
