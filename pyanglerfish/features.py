"""Which of esca's fact arrays the net reads, and how they reach torch."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import esca.tensors as tensors
import numpy as np
import torch

__all__ = [
    "DEFAULT_GROUPS",
    "TORCH_DTYPES",
    "as_tensor",
    "default_names",
    "entries",
    "group_names",
    "layout_hash",
    "names_of",
    "width",
]


def group_names() -> tuple[str, ...]:
    """Every facts group esca declares, in catalogue order."""
    return tuple(dict.fromkeys(entry["group"] for entry in tensors.layout()))


#: The groups a net reads unless told otherwise: every group but `maps`, whose
#: arrays are the ones `attacks` and `threats` already carry.
DEFAULT_GROUPS: tuple[str, ...] = tuple(name for name in group_names() if name != "maps")


def names_of(groups: Sequence[str]) -> list[str]:
    """Every array name of `groups`, `group.field`, in catalogue order."""
    return tensors.names(list(groups))


def default_names() -> list[str]:
    """Every array name of `DEFAULT_GROUPS`."""
    return names_of(DEFAULT_GROUPS)


def entries(names: Sequence[str]) -> list[dict[str, Any]]:
    """The layout entry of each name, in the order given.

    Raises `ValueError` for a name no group declares.
    """
    known = {entry["name"]: entry for entry in tensors.layout()}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"not fact arrays: {', '.join(unknown)}")
    return [known[name] for name in names]


def width(names: Sequence[str]) -> int:
    """Values one position carries over `names`, all their axes flattened."""
    return sum(math.prod(entry["shape"]) for entry in entries(names))


def layout_hash(names: Sequence[str]) -> str:
    """A digest of the arrays `names` selects: each name, dtype and shape.

    Two selections share a digest exactly when they feed a net the same values
    in the same order.
    """
    shape = [[entry["name"], entry["dtype"], list(entry["shape"])] for entry in entries(names)]
    return hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest()


#: What each type of esca's layout reaches torch as. Torch has no unsigned
#: type wider than a byte, so those widen to the narrowest signed type that
#: holds them; no value changes.
TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "uint16": torch.int32,
    "int32": torch.int32,
    "uint32": torch.int64,
    "uint64": torch.int64,
    "float32": torch.float32,
}

_WIDENED: dict[np.dtype[Any], Any] = {
    np.dtype(np.uint16): np.int32,
    np.dtype(np.uint32): np.int64,
    np.dtype(np.uint64): np.int64,
}


def as_tensor(array: np.ndarray) -> torch.Tensor:
    """`array` as a tensor of the same values, typed as `TORCH_DTYPES` says."""
    wider = _WIDENED.get(array.dtype)
    if wider is not None:
        return torch.from_numpy(array.astype(wider))
    return torch.from_numpy(np.ascontiguousarray(array))
