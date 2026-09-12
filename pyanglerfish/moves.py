"""What a legal move does, as one typed array per fact.

`esca.tensors` covers a position's facts and not a move's, so the arrays of the
`move` group are built here, under the rules `docs/features.md` states for the
tensor form: a `bool` stays a `bool`, an `i32` an `int32`, a role its code, and
an absent role -1. Nothing is scaled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from esca import AnnotatedMove, Move
from esca._esca import move_facts_catalogue

__all__ = [
    "MOVE_DTYPES",
    "MOVE_FIELDS",
    "MOVE_WIDTH",
    "ROLES",
    "arrays",
    "move_index",
]

#: Every role, in the order a role code numbers them: `p` is 0 and `k` is 5.
ROLES: tuple[str, ...] = ("p", "n", "b", "r", "q", "k")

_ROLE_CODE = {letter: code for code, letter in enumerate(ROLES)}

#: The array type each declared type is written as.
_DTYPE_OF: dict[str, str] = {
    "bool": "bool",
    "i32": "int32",
    "Role": "uint8",
    "Option<Role>": "int16",
}


def _dtypes() -> dict[str, str]:
    """The array type of every field the `move` catalogue declares."""
    out: dict[str, str] = {}
    for entry in move_facts_catalogue():
        declared = entry["type"]
        if declared not in _DTYPE_OF:
            raise RuntimeError(f"{entry['name']} is a {declared}, which has no array type here")
        out[entry["name"]] = _DTYPE_OF[declared]
    return out


#: The array type of each field.
MOVE_DTYPES: dict[str, str] = _dtypes()

#: What each field of the `move` group is called, in catalogue order.
MOVE_FIELDS: tuple[str, ...] = tuple(MOVE_DTYPES)

#: Values one move carries: every field is a single value.
MOVE_WIDTH: int = len(MOVE_FIELDS)


def _value(raw: Any, dtype: str) -> Any:
    """One field of `MoveFacts.to_dict()` as the number its array holds."""
    if dtype == "uint8":
        return _ROLE_CODE[raw]
    if dtype == "int16":
        return -1 if raw is None else _ROLE_CODE[raw]
    return raw


def arrays(annotated: Iterable[AnnotatedMove]) -> dict[str, np.ndarray]:
    """One array per field of the `move` group over the given moves.

    Each is one value per move, in the order the moves were given.
    """
    columns: dict[str, list[Any]] = {field: [] for field in MOVE_FIELDS}
    for one in annotated:
        written: dict[str, Any] = dict(one.facts.to_dict())
        for field in MOVE_FIELDS:
            columns[field].append(_value(written[field], MOVE_DTYPES[field]))
    return {field: np.array(values, dtype=MOVE_DTYPES[field]) for field, values in columns.items()}


def move_index(moves: Sequence[Move], target: Move) -> int | None:
    """The position of `target` in `moves`, or `None` when it is absent.

    Two moves are the same when their origin, destination and promotion role
    agree, which is what `Move` equality says, so a move read from text matches
    the generated one it names.
    """
    for index, mv in enumerate(moves):
        if mv == target:
            return index
    return None
