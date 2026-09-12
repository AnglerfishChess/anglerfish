"""The Lichess evaluation dump, read record by record."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard

__all__ = ["Row", "read"]


@dataclass(frozen=True, slots=True)
class Row:
    """One position of the dump and the evaluation chosen for it.

    `cp` and `mate` are side-relative, as the rest of the trainer reads them:
    the dump writes both from White's point of view, and a record with Black to
    move carries them negated. A mate row has `mate` other than zero and `cp`
    zero.
    """

    #: The four-field FEN the dump identifies the position by.
    fen: str
    cp: int
    mate: int
    #: The first move of the chosen line, as the dump spells it in UCI.
    best: str


def read(path: Path, *, min_depth: int = 0) -> Iterator[Row]:
    """The dump's rows, in file order.

    A record contributes the deepest evaluation reaching `min_depth` and the
    first line of that evaluation; a record with no such evaluation, or whose
    line is empty, is skipped. A blank line is skipped; a malformed one raises.
    """
    with open(path, "rb") as raw:
        stream = zstandard.ZstdDecompressor().stream_reader(raw)
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            row = _row(line, min_depth)
            if row is not None:
                yield row


def _row(line: str, min_depth: int) -> Row | None:
    """One JSON line as a row, or `None` where it carries none."""
    text = line.strip()
    if not text:
        return None
    record: dict[str, Any] = json.loads(text)
    fen = record["fen"]
    deepest: dict[str, Any] | None = None
    for evaluation in record["evals"]:
        if evaluation["depth"] >= min_depth and (deepest is None or evaluation["depth"] > deepest["depth"]):
            deepest = evaluation
    if deepest is None or not deepest["pvs"]:
        return None
    line_of = deepest["pvs"][0]
    best = line_of.get("line", "").split(" ", 1)[0]
    if not best:
        return None
    sign = -1 if fen.split(" ")[1] == "b" else 1
    mate = line_of.get("mate")
    if mate is not None:
        return Row(fen=fen, cp=0, mate=sign * int(mate), best=best)
    return Row(fen=fen, cp=sign * int(line_of.get("cp", 0)), mate=0, best=best)
