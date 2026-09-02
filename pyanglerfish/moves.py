"""Locating a move among a position's legal moves."""

from __future__ import annotations

from collections.abc import Sequence

from esca import Move

__all__ = ["move_index"]


def _key(mv: Move) -> tuple[str, str, str | None]:
    return mv.origin, mv.destination, mv.promotion


def move_index(moves: Sequence[Move], target: Move) -> int | None:
    """The position of `target` in `moves`, or `None` when it is absent.

    Two moves are the same when their origin, destination and promotion role
    agree; the move kind and the capture flag are ignored, so a move read from
    text matches the generated one it names.
    """
    wanted = _key(target)
    for index, mv in enumerate(moves):
        if _key(mv) == wanted:
            return index
    return None
