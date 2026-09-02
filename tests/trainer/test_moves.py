"""The move-index helper on positions whose legal moves are known by hand."""

from __future__ import annotations

from pathlib import Path

import esca
import pytest

from pyanglerfish import move_index

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq -"
PROMOTION = "4k3/P7/8/8/8/8/8/4K3 w - -"


def moves_of(fen: str) -> list[esca.Move]:
    moves, _ = esca.encode_moves(fen)
    return moves


def test_finds_a_quiet_move() -> None:
    moves = moves_of(START)
    index = move_index(moves, esca.Move("e2", "e4"))
    assert index is not None
    assert moves[index].uci == "e2e4"


def test_ignores_kind_and_capture_flag() -> None:
    moves = moves_of(CASTLING)
    # Castling is spelled king-to-rook, and the kind the caller names is not
    # part of the match.
    index = move_index(moves, esca.Move("e1", "h1", kind="quiet", is_capture=True))
    assert index is not None
    assert moves[index].is_castling


def test_distinguishes_promotion_roles() -> None:
    moves = moves_of(PROMOTION)
    queen = move_index(moves, esca.Move("a7", "a8", "q"))
    knight = move_index(moves, esca.Move("a7", "a8", "n"))
    assert queen is not None
    assert knight is not None
    assert queen != knight
    assert moves[queen].promotion == "q"
    assert move_index(moves, esca.Move("a7", "a8")) is None


@pytest.mark.parametrize("origin,destination", [("e2", "e5"), ("d4", "d5")])
def test_absent_move_is_none(origin: str, destination: str) -> None:
    assert move_index(moves_of(START), esca.Move(origin, destination)) is None


def test_every_dump_label_is_a_legal_move(sample_dump: Path) -> None:
    batch = next(iter(esca.lichess.batches(sample_dump, batch_size=64, min_depth=0)))
    for row, fen in enumerate(batch.fens):
        moves = moves_of(fen)
        index = move_index(moves, batch.best_moves[row])
        assert index is not None
        assert moves[index].uci == batch.best_moves[row].uci
