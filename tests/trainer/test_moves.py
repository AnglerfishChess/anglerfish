"""The move arrays, and locating a move among a position's legal ones."""

from __future__ import annotations

from pathlib import Path

import esca
import numpy as np
import pytest

from pyanglerfish import move_index
from pyanglerfish.dump import read
from pyanglerfish.moves import MOVE_DTYPES, MOVE_FIELDS, MOVE_WIDTH, ROLES, arrays

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq -"
PROMOTION = "4k3/P7/8/8/8/8/8/4K3 w - -"
# 1. e4 e5 2. Bc4 Nc6 3. Qh5: Qxf7 is mate, and f7 is defended by the king.
SCHOLAR = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq -"


def moves_of(fen: str) -> list[esca.Move]:
    return esca.Position.from_fen(fen).legal_moves()


def test_finds_a_quiet_move() -> None:
    moves = moves_of(START)
    index = move_index(moves, esca.Move("e2", "e4"))
    assert index is not None
    assert moves[index].uci() == "e2e4"


def test_a_castling_is_found_by_either_spelling() -> None:
    position = esca.Position.from_fen(CASTLING)
    moves = position.legal_moves()
    for text in ("e1g1", "e1h1"):
        index = move_index(moves, position.parse_uci(text))
        assert index is not None, text
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


@pytest.mark.parametrize("absent", [esca.Move("e2", "e5"), esca.Move("d4", "d5")])
def test_absent_move_is_none(absent: esca.Move) -> None:
    assert move_index(moves_of(START), absent) is None


def test_every_dump_label_is_a_legal_move(sample_dump: Path) -> None:
    for row in read(sample_dump):
        position = esca.Position.from_fen(row.fen)
        index = move_index(position.legal_moves(), position.parse_uci(row.best))
        assert index is not None, row.best


def test_a_field_per_move_fact_of_the_catalogue() -> None:
    assert len(MOVE_FIELDS) == MOVE_WIDTH == 27
    assert MOVE_FIELDS[:3] == ("victim", "mover", "promotion")
    assert MOVE_DTYPES["gives_check"] == "bool"
    assert MOVE_DTYPES["see"] == "int32"


def test_the_arrays_keep_the_type_each_fact_was_declared_with() -> None:
    annotated = esca.Position.from_fen(SCHOLAR).annotated_moves()
    written = arrays(annotated)
    assert set(written) == set(MOVE_FIELDS)
    for field, array in written.items():
        assert array.dtype == np.dtype(MOVE_DTYPES[field]), field
        assert array.shape == (len(annotated),), field


def test_a_role_is_its_code_and_an_absent_one_is_minus_one() -> None:
    position = esca.Position.from_fen(SCHOLAR)
    annotated = position.annotated_moves()
    written = arrays(annotated)
    mate = next(index for index, one in enumerate(annotated) if one.move.uci() == "h5f7")
    quiet = next(index for index, one in enumerate(annotated) if one.move.uci() == "a2a3")

    assert written["mover"][mate] == ROLES.index("q")
    assert written["victim"][mate] == ROLES.index("p")
    assert written["gives_check"][mate]
    assert written["mover"][quiet] == ROLES.index("p")
    assert written["victim"][quiet] == -1
    assert written["promotion"][quiet] == -1
