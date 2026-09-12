"""Reading the evaluation dump: which evaluation a record contributes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard

from pyanglerfish.dump import read

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def written(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    """The records as a Zstandard-compressed JSON-lines dump."""
    path = tmp_path / "dump.jsonl.zst"
    body = "".join(json.dumps(record) + "\n" for record in records).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(body))
    return path


def test_reads_every_row_of_the_sample(sample_dump: Path) -> None:
    rows = list(read(sample_dump))
    assert len(rows) == 12
    assert rows[0].fen == START
    assert rows[0].cp == 22
    assert rows[0].mate == 0
    assert rows[0].best == "e2e4"
    assert all(len(row.fen.split(" ")) == 4 for row in rows)


def test_a_record_contributes_its_deepest_evaluation(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        [
            {
                "fen": START,
                "evals": [
                    {"depth": 12, "knodes": 1, "pvs": [{"cp": 10, "line": "d2d4"}]},
                    {"depth": 30, "knodes": 9, "pvs": [{"cp": 22, "line": "e2e4 e7e5"}]},
                    {"depth": 22, "knodes": 4, "pvs": [{"cp": 15, "line": "c2c4"}]},
                ],
            }
        ],
    )
    (row,) = read(path)
    assert (row.cp, row.best) == (22, "e2e4")


def test_a_record_below_the_depth_is_skipped(tmp_path: Path) -> None:
    path = written(
        tmp_path, [{"fen": START, "evals": [{"depth": 12, "knodes": 1, "pvs": [{"cp": 10, "line": "d2d4"}]}]}]
    )
    assert list(read(path, min_depth=20)) == []
    assert len(list(read(path, min_depth=12))) == 1


BLACK = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"


@pytest.mark.parametrize(
    ("fen", "cp", "mate", "wanted"),
    [(START, 30, None, (30, 0)), (BLACK, 30, None, (-30, 0)), (START, None, 4, (0, 4)), (BLACK, None, -4, (0, 4))],
)
def test_scores_are_side_relative(
    tmp_path: Path, fen: str, cp: int | None, mate: int | None, wanted: tuple[int, int]
) -> None:
    line: dict[str, object] = {"line": "e2e4"}
    if cp is not None:
        line["cp"] = cp
    if mate is not None:
        line["mate"] = mate
    path = written(tmp_path, [{"fen": fen, "evals": [{"depth": 30, "knodes": 1, "pvs": [line]}]}])
    (row,) = read(path)
    assert (row.cp, row.mate) == wanted


def test_a_record_with_no_line_is_skipped(tmp_path: Path) -> None:
    path = written(
        tmp_path,
        [
            {"fen": START, "evals": [{"depth": 30, "knodes": 1, "pvs": []}]},
            {"fen": START, "evals": [{"depth": 30, "knodes": 1, "pvs": [{"cp": 1, "line": ""}]}]},
            {"fen": START, "evals": []},
        ],
    )
    assert list(read(path)) == []


def test_a_blank_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl.zst"
    record = json.dumps({"fen": START, "evals": [{"depth": 30, "knodes": 1, "pvs": [{"cp": 5, "line": "e2e4"}]}]})
    path.write_bytes(zstandard.ZstdCompressor().compress(f"\n{record}\n\n".encode()))
    assert len(list(read(path))) == 1
