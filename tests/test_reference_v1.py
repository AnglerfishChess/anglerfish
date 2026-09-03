"""The Rust extractor against the Python reference, value by value.

The fixture is the golden corpus the Rust side checks itself against, so a
disagreement is a disagreement about `docs/features.md`, not about a corpus.
Every row of both corpora is checked; set `ESCA_REFERENCE_ROWS` to a number to
narrow the sweep while working on the reference.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest
from reference.features import encode

DATA = Path(__file__).resolve().parents[1] / "rs_anglerfish" / "esca" / "tests" / "data"
WIDTH = 1846


def rows_wanted() -> int | None:
    setting = os.environ.get("ESCA_REFERENCE_ROWS", "all")
    return None if setting == "all" else int(setting)


def corpus(name: str) -> list[str]:
    return [
        line.strip() for line in (DATA / name).read_text().splitlines() if line.strip() and not line.startswith("#")
    ]


def vectors(name: str) -> list[list[float]]:
    raw = (DATA / name).read_bytes()
    stride = WIDTH * 4
    assert len(raw) % stride == 0, "a fixture is whole rows"
    return [list(struct.unpack(f"<{WIDTH}f", raw[start : start + stride])) for start in range(0, len(raw), stride)]


def feature_names() -> list[str]:
    from reference.features import SCHEMA

    names = []
    for group, _width, features in SCHEMA:
        for feature, _offset, width in features:
            names.extend(f"{group}.{feature}[{slot}]" for slot in range(width))
    return names


NAMES = feature_names()


def cases(fens: str, binary: str, variant: str) -> list[tuple[str, list[float], str]]:
    limit = rows_wanted()
    positions = corpus(fens)
    expected = vectors(binary)
    assert len(positions) == len(expected), "corpus and fixture disagree"
    pairs = list(zip(positions, expected, strict=True))
    if limit is not None:
        pairs = pairs[:limit]
    return [(fen, values, variant) for fen, values in pairs]


CLASSIC = cases("fens_classic.txt", "vectors_classic.bin", "chess")
CHESS960 = cases("fens_chess960.txt", "vectors_chess960.bin", "chess960")


def check(fen: str, expected: list[float], variant: str) -> None:
    actual = encode(fen, variant)
    assert len(actual) == WIDTH
    mismatches = [
        (NAMES[index], actual[index], expected[index])
        for index in range(WIDTH)
        if struct.pack("<f", actual[index]) != struct.pack("<f", expected[index])
    ]
    assert not mismatches, f"{fen}: {mismatches[:12]}"


@pytest.mark.parametrize(("fen", "expected", "variant"), CLASSIC)
def test_classic_rows_match_the_reference(fen: str, expected: list[float], variant: str) -> None:
    check(fen, expected, variant)


@pytest.mark.parametrize(("fen", "expected", "variant"), CHESS960)
def test_chess960_rows_match_the_reference(fen: str, expected: list[float], variant: str) -> None:
    check(fen, expected, variant)


def test_the_schema_the_reference_reads_is_as_wide_as_the_fixture() -> None:
    assert len(NAMES) == WIDTH
