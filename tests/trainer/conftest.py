"""What every trainer test needs: the synthetic dump and a tiny net."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_dump() -> Path:
    """The synthetic evaluation dump the `esca` tests ship."""
    root = Path(__file__).resolve().parents[2]
    return root / "rs_anglerfish" / "esca" / "tests" / "data" / "lichess_sample.jsonl.zst"
