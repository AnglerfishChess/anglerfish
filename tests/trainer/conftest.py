"""What every trainer test needs: the synthetic dump and a tiny net."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_dump() -> Path:
    """A dozen records in the format of the Lichess evaluation dump."""
    return Path(__file__).resolve().parents[1] / "data" / "lichess_sample.jsonl.zst"
