"""What every trainer test needs: the synthetic dump and shards built from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyanglerfish.build import BuildConfig, build


@pytest.fixture(scope="session")
def sample_dump() -> Path:
    """A dozen records in the format of the Lichess evaluation dump."""
    return Path(__file__).resolve().parents[1] / "data" / "lichess_sample.jsonl.zst"


@pytest.fixture(scope="session")
def sample_shards(sample_dump: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The sample dump built into shards of both splits."""
    out = tmp_path_factory.mktemp("shards")
    build(BuildConfig(dump=sample_dump, out=out, min_depth=0, holdout_every=2, shard_rows=4), log=False)
    return out
