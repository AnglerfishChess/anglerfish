"""The value target and the logistic scale it is read on."""

from __future__ import annotations

import numpy as np
import pytest

from pyanglerfish import fit_scale, win_probability


def probabilities(cp: list[float], mate: list[float], scale: float) -> np.ndarray:
    return win_probability(np.array(cp, dtype=np.float32), np.array(mate, dtype=np.float32), scale)


def test_centipawns_pass_through_the_logistic() -> None:
    values = probabilities([0.0, 200.0, -200.0, 100000.0], [0.0] * 4, 200.0)
    assert values.dtype == np.float32
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)), rel=1e-6)
    assert values[2] == pytest.approx(1.0 - values[1], rel=1e-6)
    assert 0.0 < values[3] <= 1.0


def test_mate_rows_ignore_the_centipawn_column() -> None:
    values = probabilities([0.0, 0.0, 0.0], [1.0, -1.0, 20.0], 200.0)
    assert values[0] == pytest.approx(0.5 * (1.0 + 0.999))
    assert values[1] == pytest.approx(0.5 * (1.0 - 0.999))
    assert values[2] == pytest.approx(0.5 * (1.0 + 0.98))


def test_targets_stay_inside_the_unit_interval() -> None:
    cp = np.linspace(-30000.0, 30000.0, 4001, dtype=np.float32)
    values = win_probability(cp, np.zeros_like(cp), 120.0)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.isfinite(values))


def test_the_fit_recovers_a_known_scale() -> None:
    sample = np.random.default_rng(7).logistic(0.0, 137.0, 200_000)
    assert fit_scale(sample) == pytest.approx(137.0, rel=0.03)


def test_the_fit_needs_a_label() -> None:
    with pytest.raises(ValueError, match="at least one centipawn label"):
        fit_scale(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="a label that is not zero"):
        fit_scale(np.zeros(16, dtype=np.float32))
