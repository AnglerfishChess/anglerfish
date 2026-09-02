"""The label-to-win-probability mapping and the scale it is fitted with."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["MATE_HORIZON", "fit_scale", "win_probability"]

#: Moves-to-mate beyond which a mate label carries no more weight.
MATE_HORIZON = 1000.0

_MIN_SCALE = 1.0
_MAX_SCALE = 10_000.0
_E_ABS_LOGISTIC = 2.0 * np.log(2.0)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def win_probability(
    cp: NDArray[np.float32] | NDArray[np.float64],
    mate: NDArray[np.float32] | NDArray[np.float64],
    scale: float,
) -> NDArray[np.float32]:
    """The value target in [0, 1], side-relative, one value per row.

    A row is a mate row where `mate` is not zero, and a centipawn row
    otherwise, matching the dump reader's convention. Centipawns pass through
    `sigmoid(cp / scale)`; a mate in `n` becomes `±(1 − n / MATE_HORIZON)` on
    the [−1, 1] scale, rescaled to [0, 1].
    """
    cp = np.asarray(cp, dtype=np.float64)
    mate = np.asarray(mate, dtype=np.float64)
    signed = np.sign(mate) * (1.0 - np.minimum(np.abs(mate), MATE_HORIZON) / MATE_HORIZON)
    return np.where(mate != 0.0, 0.5 * (1.0 + signed), _sigmoid(cp / scale)).astype(np.float32)


def fit_scale(cp: NDArray[np.float32] | NDArray[np.float64], *, tolerance: float = 1e-4) -> float:
    """The maximum-likelihood scale of a zero-centred logistic over `cp`.

    Under that fit `sigmoid(cp / scale)` is the sample's own probability
    integral transform, so the value targets spread evenly over [0, 1].
    Mate rows carry no centipawn label and must be left out. Raises
    `ValueError` for an empty or all-zero sample.
    """
    values = np.abs(np.asarray(cp, dtype=np.float64).ravel())
    if values.size == 0:
        raise ValueError("a scale needs at least one centipawn label")
    mean = float(values.mean())
    if mean == 0.0:
        raise ValueError("a scale needs a label that is not zero")
    # Stationary point of the log-likelihood: s = mean(|cp| · tanh(|cp| / 2s)).
    scale = mean / _E_ABS_LOGISTIC
    for _ in range(200):
        step = float(np.mean(values * np.tanh(values / (2.0 * scale))))
        step = min(max(step, _MIN_SCALE), _MAX_SCALE)
        if abs(step - scale) <= tolerance * scale:
            scale = step
            break
        scale = step
    return scale
