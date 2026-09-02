"""The Anglerfish trainer: dump to batches, batches to a two-head net.

The training loop and the checkpoint format live in `pyanglerfish.train`,
which is also the `python -m` entry point; importing it from here would make
that command warn.
"""

from .data import (
    SCALE_ROWS,
    DataConfig,
    EvalBatches,
    Sample,
    collate,
    fit_scale_on_dump,
    holdout_keys,
    position_key,
    resolve_groups,
)
from .model import NetConfig, TwoHeadNet
from .moves import move_index
from .scale import fit_scale, win_probability

__all__ = [
    "SCALE_ROWS",
    "DataConfig",
    "EvalBatches",
    "NetConfig",
    "Sample",
    "TwoHeadNet",
    "collate",
    "fit_scale",
    "fit_scale_on_dump",
    "holdout_keys",
    "move_index",
    "position_key",
    "resolve_groups",
    "win_probability",
]
