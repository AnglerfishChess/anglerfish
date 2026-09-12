"""The Anglerfish trainer: dump to shards, shards to a two-head net.

The training loop and the checkpoint format live in `pyanglerfish.train` and
the shard build in `pyanglerfish.build`, each also a `python -m` entry point;
importing either from here would make that command warn.
"""

from .data import SCALE_ROWS, Batch, DataConfig, ShardBatches, fit_scale_on_shards
from .dump import Row
from .features import DEFAULT_GROUPS, default_names, layout_hash, width
from .model import NetConfig, TwoHeadNet
from .moves import MOVE_FIELDS, MOVE_WIDTH, move_index
from .scale import fit_scale, win_probability
from .shards import Shard

__all__ = [
    "DEFAULT_GROUPS",
    "MOVE_FIELDS",
    "MOVE_WIDTH",
    "SCALE_ROWS",
    "Batch",
    "DataConfig",
    "NetConfig",
    "Row",
    "Shard",
    "ShardBatches",
    "TwoHeadNet",
    "default_names",
    "fit_scale",
    "fit_scale_on_shards",
    "layout_hash",
    "move_index",
    "width",
    "win_probability",
]
