# Training the two-head net

The Python side of the repository: positions out of the Lichess evaluation
dump, through `esca`, into shards on disk, and from those into a value head and
a policy head.

---

## 1. Data flow

Two steps, because the tensor path costs far more than the dump does and a
training run reads its input many times over.

```
lichess_db_eval.jsonl.zst
  └─ pyanglerfish.dump.read(path, min_depth=…)      zstandard + json, line by
       │   Row(fen, cp, mate, best)                 line, nothing held
       ▼
  pyanglerfish.build                                python -m pyanglerfish.build
       │   esca.tensors.facts(positions, groups=…)  one typed array per fact
       │   position.annotated_moves()               → pyanglerfish.moves.arrays
       ▼
  <out>/{train,holdout}-NNNNN.safetensors           facts packed, labels beside
  <out>/…​.safetensors.manifest.json                 the layout they were written under
       │
       ▼
  pyanglerfish.data.ShardBatches                    expanded again, batched
       │   Batch(facts, moves, move_mask, best, value)
       ▼
  pyanglerfish.model.TwoHeadNet
```

A record contributes one row: the deepest evaluation reaching `min_depth` and
the first line of it. Multi-PV is not used. The reader skips a record with no
such evaluation and one whose line is empty; the build drops a placement no
game can reach and a row whose labelled best move is not legal, and counts
both.

The dump writes `cp` and `mate` from White's point of view; the reader negates
those of a record with Black to move, so every label the trainer sees is
side-relative.

### The input is esca's typed arrays

Nothing is encoded, scaled or normalised on the way in, and nothing is written
in the mover's view. `esca.tensors.facts()` answers one array per fact, each
keeping the width and sign the fact was declared with — a flag is a `bool`, a
square set 64 flags, a per-colour value White then Black — and those arrays are
what a batch carries and what the net's first layer reads. The casting to the
compute type happens there, on the device.

`esca.tensors` covers a position's facts; a move's are built the same way in
[`pyanglerfish/moves.py`](../pyanglerfish/moves.py), from
`MoveFacts.to_dict()` under the rules esca's
[`features.md`](https://github.com/AnglerfishChess/esca/blob/main/docs/features.md)
states for the `move` group: 27 fields, a `bool` as a `bool`, an `i32` as an
`int32`, a role as its code and an absent role as −1.

### Shards

A shard is one safetensors file: the facts in esca's **packed** layout — a
square set is a `uint64` bitboard, every other flag one bit — the arrays of
every row's legal moves end to end with the cuts that separate them, and the
labels `cp`, `mate` and `best`. Beside it, `<shard>.manifest.json` records the
esca version, the form, the row count and every array's name, dtype and shape.
A load recomputes that manifest from the installed esca and refuses a shard
whose own does not match it, so a shard built under other facts is never fed to
a net as if it were these.

Shard size is the width of the training shuffle: `ShardBatches` permutes the
shards and the rows within each one.

### The split, and keeping it clean

Rows are split by their index in the reader's output: one index in
`holdout_every` is a held-out candidate, the rest are training candidates. The
split is the same on every pass over the same file at the same `min_depth`.

Positions repeat across the dump, so the index split alone would leak. Two
filters run on top of it, both keyed on the FEN without its clocks —
placement, side to move, castling rights and en-passant square, which
`position_key` cuts out:

- a held-out candidate is kept only the first time its key is seen, so the
  slice holds each position once;
- a training candidate whose key is held out is dropped.

`lichess_db_eval` carries one record per position, so on it the filters cost
nothing: over its first 2 000 000 rows at `min_depth 20`, all 31 250 held-out
candidates and all 1 968 750 training candidates have distinct keys and none
is dropped. The filters hold for a source that repeats itself.

The second filter needs every held-out key before the first training row is
written, so the build reads the dump through once for the keys and once for the
shards. That first pass is `zstandard` and `json` only — around 55 000 rows a
second — against 470 for the pass that encodes.

## 2. Targets

| Head | Target |
|---|---|
| value | `sigmoid(cp / s)` for a centipawn row; `0.5·(1 ± (1 − n/1000))` for a mate in `n`. Side-relative, in [0, 1]. |
| policy | The index of the labelled best move among the legal moves, in `annotated_moves()` order. |

A shard keeps `cp` and `mate` as the dump gave them, so the scale is a training
knob and not a property of the build.

`s` is fitted once, on the centipawn labels of the held-out shards, as the
maximum-likelihood scale of a zero-centred logistic
(`s = mean(|cp| · tanh(|cp| / 2s))`, a fixed point reached in a few dozen
iterations). Under that fit the targets spread evenly over [0, 1] instead of
piling up around 0.5. The fitted value goes into the checkpoint; passing
`--scale` skips the fit.

The dump's prefix is not the dump, so a small sample fits the prefix rather
than the corpus:

| labels fitted on | dump rows behind them | fitted `s` | against 1M |
|---|---|---|---|
| 20 000 | 1.5 M | 249.9 | +12.0 % |
| 100 000 | 7.6 M | 247.2 | +10.8 % |
| 400 000 | 30.0 M | 231.0 | +3.6 % |
| 1 000 000 | 74.2 M | 223.0 | — |

The default `--scale-rows` is 400 000: the smallest of those within 5 % of the
million-label fit.

## 3. The net

```
facts {group.field: array} ─ cast, flatten, concatenate ─ Linear·ReLU × trunk ─ Linear·ReLU ─ embedding (e)
                                                                                              ├─ Linear → value logit
                                                                                              └─ policy
moves {field: (b, m)} ─ cast, concatenate ──────────────────────────────────────────────────────┘

policy: relu(W_e·embedding + W_m·move) · w  → one score per move,
        −inf where the move is padding, softmax over the legal moves
```

The first layer takes the dict of typed arrays: each selected array is cast to
the net's own dtype where it already sits, flattened past the batch axis and
concatenated. `NetConfig.features` is the selection, a list of `group.field`
names as esca's catalogue spells them; the default is every array of every
group but `maps`, whose five arrays are ones `attacks` and `threats` already
carry. That default is 216 arrays and 4 914 values a position.

The policy head is one linear layer over the embedding joined with a move's 27
values, summed rather than concatenated so the join is never materialised.
Widths live in `NetConfig`; the defaults are a 1024–512 trunk, a 256-wide
embedding and a 128-wide move scorer.

Torch has no unsigned type wider than a byte, so a `uint16` array reaches it as
`int32` and a `uint32` as `int64`; `pyanglerfish.features.TORCH_DTYPES` is that
mapping, and no value changes under it.

Loss is `BCEWithLogits(value)` plus `--policy-weight` times
`CrossEntropy(policy)`. AdamW, gradient clipping at norm 5, cosine or constant
learning rate. The device is CUDA where there is one.

## 4. Checkpoint manifest

`torch.save` of one dict:

| Key | |
|---|---|
| `esca` | The esca version the net was trained against. |
| `features` | The array names the trunk reads, in the order they are joined. |
| `layout_hash` | A digest of those names with their dtypes and shapes. `load_checkpoint` refuses a checkpoint whose digest is not the one the installed esca answers for the same names. |
| `value_scale` | The fitted `s`. |
| `net` | `NetConfig` as a dict. |
| `step`, `state_dict` | Where training stood, and the weights. |

## 5. Running

The dump belongs in `data-external/`, fetched once with
`curl -O https://database.lichess.org/lichess_db_eval.jsonl.zst`, and is never
written to. Worktrees symlink it.

Build the shards, then train on them. Smoke, about half a minute on a CPU:

```
uv run python -m pyanglerfish.build \
    --dump data-external/lichess_db_eval.jsonl.zst \
    --out runs/smoke --min-depth 20 --max-rows 4000 --shard-rows 2048

uv run python -m pyanglerfish.train \
    --shards runs/smoke --steps 20 --batch-size 256 \
    --eval-every 10 --eval-batches 4 --log-every 5
```

A real run builds more of the dump and trains longer:

```
uv run python -m pyanglerfish.build \
    --dump data-external/lichess_db_eval.jsonl.zst \
    --out runs/v0 --min-depth 20 --max-rows 20000000

uv run python -m pyanglerfish.train \
    --shards runs/v0 --steps 200000 --batch-size 1024 \
    --eval-every 2000 --checkpoint runs/v0.pt
```

`--resume runs/v0.pt` continues from a checkpoint, keeping its scale and net
shape. `--groups state,material,…` builds shards of a subset of the groups, and
`--features state.in_check,…` trains on a subset of the arrays those shards
carry; the checkpoint records which.

The build encodes about 470 positions a second on one core, of which
`esca.tensors.facts` is 760 and the move arrays 1 430 — both an order of
magnitude above the 22 000 a second `Position.facts()` alone answers, because
each goes through the JSON form. It is one process; a bigger build is a shell
loop over slices of the dump into separate output directories.

## 6. Metrics

Reported on the held-out slice, per esca's
[`features.md`](https://github.com/AnglerfishChess/esca/blob/main/docs/features.md)
§7: value MAE and RMSE on
the probability scale and sign accuracy; policy top-1 and top-3 agreement with
the labelled best move; both heads' losses. Per-group ablation is not wired
up, but the array selection is a config knob, so a run per group is a shell
loop.
