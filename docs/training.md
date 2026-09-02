# Training the two-head net

The Python side of the repository: rows out of the Lichess evaluation dump,
through `esca`, into a value head and a policy head.

---

## 1. Data flow

```
lichess_db_eval.jsonl.zst
  └─ esca.lichess.batches(path, min_depth=…, groups=…)   Rust: zstd, JSON,
       │                                                 position encoding
       │   fens, features (n, w), cp (n,), mate (n,), best_moves
       ▼
  pyanglerfish.data.EvalBatches
       │   esca.encode_moves(fens) → per FEN the legal moves, their rows
       │                          stacked (total, 24), and the cuts (n + 1,)
       │   move_index(moves, best_move) → the policy target
       │   win_probability(cp, mate, scale) → the value target
       ▼
  dict of tensors: features (b, w), moves (b, m, 24), move_mask (b, m),
                   best (b,), value (b,)
```

A record contributes one row: the deepest evaluation reaching `min_depth`
and the first line of it. Multi-PV is not used. `esca` drops a record with no
such evaluation, an unreachable placement or an unreadable line; the dataset
drops a row whose labelled best move is absent from the legal move list and
counts it in `counts.unmatched`.

One `esca.encode_moves` call covers a whole read batch: the rows come back
stacked into one array with the offsets that cut it, so a sample holds a view
of it rather than an array of its own, and the move encoding runs on every
core the way the feature encoding does.

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
handed out, so `holdout_keys` reads the dump through once and keeps them in a
set — one extra pass, and memory linear in the held-out slice. `train()` makes
that pass and hands the set to the training split; the held-out split needs
only the keys it has already passed.

`counts` partitions the candidates a split saw: `kept`, `duplicate` (a
held-out repeat), `leaked` (a training row held out elsewhere) and
`unmatched`. Training rows pass through a reservoir shuffle buffer; held-out
rows are read in dump order.

## 2. Targets

| Head | Target |
|---|---|
| value | `sigmoid(cp / s)` for a centipawn row; `0.5·(1 ± (1 − n/1000))` for a mate in `n`. Side-relative, in [0, 1]. |
| policy | The index of the labelled best move among the legal moves, in `encode_moves` order. |

`s` is fitted once, on centipawn labels from the held-out candidates, as the
maximum-likelihood scale of a zero-centred logistic
(`s = mean(|cp| · tanh(|cp| / 2s))`, a fixed point reached in a few dozen
iterations). Under that fit the targets spread evenly over [0, 1] instead of
piling up around 0.5. The fitted value goes into the checkpoint; passing
`--scale` skips the fit.

The dump's prefix is not the dump, so a small sample fits the prefix rather
than the corpus:

| `--scale-rows` | dump rows read | fitted `s` | against 1M |
|---|---|---|---|
| 20 000 | 1.5 M | 249.9 | +12.0 % |
| 100 000 | 7.6 M | 247.2 | +10.8 % |
| 400 000 | 30.0 M | 231.0 | +3.6 % |
| 1 000 000 | 74.2 M | 223.0 | — |

The default is 400 000: the smallest of those within 5 % of the million-label
fit.

## 3. The net

```
features (w) ─ Linear·ReLU × trunk ─ Linear·ReLU ─ embedding (e)
                                                    ├─ Linear → value logit
                                                    └─ policy
moves (m, 24) ────────────────────────────────────────┘

policy: relu(W_e·embedding + W_m·move) · w  → one score per move,
        −inf where the move is padding, softmax over the legal moves
```

The policy head is one linear layer over the embedding joined with a move's
24 features, summed rather than concatenated so the join is never
materialised. Widths live in `NetConfig`; the defaults are a 1024–512 trunk,
a 256-wide embedding and a 128-wide move scorer.

Loss is `BCEWithLogits(value)` plus `--policy-weight` times
`CrossEntropy(policy)`. AdamW, gradient clipping at norm 5, cosine or constant
learning rate. The device is CUDA where there is one.

## 4. Checkpoint manifest

`torch.save` of one dict:

| Key | |
|---|---|
| `schema_id`, `schema_semver` | `esca.SCHEMA_ID` and the schema's semver at training time. `load_checkpoint` refuses a checkpoint whose `schema_id` is not the installed one. |
| `groups` | The schema groups the input row carries, in schema order. |
| `value_scale` | The fitted `s`. |
| `net` | `NetConfig` as a dict. |
| `step`, `state_dict` | Where training stood, and the weights. |

## 5. Running

The dump belongs in `data-external/`, fetched once with
`curl -O https://database.lichess.org/lichess_db_eval.jsonl.zst`, and is never
written to. Worktrees symlink it.

Smoke, about ten seconds on a CPU:

```
uv run python -m pyanglerfish.train \
    --dump data-external/lichess_db_eval.jsonl.zst \
    --max-rows 20000 --steps 20 --batch-size 256 \
    --eval-every 10 --eval-batches 4 --log-every 5
```

A real run drops `--max-rows`, raises `--steps`, and names a checkpoint:

```
uv run python -m pyanglerfish.train \
    --dump data-external/lichess_db_eval.jsonl.zst \
    --min-depth 20 --steps 200000 --batch-size 1024 \
    --eval-every 2000 --checkpoint runs/v0.pt
```

`--resume runs/v0.pt` continues from a checkpoint, keeping its scale and net
shape. `--groups state,material,…` trains on a subset of the schema; the
checkpoint records which. Reading and encoding the dump is Rust and parallel,
move encoding included. A run reads the dump once for the held-out keys and
once for the scale fit before the first optimiser step.

## 6. Metrics

Reported on the held-out slice, per `features.md` §7: value MAE and RMSE on
the probability scale and sign accuracy; policy top-1 and top-3 agreement with
the labelled best move; both heads' losses. Per-group ablation is not wired
up, but the group list is a config knob, so a run per group is a shell loop.
