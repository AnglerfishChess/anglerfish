# Architecture

Two sides of one project: a Rust engine that plays, and a Python trainer that
fits the net it plays from. Both read chess through
[`esca`](https://github.com/AnglerfishChess/esca) — an external dependency,
the `esca` crate on crates.io and the `esca` package on PyPI.

---

## 1. Repository layout

```
anglerfish/
  pyproject.toml        hatchling, the pure-Python side
  pyanglerfish/         the trainer: data, scale, model, train  (training.md)
  tests/                the trainer's tests
  rs_anglerfish/        Cargo workspace root
    Cargo.toml          [workspace] members
    anglerfish-core/    engine: search, UCI, evaluator trait
    anglerfish-nn/      net loading and forward pass              (phase 2)
  data-external/        the Lichess dump, gitignored, symlinked in worktrees
  docs/
```

| Decision | Reason |
|---|---|
| Rust under `rs_anglerfish/`, not at the repo root | The repo root is already a hatchling Python project (`packages = ["pyanglerfish"]`). One directory per language keeps `cargo` commands, `target/` and the workspace root in one place, and mirrors the existing `pyanglerfish/`. |
| One workspace, several crates | The net's dependencies — inference backend, checkpoint format — must not reach the engine that can run without a net. |
| The UCI binary lives in `anglerfish-core` as `src/main.rs` | Same shape as anglerfry. A separate binary crate would buy nothing. |

Edition 2024, `rust-version = "1.85.1"` for every crate, matching anglerfry.
MSRV is raised only when a dependency forces it, and the bump is a release
note.

---

## 2. Dependency graph

```
                  esca (crates.io, PyPI)
                    |            |
            anglerfish-core   pyanglerfish
                    |
              anglerfish-nn         (phase 2)
```

1. `anglerfish-core` depends on `esca`, `log`, `env_logger` and `rand`, and on
   no chess library of its own.
2. `anglerfish-nn` depends on `esca` and the net format crate; `core` depends
   on `nn` behind a feature flag.
3. `esca` is taken with its default features: no PyO3, no I/O, no async in the
   engine's build.
4. The trainer takes the `esca` wheel as an ordinary dependency; nothing here
   builds it.
5. Versions on both sides are independent of `esca`'s; the coupling that
   matters is `schema_id`.
6. Every dependency in the tree carries a permissive licence (MIT/Apache/BSD-class).
7. `cargo metadata` licence check runs in CI on every crate.

---

## 3. `esca` — the chess model (external)

Answers "what is true about this position" and "what is true about this move":
`Variant` (`Classic`, `Chess960`), `Position`, `Game`, `Move`, `SquareSet`,
`Facts`, `Schema`, and the versioned `f32` row a net eats. Its API, vocabulary
and feature schema are documented in
[its own repository](https://github.com/AnglerfishChess/esca), which is also
where its tests, fixtures and benchmarks live.

What this project relies on beyond the types: `Position::facts_in` allocates
nothing and reuses a caller-owned `Scratch`, so a search node extracts facts
without touching the allocator; rows in the batch encoders are independent, so
the trainer parallelises and the library spawns no threads; `schema_id` pins
the row shape, so a checkpoint and an installed `esca` either agree or the load
fails.

---

## 4. `anglerfish-core` — the engine

Started as a copy of `anglerfry/main` (UCI front end, `Limits`, the search
thread, the strategy enum) and is then developed as a serious engine. Board,
moves and game state come from `esca`: `Game` behind the UCI `position`
command, `Position` inside the search. The binary is `anglerfish`; the library
beside it carries the traits a net implements.

| Item | Shape |
|---|---|
| `Evaluator` trait | `fn value(&self, pos: &Position, facts: &Facts) -> Score` and `fn batch(&self, items: &[(Position, Facts)], out: &mut [Score])`, the latter defaulting to a loop over `value`. A batching entry point exists from day one because an MCTS-style search needs it and an alpha-beta search may ignore it. |
| `Policy` trait | `fn priors(&self, pos: &Position, facts: &Facts, moves: &[Move], out: &mut [f32])`. |
| Material evaluator | The two-ply strategy's evaluation, behind the trait, as the reference implementation and the fallback when no net is loaded. `Uniform` is the matching policy. |
| Score scale | `eval::centipawns` and `eval::score` carry a `Score` in and out of the centipawn scale the search works in, where a mate `n` plies away is `MATE - n`. |
| Time management | As inherited; a real one lands with a real search. |
| Transposition table | Phase 2. |

Two UCI options: `Strategy`, as inherited, and `UCI_Chess960`, which selects
`esca::CHESS960` and with it king-to-rook castling in `bestmove` — classic
chess keeps the two-square spelling its GUIs expect. Setting it starts a fresh
game, since the rules a position is read under have changed.

### Search family: what is deferred

The choice between MCTS with PUCT and alpha-beta with policy-guided ordering
is open. The libraries must not decide it, so both requirement sets are met:

| Needs | MCTS | Alpha-beta |
|---|---|---|
| policy prior over legal moves | required per expanded node | used as an ordering key |
| batched evaluation | required (leaf batching) | optional |
| value scale | [−1, 1] win probability | centipawns, convertible |
| board copies per node | many | make/unmake or copies |
| facts per node | once per expansion | once per node, or incrementally |
| transposition table | optional | required |
| SEE / quiescence | not needed | needed, and lives in `core`, not in `esca` |

Both are served by: `Position::facts_in` being allocation-free, `Evaluator`
having a batch method, and `Score` being convertible between the two scales.

---

## 5. Python packaging

The root project is hatchling and pure Python — `pyanglerfish` and its tests —
and takes the compiled `esca` as a plain wheel:

```toml
[project]
dependencies = ["esca>=0.3,<0.4", …]
```

`uv sync --all-groups` installs it; a newer `esca` reaches the trainer through
`uv lock`, and a `schema_id` change through a retrained net.

---

## 6. `anglerfish-nn` — the net (phase 2)

Loads a checkpoint (weights plus the schema manifest), verifies `schema_id`
against `esca::Schema::v1().id()`, refuses a mismatch, and implements
`Evaluator` and `Policy`. Format and inference backend are chosen when there
is a net to load.

---

## 7. Testing

| Kind | What |
|---|---|
| **Engine** | Inherited from anglerfry: legality of every played move in self-play, under both variants; protocol behaviour driven over the binary's stdin and stdout; `Limits` read from any `go` line under `proptest`; UCI conformance via `uci-test-suite`. |
| **Trainer** | `pytest` over a synthetic dump: the read pipeline and its split filters, the fitted value scale, the net's shapes and its masking of padded moves, and the checkpoint round-trip including the `schema_id` refusal. |

---

## 8. Milestones

| Milestone | Contents |
|---|---|
| **M1** | `esca` core: `Variant` with `Classic` and `Chess960`, `Position`, `Game`, UCI and SAN move text, `Facts` with the v0 groups `state`, `material`, `pawns`, `pieces`, `king`, `mobility`, `attacks`, `tactics`, `planes`, `MoveFacts`, `Schema` and `schema_id`, batch encoding. Feature `python`: the module and its stubs. Feature `lichess`: the dump reader. `anglerfish-core` copied from anglerfry with the `Evaluator`/`Policy` traits and the material evaluator behind them. Differential, property, stability tests. Benchmarks. |
| **M2** | The trainer of [`training.md`](training.md): the dump pipeline, the two-head net, the fitted value scale and the checkpoint manifest. Then `anglerfish-nn`, the schema check on load, and the first trained net serving `Evaluator`. |
| **M3** | The search family, chosen on measurements: transposition table, time management, quiescence and SEE if alpha-beta wins; tree, PUCT and leaf batching if MCTS does. |
| **M4** | `esca` published to crates.io and PyPI: done, and continued in [its own repository](https://github.com/AnglerfishChess/esca). |
