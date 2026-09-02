# Crate architecture

The Rust side of Anglerfish: a Cargo workspace of small crates, one of which —
`esca` — is meant to be useful outside this project.

---

## 1. Repository layout

```
anglerfish/
  pyproject.toml        hatchling, the pure-Python side
  pyanglerfish/         the trainer: data, scale, model, train  (training.md)
  rs_anglerfish/        Cargo workspace root
    Cargo.toml          [workspace] members
    esca/               chess model + position facts + wheel     (public)
    anglerfish-core/    engine: search, UCI, evaluator trait      (internal)
    anglerfish-nn/      net loading and forward pass              (internal, phase 2)
  data-external/        the Lichess dump, gitignored, symlinked in worktrees
  docs/
```

| Decision | Reason |
|---|---|
| Rust under `rs_anglerfish/`, not at the repo root | The repo root is already a hatchling Python project (`packages = ["pyanglerfish"]`). One directory per language keeps `cargo` commands, `target/` and the workspace root in one place, and mirrors the existing `pyanglerfish/`. |
| One workspace, several crates | `esca` must be publishable and depend on nothing of ours; the engine and the net must not force their dependencies on it. |
| The UCI binary lives in `anglerfish-core` as `src/main.rs` | Same shape as anglerfry. A separate binary crate would buy nothing. |
| The Python binding and the dump reader are Cargo features of `esca`, not crates | Both are thin skins over the same types. A separate crate would re-export the whole API to add a decorator to it. |

Edition 2024, `rust-version = "1.85.1"` for every crate, matching anglerfry.
MSRV is raised only when a dependency forces it, and the bump is a release
note.

---

## 2. Dependency graph

```
   cozy-chess (MIT)   pyo3, numpy,     zstd, serde_json
          |            rayon            (feature
          |            (feature python)  lichess)
          |               |                 |
          +───────────── esca ──────────────+
                          |
                  anglerfish-core
                          |
                    anglerfish-nn         (phase 2)
```

Ten lines of rules:

1. `esca` depends on `cozy-chess` and nothing else of ours.
2. No `cozy_chess` item appears in `esca`'s public API, in either language.
3. The default build of `esca` has no PyO3, no I/O, no async, and no
   allocation in the hot path; `python` and `lichess` are off by default.
4. `anglerfish-core` depends on `esca`, `log`, `env_logger` and `rand`, and on
   no chess library of its own.
5. `anglerfish-nn` depends on `esca` and the net format crate; `core` depends
   on `nn` behind a feature flag.
6. No crate of ours enables `esca/python`; only the wheel build does.
7. `esca` never depends on `core` or `nn`.
8. The Rust crate `esca` and the Python package `esca` are one source tree and
   one version.
9. Every dependency in the tree carries a permissive licence (MIT/Apache/BSD-class).
10. `cargo metadata` licence check runs in CI on every crate.

---

## 3. `esca` — chess model and position facts (public, reusable)

Answers "what is true about this position" and "what is true about this move",
in a chess model of its own: `Variant` (`Classic`, `Chess960`), `Position`,
`Game`, `Move`, `SquareSet`, `Facts`, `Schema`. cozy-chess supplies board
representation and move generation as an implementation detail.

Two audiences, one computation: a reader who wants a readable question
answered (`facts.pawns.passed[Us]`), and a net that wants a flat `f32` row.

| Document | Contents |
|---|---|
| [`esca-api.md`](esca-api.md) | Every public signature, in Rust and in Python. |
| [`esca-vocabulary.md`](esca-vocabulary.md) | Every term the API and the docs use. |
| [`features.md`](features.md) | Feature definitions, encodings, group widths, schema versioning. |

| Public | Internal |
|---|---|
| The types above, `MoveFacts`, `GroupSet`, `Scratch`, the encoders, the `lichess` reader, the glossary | Attack-map construction, scratch layout, per-group writers, every cozy-chess type |

Cargo features: `python` (§5), `lichess` (dump reader), `serde` (manifest and
value serialisation).

A search node extracts facts through `Position::facts_in`, which allocates
nothing and reuses a caller-owned `Scratch`. Rows in the batch encoders are
independent; the caller parallelises, the crate spawns no threads.

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

## 5. Python packaging of `esca`

One source tree produces the crate and the wheel — maturin's mixed layout:

```
rs_anglerfish/esca/
  Cargo.toml            the crate
  pyproject.toml        maturin backend, project name `esca`
  src/                  Rust, including the PyO3 module behind feature `python`
  python/esca/
    __init__.py         re-exports the compiled extension
    __init__.pyi        what the package exports
    _esca.pyi           stubs for the whole compiled surface
    lichess.py          the dump batches, re-exported
    lichess.pyi
    py.typed
  python/tests/         the Python side's tests
```

Import name and distribution name are both `esca`. Wheels are built with
`--features python,lichess,pyo3/extension-module`, the last of which a test
binary must not have, because it resolves the interpreter's symbols itself;
`abi3` from the lowest supported CPython, so one wheel per platform covers
every version.

The root `anglerfish` project stays hatchling and pure-Python, and depends on
`esca` as a local editable source:

```toml
[project]
dependencies = ["esca", …]

[tool.uv.sources]
esca = { path = "rs_anglerfish/esca", editable = true }
```

`uv sync` then builds the extension in place, and a Rust change reaches the
trainer by re-running it. Publishing `esca` to PyPI and crates.io is M4 and
the owner's decision.

---

## 6. `anglerfish-nn` — the net (phase 2)

Loads a checkpoint (weights plus the schema manifest), verifies `schema_id`
against `esca::Schema::v0().id()`, refuses a mismatch, and implements
`Evaluator` and `Policy`. Format and inference backend are chosen when there
is a net to load.

---

## 7. Testing

| Kind | What |
|---|---|
| **Differential — facts** | A slow, readable Python reference of every feature (`tests/reference/`, driven by `tests/test_reference_v0.py`), over plain FEN parsing and explicit loops; it has no third-party chess dependency. Both implementations run on a corpus of ~20k positions sampled from the dump plus hand-picked cases (checks, promotions, en passant, opposite bishops, back-rank mates, bare-king endgames); every value must match exactly. The reference is the specification's executable form; when they disagree, `features.md` decides which one is wrong. |
| **Differential — rules** | Move generation and terminal conditions against published perft counts: the classic start position and the standard tricky FENs to depth 6, and Chess960 start positions to depth 5. Plus known-answer cases per variant: castling through and out of check, castling where king or rook destination coincides with an origin, en-passant pins, promotion under check, and every draw condition of `esca-vocabulary.md` §3. |
| **Property** | With `proptest` over random legal positions: mirror invariance (facts of a position equal facts of its colour-and-rank-mirrored twin with the sides exchanged); determinism; emitted length equals the declared width; every value finite and within [−1, 1]; every square-set-derived mask a subset of its base; `facts_in` output identical to `facts` output. |
| **Stability** | Golden fixtures: 231 fixed classic FENs and 60 Chess960 ones with their vectors, under `rs_anglerfish/esca/tests/data/` and regenerated by `cargo run --release --example fixtures`, stored per schema version. A changed output fails until the group version is bumped and a new fixture added. Old fixtures are kept and still checked while their group version is supported. |
| **Schema** | `schema_id` is recomputed in a test and compared with a checked-in constant; the canonical text is a golden file. |
| **Python** | The stubs typecheck against the compiled module; `pyrefly` runs over `python/esca/`. Round-trips: pickle, hash, and `Position`/`Move` equality. |
| **Engine** | Inherited from anglerfry: legality of every played move in self-play, under both variants; protocol behaviour driven over the binary's stdin and stdout; `Limits` read from any `go` line under `proptest`; UCI conformance via `uci-test-suite`. |
| **Benchmarks** | `criterion`: nanoseconds per position for the whole extractor and per group, on a fixed 10k corpus. Regressions above 10 % fail CI. |

---

## 8. Milestones

| Milestone | Contents |
|---|---|
| **M1** | `esca` core: `Variant` with `Classic` and `Chess960`, `Position`, `Game`, UCI and SAN move text, `Facts` with the v0 groups `state`, `material`, `pawns`, `pieces`, `king`, `mobility`, `attacks`, `tactics`, `planes`, `MoveFacts`, `Schema` and `schema_id`, batch encoding. Feature `python`: the module and its stubs. Feature `lichess`: the dump reader. `anglerfish-core` copied from anglerfry with the `Evaluator`/`Policy` traits and the material evaluator behind them. Differential, property, stability tests. Benchmarks. |
| **M2** | The trainer of [`training.md`](training.md): the dump pipeline, the two-head net, the fitted value scale and the checkpoint manifest. Then `anglerfish-nn`, the schema check on load, and the first trained net serving `Evaluator`. |
| **M3** | The search family, chosen on measurements: transposition table, time management, quiescence and SEE if alpha-beta wins; tree, PUCT and leaf batching if MCTS does. |
| **M4** | Publishing `esca` to crates.io and PyPI, after the API has survived M2. Owner's decision, per repo policy. |
