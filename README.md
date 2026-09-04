# Anglerfish

[![CI](https://github.com/AnglerfishChess/anglerfish/actions/workflows/ci.yml/badge.svg)](https://github.com/AnglerfishChess/anglerfish/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A chess engine that plays from a learned evaluation, and the trainer that produces it.

The engine is Rust: a UCI front end, a search, and an evaluator the net plugs into. The trainer is
Python: it reads the Lichess evaluation dump, turns positions into feature rows, and fits the net
the engine loads.

Relies on [`esca`](https://github.com/AnglerfishChess/esca) — the chess library both sides read
positions through: rules, position facts, PGN, opening books and a UCI client, as the `esca` crate
and the `esca` Python package.

## Layout

```
pyanglerfish/       trainer, data tooling, CLI                      (Python)
tests/              tests for the Python side
rs_anglerfish/      Cargo workspace                                 (Rust)
  anglerfish-core/  engine: UCI, search, evaluator interface
docs/               architecture and training
data-external/      the Lichess dump; gitignored, symlinked in worktrees
```

## Python

```sh
uv sync --all-groups
uv run pytest
uvx ruff check .
uvx ruff format --check .
uvx pyrefly check
```

Train the net over the Lichess evaluation dump — see [`docs/training.md`](docs/training.md):
```sh
uv run python -m pyanglerfish.train --help
```

## Rust

```sh
cd rs_anglerfish
cargo build --release
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

The engine reads UCI commands on stdin; add `rs_anglerfish/target/release/anglerfish` to any chess
GUI. Set `RUST_LOG=debug` for a trace on stderr. Protocol conformance is checked with
[uci-test-suite](https://github.com/AnglerfishChess/uci-test-suite):

```sh
uvx uci-test-suite ./target/release/anglerfish
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the engine, the trainer, and how they are split.
- [`docs/training.md`](docs/training.md) — the dump pipeline, the net and the checkpoint.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Lichess, for the evaluation dump and the game database; Stockfish and Leela Chess Zero, the engines
this one spars with and is measured against. The library's own credits are in
[esca's README](https://github.com/AnglerfishChess/esca#acknowledgements).
