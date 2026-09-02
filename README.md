# Anglerfish

A chess engine that plays from a learned evaluation, and the trainer that produces it.

The engine is Rust: a UCI front end, a search, and an evaluator the net plugs into. The trainer is
Python: it reads the Lichess evaluation dump, turns positions into feature rows, and fits the net
the engine loads. The two meet at [`esca`](rs_anglerfish/esca) — the chess model that answers what
is true about a position, and the versioned schema of the row a net eats. `esca` is the one part
meant to be useful on its own: it is a crate and a Python package, published from this repository.

## Layout

```
pyanglerfish/       trainer, data tooling, CLI                      (Python)
tests/              tests for the Python side
rs_anglerfish/      Cargo workspace                                 (Rust)
  esca/             chess model, position facts, the wheel          (published)
  anglerfish-core/  engine: UCI, search, evaluator interface
docs/               architecture, the esca API, the feature schema
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

- [`docs/architecture.md`](docs/architecture.md) — the crates, and why they are split that way.
- [`docs/esca-api.md`](docs/esca-api.md) — the esca API in both languages.
- [`docs/esca-vocabulary.md`](docs/esca-vocabulary.md) — the terms everything is named after.
- [`docs/features.md`](docs/features.md) — the facts, and how they encode into a row.

## License

MIT — see [LICENSE](LICENSE).
