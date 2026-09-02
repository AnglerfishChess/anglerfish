# Anglerfish

AI-based chess toolkit.

## Environment usage

Install dependencies (creates `.venv` automatically):
```sh
uv sync --all-groups
```

Run the CLI:
```sh
uv run python -m pyanglerfish.anglerfish --help
```

## Rust

The Rust side is a Cargo workspace under [`rs_anglerfish/`](rs_anglerfish), holding
[`esca`](rs_anglerfish/esca) — the chess model: variants, positions, games and move text — and
[`anglerfish-core`](rs_anglerfish/anglerfish-core) — the engine, and the `anglerfish` UCI binary.

```sh
cd rs_anglerfish
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

The engine reads UCI commands on stdin; add `target/release/anglerfish` to any chess GUI. Set
`RUST_LOG=debug` for a trace on stderr.

## License

MIT — see [LICENSE](LICENSE).
