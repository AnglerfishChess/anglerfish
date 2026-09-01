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
[`esca`](rs_anglerfish/esca) — the chess model: variants, positions, games and move text.

```sh
cd rs_anglerfish
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## License

MIT — see [LICENSE](LICENSE).
