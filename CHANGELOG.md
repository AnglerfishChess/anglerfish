# Changelog

## Unreleased

- The engine and the trainer live in this repository, and take `esca` from
  crates.io and PyPI like any other dependency. The library's own changelog
  continues at [AnglerfishChess/esca](https://github.com/AnglerfishChess/esca).

## 0.1.0 (2026-09-03)

First release, made from the repository this one was split out of; the engine
and the trainer shipped in it rather than to an index.

- `anglerfish-core`: the `anglerfish` UCI binary — protocol loop, bounded
  search, move-picking strategies, and the `Evaluator`/`Policy` traits a net
  will plug into (`Uniform` and `Material` until then).
