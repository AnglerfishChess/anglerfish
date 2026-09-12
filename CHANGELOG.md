# Changelog

## Unreleased

- The engine and the trainer live in this repository, and take `esca` from
  crates.io and PyPI like any other dependency. The library's own changelog
  continues at [AnglerfishChess/esca](https://github.com/AnglerfishChess/esca).
- Both sides move to `esca` 0.4, whose facts are typed, absolute-coloured and
  carry no move data. The engine reads material by `Colour` and takes a leaf's
  legal moves from the variant; the trainer's input is esca's typed tensors,
  one array per fact, cast to the compute type by the net's first layer instead
  of being encoded, scaled or written in the mover's view.
- The trainer reads the dump itself, with `zstandard` and `json`, and builds
  shards: one safetensors file of packed facts, move arrays and labels, with a
  manifest naming the layout it was written under, which a load checks. A
  training run reads shards rather than the dump, so the tensor cost is paid
  once. `python -m pyanglerfish.build` is the new entry point;
  `python -m pyanglerfish.train` takes `--shards` and `--features`.
- A checkpoint records the esca version, the array names the net reads and a
  digest of their types and shapes, in place of the schema id 0.3 pinned;
  checkpoints from before this are not loadable.

## 0.1.0 (2026-09-03)

First release, made from the repository this one was split out of; the engine
and the trainer shipped in it rather than to an index.

- `anglerfish-core`: the `anglerfish` UCI binary — protocol loop, bounded
  search, move-picking strategies, and the `Evaluator`/`Policy` traits a net
  will plug into (`Uniform` and `Material` until then).
