# Changelog

## Unreleased

- The v1 schema, `schema_semver` 1.0.0: fourteen groups in the order
  `features.md` §2 gives them. `Schema::v1` replaces `Schema::v0`, which is
  gone, as is every net trained on it; Python names it `esca.SCHEMA_V1` and
  `esca.SCHEMA`.
- The clock and repetition facts move out of `state` into the new `history`
  group, so a training source without clocks omits one group name.
  `Facts::history` carries them; `StateFacts` no longer does. The group also
  says what the recent plies did: captures and checks in the last eight, the
  plies since either, the material trend over them, and what the last move
  moved, took and gave. A `Game` supplies them; a bare `Position` does not.
- `pieces.knights_on_outpost` becomes `pieces.minors_on_outpost`, which counts
  bishops too; `king.king_distance` drops the two distances two kings can
  never stand at; `state.repetition_available_them` is gone.
- `attacks` gains the value twins `attacked_count`, `attacked_value`,
  `en_prise_value` and `pinned_value`.
- The new `placement` group: the raw board as twelve 64-square planes in the
  mover's view, so a run measuring the augmentation against the board alone
  selects a group instead of building its own input. `Facts::placement`.
- Static exchange evaluation: `Position::see` and `Position::see_capture`, the
  new `exchange` group over them, and `tactics.winning_capture_available`,
  `winning_capture_max_gain`, `equal_capture_count` and `losing_capture_count`
  redefined from the exchange instead of the 1-ply victim-versus-capturer
  test. SEE is no longer on the excluded list of `features.md` §4.

## 0.1.0 (2026-09-03)

First release. `esca` is published to crates.io and PyPI; the engine and the
trainer ship in the repository only.

- `esca`: `Position`, `Game` and the `Variant` trait, with `Classic` and
  `Chess960` behind it; FEN and EPD, X-FEN and Shredder-FEN castling, legal
  move generation, UCI and SAN move text, and every terminal and claimable
  outcome.
- `Facts`, the v0 schema: nine groups of side-relative position facts plus
  per-move facts, a `schema_id` pinned by golden fixtures, and batch encoders
  that write 1065-wide `f32` rows.
- The `esca` Python package: an abi3-py312 wheel built from the crate, typed,
  with NumPy batch encoding.
- The `lichess` reader: streams the Lichess evaluation dump into encoded
  batches with their targets.
- `anglerfish-core`: the `anglerfish` UCI binary — protocol loop, bounded
  search, move-picking strategies, and the `Evaluator`/`Policy` traits a net
  will plug into (`Uniform` and `Material` until then).
