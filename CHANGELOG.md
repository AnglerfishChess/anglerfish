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
- `pawns` gains the structure a player names out loud: the longest chain and
  whether its base is under attack, the majority on each wing, the holes a
  side can never cover again and the enemy minors sitting on them, fixed and
  blockaded pawns, the lead passer's distance to promotion, both kings'
  distances to its promotion square, whether the defending king is in its
  square and whether its path is empty, the files semi-open at the enemy king,
  and the backward pawns the enemy can actually attack.
- `pieces.knights_on_outpost` becomes `pieces.minors_on_outpost`, which counts
  bishops too; `king.king_distance` drops the two distances two kings can
  never stand at; `state.repetition_available_them` is gone.
- `pieces` gains what makes a minor good or bad and a rook worth its rank: the
  own fixed pawns standing on an own bishop's colour, the bishop pair against a
  knight pair as one signed value, a rook on the seventh with the enemy king
  shut on the eighth, and the units with nowhere safe to go, by count and by
  value. A trapped unit is `mobility.immobile_pieces` with safety asked of the
  squares it does reach.
- `attacks` gains the value twins `attacked_count`, `attacked_value`,
  `en_prise_value` and `pinned_value`.
- The new `placement` group: the raw board as twelve 64-square planes in the
  mover's view, so a run measuring the augmentation against the board alone
  selects a group instead of building its own input. `Facts::placement`.
- The `endgame` group fills in: how central each king stands, the plies each
  side's leading passer still needs and the difference between them, the
  opposition the kings stand in, a king on a key square of its own passer, a
  wrong-colour bishop, and the three drawn material configurations
  `material.insufficient_material` deliberately excludes. `Facts::endgame`,
  with the `Opposition` and `DrawishMaterial` enums. The opposition is no
  longer on the excluded list of `features.md` §4.
- `history.last_move_was_check` is gone: with a history it is `state.in_check`
  read from the other end, and a check cannot survive the move that answers
  it.
- Static exchange evaluation: `Position::see` and `Position::see_capture`, the
  new `exchange` group over them, and `tactics.winning_capture_available`,
  `winning_capture_max_gain`, `equal_capture_count` and `losing_capture_count`
  redefined from the exchange instead of the 1-ply victim-versus-capturer
  test. SEE is no longer on the excluded list of `features.md` §4.
- The new `threats` group, `Facts::threats`: what each side stands to lose,
  read from the exchange on each of its own units — which are threatened, what
  they are worth and the largest gain against them — plus the units a lesser
  enemy unit attacks, the overloaded and removable defenders, the loose units,
  the units more cheap attackers bear on than defenders, and the slider
  geometry a threat comes from: x-rays through an enemy unit, batteries, and
  batteries whose line meets the enemy king ring.
- The `move` schema grows from 24 values to 40. Each legal move now also
  carries its SEE, the largest one it leaves us next, whether it moves an
  attacked unit or interposes on a check, whether it advances or creates a
  passer, the isolated, doubled and backward pawns it makes, a king file it
  opens for us, what it does to both king rings and to both hanging counts,
  whether it leaves a piece hanging, and whether it uncovers a slider.
  `esca.MOVE_WIDTH` and `MoveFacts::WIDTH` say 40.
- `king` gains the defensive half of the siege and the exposure a player reads
  off the board: the pieces defending their own king's ring and their weight,
  the besieging weight less that, the open rays off the king, whether it has
  luft, which side it reads as having castled to, and whether the two kings
  stand on opposite wings. `KingFacts` carries them, with the `CastledSide`
  enum and `KingFacts::ring_attacker_surplus`; `castled_side` joins the
  features `features.md` §4 keeps to classic chess.
- `tactics` gains six bits per side: a check that also captures safely, a
  discovery onto the enemy queen, a back-rank mate threat, a quiet move that
  leaves more to be won than stands to be won now, a side with no safe move at
  all, and a promotion the exchange on its square makes profitable.
- The move row is a section of the schema of its own, named `move`, with a
  version, named features, widths and encodings, and its text folded into the
  one `schema_id`. A checkpoint that stores the id now refuses a move row of
  another shape as surely as a position row of one. `Schema::moves`,
  `GroupSpec::canonical` and `GroupSpec::feature`; `esca.MOVE_SCHEMA` and
  `Schema.moves()` in Python.
- `material` gains the bishop-pair imbalance and the piece-value difference,
  `mobility` the safe-mobility difference per type, and `planes` a ninth
  plane: where their threatened units stand.

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
