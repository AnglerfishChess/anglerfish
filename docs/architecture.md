# Crate architecture

The Rust side of Anglerfish: a Cargo workspace of small crates on
[cozy-chess](https://crates.io/crates/cozy-chess), one of which is meant to be
useful outside this project.

`FACTS` below stands for the position-facts crate, whose name is chosen from
`naming.md`.

---

## 1. Repository layout

```
anglerfish/
  pyproject.toml        hatchling, the pure-Python side
  pyanglerfish/         trainer, data tooling, CLI
  rs_anglerfish/        Cargo workspace root
    Cargo.toml          [workspace] members
    FACTS/              position facts + heuristics library   (public)
    anglerfish-core/    engine: search, UCI, evaluator trait  (internal)
    anglerfish-py/      PyO3 module                           (internal)
    anglerfish-data/    Lichess dump reader                   (internal, phase 2)
    anglerfish-nn/      net loading and forward pass          (internal, phase 2)
  data-external/        the Lichess dump, gitignored, symlinked in worktrees
  docs/
```

| Decision | Reason |
|---|---|
| Rust under `rs_anglerfish/`, not at the repo root | The repo root is already a hatchling Python project (`packages = ["pyanglerfish"]`). One directory per language keeps `cargo` commands, `target/` and the workspace root in one place, and mirrors the existing `pyanglerfish/`. |
| One workspace, several crates | The facts library must be publishable and depend on nothing of ours; the engine and the binding must not force their dependencies on it. |
| The UCI binary lives in `anglerfish-core` as `src/main.rs` | Same shape as anglerfry. A separate binary crate would buy nothing. |
| `anglerfish-py` is built by maturin and installed as a path dependency via `tool.uv.sources` | The trainer imports the same extractor the engine runs. The root `pyproject.toml` stays hatchling and pure-Python. |

Edition 2024, `rust-version = "1.85.1"` for every crate, matching anglerfry.
MSRV is raised only when a dependency forces it, and the bump is a release
note.

---

## 2. Dependency graph

```
        cozy-chess (MIT)
              |
            FACTS  ────────────────┐
           /     \                 |
  anglerfish-core  anglerfish-py   |
        |                |         |
  anglerfish-nn    anglerfish-data ┘   (phase 2)
```

Ten lines of rules:

1. `FACTS` depends on `cozy-chess` and nothing else of ours.
2. `FACTS` has no PyO3, no async, no I/O, no allocation in the hot path.
3. `anglerfish-core` depends on `FACTS`, `cozy-chess`, `log`.
4. `anglerfish-nn` depends on `FACTS` and the net format crate; `core`
   depends on `nn` behind a feature flag.
5. `anglerfish-py` depends on `FACTS`, `anglerfish-data`, `pyo3`, `numpy`.
6. `anglerfish-data` depends on a zstd reader and a JSON parser.
7. Nothing depends on `anglerfish-py`.
8. `FACTS` never depends on `core`, `nn`, `data` or `py`.
9. Every dependency in the tree carries a permissive licence (MIT/Apache/BSD-class).
10. `cargo metadata` licence check runs in CI on every crate.

---

## 3. `FACTS` — position facts and heuristics (public, reusable)

Answers "what is true about this position" and "what is true about this move",
for a `cozy_chess::Board`. Two audiences: a reader who wants a readable
question answered (`facts.pawns().passed(Side::Us)`), and a net that wants a
flat `f32` vector. Both are served from one computation.

### Building

```rust
pub struct Facts { /* … */ }

/// Scratch buffers reused across positions: attack maps, pawn spans, the move list.
pub struct Scratch { /* … */ }

pub struct Context<'a> {
    /// Halfmove clock, when the caller knows it.
    pub halfmove_clock: Option<u8>,
    /// Zobrist hashes of earlier positions in the game, for repetition facts.
    pub history: Option<&'a [u64]>,
}

impl Facts {
    pub fn new(board: &Board) -> Facts;
    pub fn with_context(board: &Board, ctx: &Context<'_>) -> Facts;
    pub fn in_scratch(board: &Board, ctx: &Context<'_>, scratch: &mut Scratch) -> Facts;
}
```

`in_scratch` is the search-node entry point: no allocation, buffers reused.

### Reading

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side { Us, Them }

impl Facts {
    pub fn state(&self) -> State;
    pub fn material(&self) -> Material;
    pub fn pawns(&self) -> Pawns;
    pub fn pieces(&self) -> Pieces;
    pub fn king(&self) -> King;
    pub fn mobility(&self) -> Mobility;
    pub fn attacks(&self) -> Attacks;
    pub fn tactics(&self, side: Side) -> Tactics;
}

impl Pawns {
    pub fn passed(&self, side: Side) -> BitBoard;
    pub fn backward(&self, side: Side) -> BitBoard;
    pub fn open_files(&self) -> FileSet;
    pub fn semi_open_files(&self, side: Side) -> FileSet;
    pub fn islands(&self, side: Side) -> u8;
}

impl Attacks {
    pub fn by(&self, side: Side) -> BitBoard;
    pub fn by_pawns(&self, side: Side) -> BitBoard;
    pub fn attackers_of(&self, square: Square, side: Side) -> BitBoard;
    pub fn is_hanging(&self, square: Square) -> bool;
    pub fn is_en_prise(&self, square: Square) -> bool;
}

impl King {
    pub fn ring(&self, side: Side) -> BitBoard;
    pub fn ring_attack_weight(&self, side: Side) -> u16;
    pub fn shelter(&self, side: Side) -> [Option<u8>; 3];
}

impl Tactics {
    pub fn gives_check(&self) -> bool;
    pub fn safe_checks(&self) -> u8;
    pub fn mate_in_1(&self) -> bool;
    pub fn forks(&self) -> u8;
}
```

Accessors are cheap reads of already-computed state; the work happens in
`Facts::new`. Definitions are those in `features.md` §1, and the doc comments
repeat them.

### Per-move annotations

```rust
pub struct MoveFacts { /* … */ }

impl Facts {
    pub fn annotate(&self, board: &Board, mv: Move) -> MoveFacts;
    pub fn annotate_all(&self, board: &Board, out: &mut Vec<(Move, MoveFacts)>);
}

impl MoveFacts {
    pub fn is_capture(&self) -> bool;
    pub fn victim(&self) -> Option<Piece>;
    pub fn gives_check(&self) -> bool;
    pub fn gives_safe_check(&self) -> bool;
    pub fn is_safe(&self) -> bool;
    pub fn captures_hanging(&self) -> bool;
    pub fn escapes_attack(&self) -> bool;
    pub fn forks(&self) -> u8;
}
```

### Flat encoding

```rust
pub struct Schema { /* groups, versions, widths, offsets */ }
pub struct GroupSet(u16);   // bitset over the schema's groups

pub const SCHEMA: &Schema;

impl Schema {
    pub fn id(&self) -> [u8; 16];
    pub fn width(&self) -> usize;
    pub fn width_of(&self, groups: GroupSet) -> usize;
    pub fn groups(&self) -> &[GroupSpec];
    /// The canonical text the id hashes, for diagnostics.
    pub fn canonical(&self) -> String;
}

impl Facts {
    pub fn write_into(&self, groups: GroupSet, out: &mut [f32]) -> usize;
    pub fn to_vec(&self, groups: GroupSet) -> Vec<f32>;
}

impl MoveFacts {
    pub const WIDTH: usize = 24;
    pub fn write_into(&self, out: &mut [f32]);
}
```

`write_into` panics on a short slice — a length mismatch is a caller bug, not
a runtime condition.

### Batch

```rust
/// Writes one row per FEN, row-major, into `out` of length `fens.len() * width`.
pub fn extract_fens(
    fens: &[&str],
    groups: GroupSet,
    out: &mut [f32],
) -> Result<(), (usize, ParseError)>;

pub fn extract_boards(boards: &[Board], groups: GroupSet, out: &mut [f32]);
```

Rows are independent; the caller parallelises, the crate does not spawn
threads.

### Surface

| Public | Internal |
|---|---|
| `Facts`, `MoveFacts`, group structs, `Side`, `Schema`, `GroupSet`, batch functions, the glossary in the docs | attack-map construction, scratch layout, the per-group writers |

`FACTS` is not API-compatible with any existing library and does not try to
be. It re-exports nothing from `cozy-chess`: callers already have `Board`,
`BitBoard`, `Square`.

Optional features: `serde` (manifest serialisation only), `std` (default; the
core is `no_std`-friendly since `cozy-chess` is).

---

## 4. `anglerfish-core` — the engine

Starts as a copy of `anglerfry/main` (UCI front end, `Limits`, the search
thread, the strategy enum) and is then developed as a serious engine. What
changes immediately:

| Item | Shape |
|---|---|
| `Evaluator` trait | `fn value(&self, board: &Board, facts: &Facts) -> Score` and `fn batch(&self, positions: &[(Board, Facts)], out: &mut [Score])`. A batching entry point exists from day one because an MCTS-style search needs it and an alpha-beta search may ignore it. |
| `Policy` trait | `fn priors(&self, board: &Board, facts: &Facts, moves: &[Move], out: &mut [f32])`. |
| Material evaluator | The two-ply strategy's evaluation, behind the trait, as the reference implementation and the fallback when no net is loaded. |
| Time management | As inherited; a real one lands with a real search. |
| Transposition table | Phase 2. |

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
| SEE / quiescence | not needed | needed, and lives in `core`, not in `FACTS` |

Both are served by: `Facts::in_scratch` being allocation-free, `Evaluator`
having a batch method, and `Score` being convertible between the two scales.

---

## 5. `anglerfish-py` — the PyO3 module

Built with maturin, imported as `anglerfish_facts`.

```python
import anglerfish_facts as af

af.SCHEMA_ID                      # "b7f0…", 32 hex chars
af.schema()                       # [{"name","version","width","offset"}, …]
af.WIDTH                          # 1065

# (n, width) float32, allocated once in Rust, no copy on the way out
x = af.extract(fens, groups=["state", "material", "pawns"])
af.extract_into(fens, out)        # writes into a caller-owned array

# (m, 24) float32 plus the moves, for the policy head
moves, mx = af.extract_moves(fen)

# streaming the dump: yields (fens, features, cp, mate, best_moves) batches
for batch in af.iter_dump(path, batch_size=8192, min_depth=20):
    ...
```

| Contract | |
|---|---|
| Output arrays are `numpy` `float32`, C-contiguous, allocated by Rust and handed over without a copy. | |
| Batch calls release the GIL and parallelise rows with rayon. | |
| A malformed FEN raises `ValueError` naming the row index. | |
| The module exposes the extractor and the dump reader only: no board object, no move generation, no search, no training. The single code path is the point. | |

---

## 6. `anglerfish-data` — the Lichess evaluation dump (phase 2)

The dump is Zstandard-compressed JSON lines:

```json
{"fen":"7r/1p3k2/… b - -",
 "evals":[{"pvs":[{"cp":69,"line":"f7g7 e6e2 …"}, …],"knodes":4189972,"depth":46}, …]}
```

| Fact | Consequence |
|---|---|
| FENs have 4 fields: no halfmove clock, no move number | The reader appends `" 0 1"` before parsing. See `features.md` §5. |
| Several `evals` per position, differing in depth and knodes | The reader exposes all of them; picking one is the trainer's policy, not the reader's. |
| `pvs` is multi-PV: each entry is `cp` or `mate` plus a UCI line | The first token of each line is a labelled move; that is the policy target. |
| 10.5 GiB compressed | Streaming only; the reader never holds the file. |

```rust
pub struct Record { pub fen: String, pub evals: Vec<Eval> }
pub struct Eval { pub depth: u32, pub knodes: u64, pub pvs: Vec<Pv> }
pub struct Pv { pub score: Score, pub line: String }

pub fn read(path: &Path) -> io::Result<impl Iterator<Item = io::Result<Record>>>;
```

Phase 1 uses the existing Python `zstandard` reader and passes FEN batches to
`af.extract`. This crate lands when that reader is the bottleneck, measured.

---

## 7. `anglerfish-nn` — the net (phase 2)

Loads a checkpoint (weights plus the schema manifest), verifies `schema_id`
against `FACTS::SCHEMA.id()`, refuses a mismatch, and implements `Evaluator`
and `Policy`. Format and inference backend are chosen when there is a net to
load.

---

## 8. Testing

| Kind | What |
|---|---|
| **Differential** | A slow, readable Python reference of every feature, over plain FEN parsing and explicit loops; it has no third-party chess dependency. Both implementations run on a corpus of ~20k positions sampled from the dump plus hand-picked cases (checks, promotions, en passant, opposite bishops, back-rank mates, bare-king endgames); every value must match exactly. The reference is the specification's executable form; when they disagree, `features.md` decides which one is wrong. |
| **Property** | With `proptest` over random legal positions: mirror invariance (facts of a position equal facts of its colour-and-rank-mirrored twin with the sides exchanged); determinism; emitted length equals the declared width; every value finite and within [−1, 1]; every bitboard-derived mask a subset of its base; `in_scratch` output identical to `new` output. |
| **Stability** | Golden fixtures: 1000 fixed FENs and their vectors, stored per schema version. A changed output fails until the group version is bumped and a new fixture added. Old fixtures are kept and still checked while their group version is supported. |
| **Schema** | `schema_id` is recomputed in a test and compared with a checked-in constant; the canonical text is a golden file. |
| **Engine** | Inherited from anglerfry: legality of every played move in self-play, UCI conformance via `uci-test-suite`. |
| **Benchmarks** | `criterion`: nanoseconds per position for the whole extractor and per group, on a fixed 10k corpus. Regressions above 10 % fail CI. |

---

## 9. Milestones

| Milestone | Contents |
|---|---|
| **M1** | `FACTS` with groups `state`, `material`, `pawns`, `pieces`, `king`, `mobility`, `attacks`, `tactics`, `planes`; `Schema` and `schema_id`; `MoveFacts`; batch extraction. `anglerfish-py` with `extract`, `extract_into`, `schema`. `anglerfish-core` copied from anglerfry with `Evaluator`/`Policy` traits and the material evaluator behind them. Differential, property, stability tests. Benchmarks. |
| **M2** | `anglerfish-nn`, checkpoint format, schema check on load, the first trained net serving `Evaluator`. `anglerfish-data`. `iter_dump` in the binding. |
| **M3** | The search family, chosen on measurements: transposition table, time management, quiescence and SEE if alpha-beta wins; tree, PUCT and leaf batching if MCTS does. |
| **M4** | Publishing `FACTS` to crates.io, after the API has survived M2. Owner's decision, per repo policy. |
