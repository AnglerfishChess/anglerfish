//! The uniformly random strategy.

use std::time::Duration;

use esca::{Game, Move, Score};
use rand::seq::IndexedRandom;

use super::{report, root_moves};
use crate::search::Limits;

/// A move `limits` allow in `game`, drawn uniformly, or `None` when there is none.
pub fn pick(game: &Game, limits: &Limits) -> Option<Move> {
    let moves = root_moves(game, limits);
    let chosen = *moves.as_slice().choose(&mut rand::rng())?;
    report(game, 1, Score::Cp(0), 1, Duration::ZERO, chosen);
    Some(chosen)
}
