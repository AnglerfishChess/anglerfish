//! What a position is worth, and which of its moves deserve looking at.

mod material;
mod uniform;

use esca::{Facts, Move, Position, Score};

pub use material::Material;
pub use uniform::Uniform;

/// The centipawn score of delivering mate now. A mate `n` plies away is
/// `MATE - n`, and being mated `n` plies away is `-(MATE - n)`.
pub const MATE: i32 = 100_000;

/// The smallest centipawn score that still means a mate.
const MATE_MIN: i32 = MATE - 1_000;

/// `score` on the centipawn scale.
pub fn centipawns(score: Score) -> i32 {
    match score {
        Score::Cp(cp) => cp,
        Score::Mate(moves) if moves > 0 => MATE - (2 * moves - 1),
        Score::Mate(moves) => -MATE - 2 * moves,
    }
}

/// `centipawns` as a score, read back as a mate when it names one.
pub fn score(centipawns: i32) -> Score {
    if centipawns >= MATE_MIN {
        Score::Mate((MATE - centipawns + 1) / 2)
    } else if centipawns <= -MATE_MIN {
        Score::Mate(-(MATE + centipawns) / 2)
    } else {
        Score::Cp(centipawns)
    }
}

/// A judgement of a position.
pub trait Evaluator: Send + Sync {
    /// What `position` is worth, positive for the side to move. `facts` are
    /// the facts of `position`.
    fn value(&self, position: &Position, facts: &Facts) -> Score;

    /// The values of `items`, written to the first `items.len()` slots of
    /// `out`.
    ///
    /// # Panics
    /// If `out` is shorter than `items`.
    fn batch(&self, items: &[(Position, Facts)], out: &mut [Score]) {
        assert!(out.len() >= items.len(), "an output slot per item");
        for (slot, (position, facts)) in out.iter_mut().zip(items) {
            *slot = self.value(position, facts);
        }
    }
}

/// A prior over the moves of a position.
pub trait Policy: Send + Sync {
    /// A weight per move of `moves`, in order, written to the first
    /// `moves.len()` slots of `out`. The weights are non-negative and sum to
    /// one. `facts` are the facts of `position`, and every move of `moves` is
    /// legal there.
    ///
    /// # Panics
    /// If `out` is shorter than `moves`.
    fn priors(&self, position: &Position, facts: &Facts, moves: &[Move], out: &mut [f32]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mate_scores_survive_the_centipawn_scale() {
        for moves in [-5, -2, -1, 1, 2, 5] {
            let mate = Score::Mate(moves);
            assert_eq!(score(centipawns(mate)), mate, "{mate}");
        }
    }

    #[test]
    fn ordinary_scores_stay_centipawns() {
        for cp in [-9_000, -1, 0, 25, 9_000] {
            assert_eq!(score(cp), Score::Cp(cp));
            assert_eq!(centipawns(Score::Cp(cp)), cp);
        }
    }

    #[test]
    fn counts_a_mate_from_the_plies_it_takes() {
        // Mate in one is delivered on the next ply; being mated in one, on
        // the ply after our own move.
        assert_eq!(score(MATE - 1), Score::Mate(1));
        assert_eq!(score(MATE - 3), Score::Mate(2));
        assert_eq!(score(-MATE + 2), Score::Mate(-1));
        assert_eq!(score(-MATE + 4), Score::Mate(-2));
    }
}
