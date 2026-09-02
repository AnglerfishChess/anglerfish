//! The ways Anglerfish picks a move, and the UCI option that selects one.

mod random;
mod two_ply;

use std::time::Duration;

use esca::{Game, Move, MoveList, Score};

use crate::eval::Material;
use crate::search::Limits;
use crate::uci;

/// A way of picking a move.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Strategy {
    /// Uniformly random among the legal moves.
    #[default]
    Random,
    /// Shallow negamax over an [`crate::eval::Evaluator`].
    TwoPly,
}

impl Strategy {
    /// The name of the UCI option selecting a strategy.
    pub const OPTION: &'static str = "Strategy";

    /// Every strategy, in the order the option offers them.
    const ALL: [Strategy; 2] = [Strategy::Random, Strategy::TwoPly];

    /// The option value naming this strategy.
    pub fn name(self) -> &'static str {
        match self {
            Strategy::Random => "random",
            Strategy::TwoPly => "two-ply",
        }
    }

    /// The strategy that `name` selects, if any.
    pub fn from_name(name: &str) -> Option<Strategy> {
        Strategy::ALL
            .into_iter()
            .find(|strategy| strategy.name().eq_ignore_ascii_case(name))
    }

    /// The `option` line offering the choice of strategy.
    pub fn option() -> String {
        let mut line = format!(
            "option name {} type combo default {}",
            Strategy::OPTION,
            Strategy::default().name()
        );
        for strategy in Strategy::ALL {
            line.push_str(" var ");
            line.push_str(strategy.name());
        }
        line
    }

    /// The move to play in `game` within `limits`, or `None` when the game is over there.
    /// Emits at least one `info` line whenever it returns a move.
    pub fn pick(self, game: &Game, limits: &Limits) -> Option<Move> {
        match self {
            Strategy::Random => random::pick(game, limits),
            Strategy::TwoPly => two_ply::pick(game, limits, &Material),
        }
    }
}

/// The moves in `game` a search may answer with: those `limits` name, else every legal one.
fn root_moves(game: &Game, limits: &Limits) -> MoveList {
    if limits.search_moves.is_empty() {
        game.legal_moves()
    } else {
        limits.search_moves.clone()
    }
}

/// Emits the `info` line of a search that settled on `best`.
fn report(game: &Game, depth: u8, score: Score, nodes: u64, elapsed: Duration, best: Move) {
    uci::send(&format!(
        "info depth {depth} time {} nodes {nodes} score {score} pv {}",
        elapsed.as_millis(),
        game.move_to_uci(best)
    ));
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use esca::{Variant, chess960, classic};

    use super::*;
    use crate::uci::Go;

    #[test]
    fn names_round_trip() {
        for strategy in Strategy::ALL {
            assert_eq!(Strategy::from_name(strategy.name()), Some(strategy));
        }
        assert_eq!(Strategy::from_name("nonsense"), None);
    }

    #[test]
    fn offers_every_strategy() {
        assert_eq!(
            Strategy::option(),
            "option name Strategy type combo default random var random var two-ply"
        );
    }

    /// Picks with every strategy, checking that the answer is one of the named moves.
    #[test]
    fn every_strategy_answers_within_searchmoves() {
        let game = Game::new(classic());
        let limits = Limits::new(
            &Go {
                depth: Some(2),
                search_moves: ["a2a3", "h2h3"].map(str::to_owned).into(),
                ..Go::default()
            },
            &game,
        );
        for strategy in Strategy::ALL {
            let played = game.move_to_uci(strategy.pick(&game, &limits).expect("a legal move"));
            assert!(["a2a3", "h2h3"].contains(&played.as_str()), "{played}");
        }
    }

    /// Plays both strategies against each other from `variant`'s `seed` start position,
    /// checking that every move they pick is one the rules allow.
    fn self_play(variant: Arc<dyn Variant>, seed: u64) {
        let mut game = Game::with_seed(variant, seed);
        for ply in 0..40 {
            if game.outcome().is_some() {
                break;
            }
            let limits = Limits::new(
                &Go {
                    depth: Some(2),
                    ..Go::default()
                },
                &game,
            );
            let strategy = if ply % 2 == 0 {
                Strategy::Random
            } else {
                Strategy::TwoPly
            };
            let played = strategy.pick(&game, &limits).expect("a legal move");
            game.play(played).expect("a legal move");
        }
    }

    #[test]
    fn self_play_stays_legal() {
        self_play(classic(), 0);
    }

    #[test]
    fn self_play_stays_legal_in_chess960() {
        // A shuffled back rank, so castling is not the classic geometry.
        self_play(chess960(), 42);
    }
}
