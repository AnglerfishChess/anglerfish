//! The shallow negamax strategy.

use std::time::Instant;

use esca::{Game, Move, MoveList, Position, Scratch, Variant};

use super::{report, root_moves};
use crate::eval::{self, Evaluator, MATE};
use crate::search::Limits;

/// The deepest this strategy searches, whatever depth is asked of it.
const MAX_DEPTH: u8 = 4;

/// A score below every reachable one.
const WORST: i32 = -1_000_000;

/// The halfmove clock at which a search calls the game drawn.
const DRAW_CLOCK: u32 = 100;

/// The score of a position with no legal move, `ply` plies into the search: being mated, or a
/// stalemate. A mate found later scores below one found sooner.
fn terminal(position: &Position, ply: u8) -> i32 {
    if position.in_check() {
        -MATE + i32::from(ply)
    } else {
        0
    }
}

/// One negamax search: the rules it plays by, the judgement it asks, the limits it stops at,
/// and the buffers it works in.
struct Search<'a> {
    variant: &'a dyn Variant,
    evaluator: &'a dyn Evaluator,
    limits: &'a Limits,
    scratch: Scratch,
    nodes: u64,
}

impl Search<'_> {
    /// The score of `position` for the side to move, searched `depth` further plies from `ply`
    /// plies into the game, generating into one list of `stack` per ply. Returns early, and
    /// then meaninglessly, once the limits are spent.
    fn negamax(&mut self, position: &Position, depth: u8, ply: u8, stack: &mut [MoveList]) -> i32 {
        self.nodes += 1;
        if position.halfmove_clock() >= DRAW_CLOCK {
            return 0;
        }
        // A leaf: a node at depth zero, or one the stack has no room below. Its facts carry
        // the legal moves, so the terminal question is answered without generating them twice.
        let Some((moves, rest)) = stack.split_first_mut().filter(|_| depth > 0) else {
            let facts = position.facts_in(self.variant, &mut self.scratch);
            if facts.moves.is_empty() {
                return terminal(position, ply);
            }
            return eval::centipawns(self.evaluator.value(position, &facts));
        };
        moves.clear();
        self.variant.legal_moves(position, moves);
        if moves.is_empty() {
            return terminal(position, ply);
        }
        let mut best = WORST;
        for &played in moves.as_slice() {
            if self.limits.spent(self.nodes) {
                break;
            }
            let child = self.variant.play(position, played);
            best = best.max(-self.negamax(&child, depth - 1, ply + 1, rest));
        }
        best
    }
}

/// The move to play in `game`, searched as deeply as `limits` allow and chosen among the moves
/// they allow, or `None` when there is no move to play.
pub fn pick(game: &Game, limits: &Limits, evaluator: &dyn Evaluator) -> Option<Move> {
    let started = Instant::now();
    let moves = root_moves(game, limits);
    let mut best = *moves.first()?;
    let max_depth = limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
    let mut stack = vec![MoveList::new(); usize::from(max_depth)];
    let mut search = Search {
        variant: game.variant(),
        evaluator,
        limits,
        scratch: Scratch::new(),
        nodes: 0,
    };
    for depth in 1..=max_depth {
        let mut candidate = None;
        let mut best_score = WORST;
        for &played in moves.as_slice() {
            if limits.spent(search.nodes) {
                break;
            }
            let child = game.variant().play(game.position(), played);
            let score = -search.negamax(&child, depth - 1, 1, &mut stack);
            if score > best_score {
                best_score = score;
                candidate = Some(played);
            }
        }
        // An iteration cut short by the limits carries no usable score.
        if limits.spent(search.nodes) {
            break;
        }
        if let Some(candidate) = candidate {
            best = candidate;
            report(
                game,
                depth,
                eval::score(best_score),
                search.nodes,
                started.elapsed(),
                best,
            );
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use esca::classic;

    use super::*;
    use crate::eval::Material;
    use crate::uci::Go;

    /// The move picked in the position `fen`, searched `depth` plies.
    fn pick_in(fen: &str, depth: u8) -> Option<String> {
        let game = Game::from_fen(classic(), fen).expect("a legal position");
        let limits = Limits::new(
            &Go {
                depth: Some(depth),
                ..Go::default()
            },
            &game,
        );
        let played = pick(&game, &limits, &Material)?;
        Some(game.move_to_uci(played))
    }

    #[test]
    fn takes_the_free_piece() {
        assert_eq!(
            pick_in("4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1", 2).as_deref(),
            Some("e4d5")
        );
    }

    #[test]
    fn finds_mate_in_one() {
        assert_eq!(
            pick_in(
                "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
                2
            )
            .as_deref(),
            Some("h5f7")
        );
    }

    #[test]
    fn has_no_move_when_the_game_is_over() {
        assert_eq!(pick_in("7k/5KQ1/8/8/8/8/8/8 b - - 0 1", 2), None);
    }

    #[test]
    fn answers_a_chess960_castling_king_to_rook() {
        let game = Game::from_fen(esca::chess960(), "4k3/8/8/8/8/8/8/1KR5 w C - 0 1")
            .expect("a legal position");
        let limits = Limits::new(
            &Go {
                depth: Some(1),
                search_moves: vec!["b1c1".to_owned()],
                ..Go::default()
            },
            &game,
        );
        let played = pick(&game, &limits, &Material).expect("a legal move");
        assert!(played.is_castling());
        assert_eq!(game.move_to_uci(played), "b1c1");
    }
}
