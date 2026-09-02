//! Limits of a single `go`, and the thread that runs it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use esca::{Colour, Game, Move, MoveList};
use log::{error, warn};

use crate::strategy::Strategy;
use crate::uci::{self, Go};

/// Moves the clock is expected to have to cover.
const MOVES_AHEAD: u32 = 30;

/// Longest a single move may take when a clock is given.
const MAX_BUDGET: Duration = Duration::from_secs(10);

/// Shortest a single move may take when a clock is given.
const MIN_BUDGET: Duration = Duration::from_millis(10);

/// Clock time held back for the moves that follow.
const RESERVE: Duration = Duration::from_millis(50);

/// How often a finished but withheld search rechecks its limits.
const POLL: Duration = Duration::from_millis(1);

/// When a search must give up, how deep it may go before then, and what it may answer with.
pub struct Limits {
    stop: Arc<AtomicBool>,
    deadline: Option<Instant>,
    /// Plies to search; a strategy may cap this further.
    pub depth: Option<u8>,
    /// Positions to visit at most.
    pub nodes: Option<u64>,
    /// The moves the answer must come from; empty allows every legal move.
    pub search_moves: MoveList,
    /// Whether the move must be withheld until `stop`.
    pub infinite: bool,
}

impl Limits {
    /// Turns the limits of a `go` into a deadline and a move list for a search in `game`.
    pub fn new(go: &Go, game: &Game) -> Limits {
        let (remaining, increment) = match game.position().side_to_move() {
            Colour::White => (go.white_time, go.white_increment),
            Colour::Black => (go.black_time, go.black_increment),
        };
        let deadline = match (go.movetime, remaining) {
            (Some(movetime), _) => Some(Instant::now() + movetime),
            (None, Some(remaining)) => Some(
                Instant::now() + budget(remaining, increment.unwrap_or_default(), go.moves_to_go),
            ),
            (None, None) => None,
        };
        // A mate in `moves` is at most that many moves of each side deep.
        let depth = go.depth.or(go.mate.map(|moves| moves.saturating_mul(2)));
        Limits {
            stop: Arc::new(AtomicBool::new(false)),
            deadline,
            depth,
            nodes: go.nodes,
            search_moves: playable(game, &go.search_moves),
            // A `go` that names no bound at all also runs until stopped.
            infinite: go.infinite || (deadline.is_none() && depth.is_none() && go.nodes.is_none()),
        }
    }

    /// Whether the search must yield the best move it has.
    pub fn expired(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Whether the search, having visited `nodes` positions, must yield the best move it has.
    pub fn spent(&self, nodes: u64) -> bool {
        self.nodes.is_some_and(|limit| nodes >= limit) || self.expired()
    }
}

/// The moves of `wanted`, in UCI notation, that can be played in `game`; the rest are dropped.
fn playable(game: &Game, wanted: &[String]) -> MoveList {
    let mut moves = MoveList::new();
    for text in wanted {
        match game.variant().move_from_uci(game.position(), text) {
            Ok(played) => moves.push(played),
            Err(error) => warn!("Ignoring searchmoves {text}: {error}"),
        }
    }
    moves
}

/// Time to spend on one move, given the clock and the increment of the side to move.
fn budget(remaining: Duration, increment: Duration, moves_to_go: Option<u32>) -> Duration {
    let share = moves_to_go.unwrap_or(MOVES_AHEAD).clamp(1, MOVES_AHEAD);
    let planned = (remaining / share + increment / 2).min(MAX_BUDGET);
    planned
        .min(remaining.saturating_sub(RESERVE))
        .max(MIN_BUDGET)
}

/// A search in flight.
pub struct Handle {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

impl Handle {
    /// Asks the search to report its best move now.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Asks the search to stop, and waits until it has reported.
    pub fn finish(self) {
        self.stop();
        if self.thread.join().is_err() {
            error!("Search thread panicked");
        }
    }
}

/// Starts a search that emits its `info` lines and exactly one `bestmove`.
pub fn spawn(strategy: Strategy, game: Game, limits: Limits) -> Handle {
    let stop = Arc::clone(&limits.stop);
    let thread = thread::spawn(move || {
        let best = strategy.pick(&game, &limits);
        while limits.infinite && !limits.expired() {
            thread::sleep(POLL);
        }
        uci::send(&best_move(&game, best));
    });
    Handle { stop, thread }
}

/// The `bestmove` line reporting `played` in `game`, or the null move when there is none.
fn best_move(game: &Game, played: Option<Move>) -> String {
    match played {
        Some(played) => format!("bestmove {}", game.move_to_uci(played)),
        None => "bestmove 0000".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use esca::{CastlingOutput, chess960, classic};

    use super::*;

    /// The limits of a `go` at the initial position.
    fn limits_of(go: &Go) -> Limits {
        Limits::new(go, &Game::new(classic()))
    }

    /// The limits of a `go` with `depth` plies asked for and nothing else.
    fn to_depth(depth: u8) -> Limits {
        limits_of(&Go {
            depth: Some(depth),
            ..Go::default()
        })
    }

    /// A classic game from `fen`, writing castling the way GUIs do.
    fn game_of(fen: &str) -> Game {
        let mut game = Game::from_fen(classic(), fen).expect("a legal position");
        game.set_castling_output(CastlingOutput::KingTwoSquares);
        game
    }

    #[test]
    fn spends_a_slice_of_the_clock() {
        assert_eq!(
            budget(Duration::from_secs(120), Duration::from_secs(2), None),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn spends_more_when_the_moves_ahead_are_counted() {
        assert_eq!(
            budget(Duration::from_secs(60), Duration::ZERO, Some(10)),
            Duration::from_secs(6)
        );
    }

    #[test]
    fn caps_the_time_spent_on_one_move() {
        assert_eq!(
            budget(Duration::from_secs(3600), Duration::ZERO, None),
            MAX_BUDGET
        );
    }

    #[test]
    fn stays_within_the_clock() {
        assert_eq!(
            budget(Duration::from_millis(60), Duration::from_secs(10), None),
            MIN_BUDGET
        );
    }

    #[test]
    fn runs_until_stopped_only_without_any_bound() {
        assert!(limits_of(&Go::default()).infinite);
        assert!(
            limits_of(&Go {
                infinite: true,
                ..Go::default()
            })
            .infinite
        );
        assert!(!to_depth(3).infinite);
    }

    #[test]
    fn searches_a_mate_to_a_bounded_depth() {
        let limits = limits_of(&Go {
            mate: Some(2),
            ..Go::default()
        });
        assert_eq!(limits.depth, Some(4));
        assert!(!limits.infinite);
    }

    #[test]
    fn keeps_only_the_searchmoves_that_can_be_played() {
        let game = Game::new(classic());
        let limits = Limits::new(
            &Go {
                search_moves: ["e2e4", "e2e5", "nonsense"].map(str::to_owned).into(),
                ..Go::default()
            },
            &game,
        );
        assert_eq!(limits.search_moves.len(), 1);
        assert_eq!(game.move_to_uci(limits.search_moves[0]), "e2e4");
    }

    #[test]
    fn expires_at_the_deadline() {
        let limits = limits_of(&Go {
            movetime: Some(Duration::ZERO),
            ..Go::default()
        });
        assert!(limits.expired());
    }

    #[test]
    fn expires_when_stopped() {
        let limits = to_depth(3);
        assert!(!limits.expired());
        limits.stop.store(true, Ordering::Relaxed);
        assert!(limits.expired());
    }

    #[test]
    fn expires_when_the_nodes_run_out() {
        let limits = limits_of(&Go {
            nodes: Some(100),
            ..Go::default()
        });
        assert!(!limits.spent(99));
        assert!(limits.spent(100));
    }

    #[test]
    fn reports_castling_the_way_guis_write_it() {
        let game = game_of("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        let castle = game
            .variant()
            .move_from_uci(game.position(), "e1g1")
            .expect("a legal move");
        assert_eq!(best_move(&game, Some(castle)), "bestmove e1g1");
        assert_eq!(best_move(&game, None), "bestmove 0000");
    }

    #[test]
    fn reports_castling_king_to_rook_in_chess960() {
        let game =
            Game::from_fen(chess960(), "4k3/8/8/8/8/8/8/1KR5 w C - 0 1").expect("a legal position");
        let castle = game
            .variant()
            .move_from_uci(game.position(), "b1c1")
            .expect("a legal move");
        assert_eq!(best_move(&game, Some(castle)), "bestmove b1c1");
    }
}
