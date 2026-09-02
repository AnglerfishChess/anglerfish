//! The UCI protocol loop.
//!
//! This loop never blocks on a search: `go` hands the position to a search thread and comes
//! straight back to stdin, so `isready`, `stop` and `quit` stay answerable while thinking.

mod parse;

use std::io::{self, BufRead};
use std::ops::ControlFlow;
use std::sync::Arc;

use esca::{CLASSIC, CastlingOutput, Game, Variant, chess960, classic};
use log::{debug, warn};

use parse::{Command, parse};
pub use parse::{Go, Setup};

use crate::search::{self, Limits};
use crate::strategy::Strategy;

/// The `id name` of this engine.
const NAME: &str = concat!("Anglerfish ", env!("CARGO_PKG_VERSION"));

/// The `id author` of this engine.
const AUTHOR: &str = "Alexander Myodov";

/// The name of the UCI option selecting Chess960.
const CHESS960_OPTION: &str = "UCI_Chess960";

/// Writes one GUI-bound message as a single line on stdout.
pub fn send(message: &str) {
    println!("{message}");
}

/// Serves the UCI protocol on stdin/stdout until `quit` or end of input.
pub fn run() {
    let mut session = Session::new();
    for line in io::stdin().lock().lines().map_while(Result::ok) {
        match parse(&line) {
            Some(command) => {
                if session.handle(command).is_break() {
                    break;
                }
            }
            None => debug!("Unrecognised command: {line:?}"),
        }
    }
    session.abandon_search();
}

/// The state a UCI session carries between commands.
struct Session {
    variant: Arc<dyn Variant>,
    game: Game,
    strategy: Strategy,
    search: Option<search::Handle>,
}

impl Session {
    /// A session at the initial position of classic chess, with no search in flight.
    fn new() -> Session {
        let variant = classic();
        Session {
            game: start(&variant),
            variant,
            strategy: Strategy::default(),
            search: None,
        }
    }

    /// Acts on one command, breaking once the session is over.
    fn handle(&mut self, command: Command) -> ControlFlow<()> {
        match command {
            Command::Uci => {
                send(&format!("id name {NAME}"));
                send(&format!("id author {AUTHOR}"));
                send(&Strategy::option());
                send(&format!(
                    "option name {CHESS960_OPTION} type check default false"
                ));
                send("uciok");
            }
            Command::IsReady => send("readyok"),
            Command::SetOption { name, value } => self.set_option(&name, value.as_deref()),
            Command::NewGame => {
                self.finish_search();
                self.game = start(&self.variant);
            }
            Command::Position(setup) => self.set_position(&setup),
            Command::Go(go) => self.go(&go),
            Command::Stop => {
                if let Some(handle) = &self.search {
                    handle.stop();
                }
            }
            Command::Quit => {
                self.abandon_search();
                return ControlFlow::Break(());
            }
            Command::Nothing => {}
        }
        ControlFlow::Continue(())
    }

    /// Applies a `setoption`; unknown names and values are dropped.
    fn set_option(&mut self, name: &str, value: Option<&str>) {
        if name.eq_ignore_ascii_case(Strategy::OPTION) {
            match value.and_then(Strategy::from_name) {
                Some(strategy) => self.strategy = strategy,
                None => debug!("Ignoring unknown {} value {value:?}", Strategy::OPTION),
            }
        } else if name.eq_ignore_ascii_case(CHESS960_OPTION) {
            match value.and_then(|value| value.parse::<bool>().ok()) {
                Some(wanted) => self.play(if wanted { chess960() } else { classic() }),
                None => debug!("Ignoring unknown {CHESS960_OPTION} value {value:?}"),
            }
        } else {
            debug!("Ignoring unknown option {name:?}");
        }
    }

    /// Plays under `variant` from now on, from its standard array.
    fn play(&mut self, variant: Arc<dyn Variant>) {
        self.game = start(&variant);
        self.variant = variant;
    }

    /// Takes the position of a `position`; one the variant cannot play from is dropped whole.
    fn set_position(&mut self, setup: &Setup) {
        let mut game = match &setup.fen {
            None => start(&self.variant),
            Some(fen) => match Game::from_fen(Arc::clone(&self.variant), fen) {
                Ok(game) => game,
                Err(error) => {
                    warn!("Ignoring position {fen:?}: {error}");
                    return;
                }
            },
        };
        game.set_castling_output(castling_output(self.variant.as_ref()));
        for text in &setup.moves {
            if let Err(error) = game.play_uci(text) {
                warn!("Ignoring position: {text} is unplayable ({error})");
                return;
            }
        }
        self.game = game;
    }

    /// Starts a search for the current position, after any search in flight has reported.
    fn go(&mut self, go: &Go) {
        self.finish_search();
        let limits = Limits::new(go, &self.game);
        self.search = Some(search::spawn(self.strategy, self.game.clone(), limits));
    }

    /// Stops the search in flight and waits for its `bestmove`.
    fn finish_search(&mut self) {
        if let Some(handle) = self.search.take() {
            handle.finish();
        }
    }

    /// Stops the search in flight without waiting for it.
    fn abandon_search(&mut self) {
        if let Some(handle) = self.search.take() {
            handle.stop();
        }
    }
}

/// A game at the standard array, played under `variant` and written the way its GUIs write it.
fn start(variant: &Arc<dyn Variant>) -> Game {
    let mut game = Game::from_position(Arc::clone(variant), CLASSIC.start_position(0))
        .expect("the standard array is playable");
    game.set_castling_output(castling_output(variant.as_ref()));
    game
}

/// How `variant`'s GUIs write castling: two squares in classic chess, king to rook elsewhere.
fn castling_output(variant: &dyn Variant) -> CastlingOutput {
    if variant.name() == CLASSIC.name() {
        CastlingOutput::KingTwoSquares
    } else {
        CastlingOutput::KingToRook
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard array, as `position startpos` sets it.
    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /// A position only Chess960 can play from: the rook stands beside its king.
    const SHUFFLED: &str = "4k3/8/8/8/8/8/8/1KR5 w C - 0 1";

    /// A session told whether to play Chess960.
    fn session_of(chess960: bool) -> Session {
        let mut session = Session::new();
        session.set_option(CHESS960_OPTION, Some(&chess960.to_string()));
        session
    }

    #[test]
    fn selects_a_strategy_by_option() {
        let mut session = Session::new();
        session.set_option("Strategy", Some("two-ply"));
        assert_eq!(session.strategy, Strategy::TwoPly);

        session.set_option("Strategy", Some("nonsense"));
        assert_eq!(session.strategy, Strategy::TwoPly);

        session.set_option("Nonsense", Some("random"));
        assert_eq!(session.strategy, Strategy::TwoPly);
    }

    #[test]
    fn selects_a_variant_by_option() {
        assert_eq!(session_of(true).variant.name(), "chess960");
        assert_eq!(session_of(false).variant.name(), "chess");

        let mut session = session_of(true);
        session.set_option(CHESS960_OPTION, Some("nonsense"));
        assert_eq!(session.variant.name(), "chess960");
    }

    #[test]
    fn takes_the_position_from_the_gui() {
        let mut session = Session::new();
        let Some(command) = parse("position startpos moves d2d4") else {
            panic!("expected a position");
        };
        session.handle(command);
        assert_eq!(session.game.ply(), 1);

        session.handle(Command::NewGame);
        assert_eq!(session.game.ply(), 0);
        assert_eq!(session.game.position().fen(), START);
    }

    #[test]
    fn keeps_the_position_it_cannot_play_from() {
        let mut session = Session::new();
        for line in [
            "position fen not/a/fen",
            "position fen 8/8/8/8/8/8/8/8 w - - 0 1",
            "position startpos moves e2e5",
            &format!("position fen {SHUFFLED}"),
        ] {
            let Some(command) = parse(line) else {
                panic!("expected a position from {line:?}");
            };
            session.handle(command);
            assert_eq!(session.game.position().fen(), START, "{line}");
        }
    }

    #[test]
    fn plays_a_shuffled_back_rank_under_chess960() {
        let mut session = session_of(true);
        session.set_position(&Setup {
            fen: Some(SHUFFLED.to_owned()),
            moves: vec!["b1c1".to_owned()],
        });
        assert_eq!(session.game.moves().len(), 1);
        assert!(session.game.moves()[0].is_castling());
    }
}
