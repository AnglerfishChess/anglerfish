//! Protocol behaviour of the engine binary, driven over its stdin and stdout.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// How long an expected reply may take.
const TIMEOUT: Duration = Duration::from_secs(5);

/// How long to wait before calling the engine silent.
const SILENCE: Duration = Duration::from_millis(300);

/// A forced-move position: the white king must capture the queen.
const FORCED: &str = "k7/8/8/8/8/8/6q1/7K w - - 0 1";

/// A position where white mates in one, by `h5f7`.
const MATE_IN_ONE: &str = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1";

/// A position white can castle from, in the classic geometry.
const CASTLING: &str = "4k3/8/8/8/8/8/8/4K2R w K - 0 1";

/// A position only Chess960 can play from: the rook stands beside its king, and castling
/// still puts the king on g1 and the rook on f1.
const SHUFFLED: &str = "4k3/8/8/8/8/8/8/1KR5 w C - 0 1";

/// Every legal first move of white.
const OPENINGS: [&str; 20] = [
    "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4", "f2f3", "f2f4",
    "g2g3", "g2g4", "h2h3", "h2h4", "b1a3", "b1c3", "g1f3", "g1h3",
];

/// A running engine process.
struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Engine {
    /// Starts the engine and completes the `uci` handshake, returning it and the handshake lines.
    fn handshake() -> (Engine, Vec<String>) {
        let mut engine = Engine::start();
        engine.send("uci");
        let lines = engine.until("uciok");
        (engine, lines)
    }

    /// Starts the engine.
    fn start() -> Engine {
        let mut child = Command::new(env!("CARGO_BIN_EXE_anglerfish"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("the engine binary to run");
        let stdin = child.stdin.take().expect("a pipe to the engine");
        let stdout = child.stdout.take().expect("a pipe from the engine");
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Engine {
            child,
            stdin,
            lines,
        }
    }

    /// Writes one command.
    fn send(&mut self, command: &str) {
        writeln!(self.stdin, "{command}").expect("the engine to accept a command");
    }

    /// The next line, waiting up to `patience` for it.
    fn line(&mut self, patience: Duration) -> Option<String> {
        match self.lines.recv_timeout(patience) {
            Ok(line) => Some(line),
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => None,
        }
    }

    /// The lines up to and including the first one starting with `prefix`.
    fn until(&mut self, prefix: &str) -> Vec<String> {
        self.until_within(TIMEOUT, prefix)
    }

    /// Like `until`, with its own patience for a search that is slow in a
    /// debug build on a loaded machine.
    fn until_within(&mut self, patience: Duration, prefix: &str) -> Vec<String> {
        let mut lines = Vec::new();
        loop {
            let line = self
                .line(patience)
                .unwrap_or_else(|| panic!("expected a {prefix:?} line, got {lines:#?}"));
            let found = line.starts_with(prefix);
            lines.push(line);
            if found {
                return lines;
            }
        }
    }

    /// The `bestmove` line of a search, with the keyword stripped.
    fn best_move(&mut self) -> String {
        let line = self.until("bestmove").pop().expect("a bestmove");
        line["bestmove ".len()..].to_owned()
    }

    /// Panics unless nothing more arrives.
    fn expect_silence(&mut self) {
        if let Some(line) = self.line(SILENCE) {
            panic!("expected silence, got {line:?}");
        }
    }

    /// The exit status, waiting up to `TIMEOUT` for the process to end.
    fn wait(&mut self) -> Option<bool> {
        let deadline = Instant::now() + TIMEOUT;
        while Instant::now() < deadline {
            match self.child.try_wait().expect("the engine to be waitable") {
                Some(status) => return Some(status.success()),
                None => thread::sleep(Duration::from_millis(10)),
            }
        }
        None
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn identifies_itself_and_its_options() {
    let (mut engine, lines) = Engine::handshake();

    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("id name Anglerfish "))
    );
    assert!(
        lines
            .iter()
            .any(|line| line == "id author Alexander Myodov")
    );
    assert!(lines.iter().any(
        |line| line == "option name Strategy type combo default random var random var two-ply"
    ));
    assert!(
        lines
            .iter()
            .any(|line| line == "option name UCI_Chess960 type check default false")
    );

    engine.send("isready");
    assert_eq!(engine.line(TIMEOUT).as_deref(), Some("readyok"));
}

#[test]
fn plays_from_the_position_the_gui_set() {
    let (mut engine, _) = Engine::handshake();

    engine.send(&format!("position fen {FORCED}"));
    engine.send("go movetime 50");
    assert_eq!(engine.best_move(), "h1g2");

    engine.send(&format!("position fen {FORCED} moves h1g2"));
    engine.send("go movetime 50");
    assert!(["a8a7", "a8b7", "a8b8"].contains(&engine.best_move().as_str()));
}

#[test]
fn answers_isready_while_searching_and_stops_once() {
    let (mut engine, _) = Engine::handshake();

    engine.send("position startpos");
    engine.send("go infinite");
    engine.send("isready");
    let lines = engine.until("readyok");
    assert!(
        lines[..lines.len() - 1]
            .iter()
            .all(|line| line.starts_with("info "))
    );

    engine.send("stop");
    assert!(OPENINGS.contains(&engine.best_move().as_str()));
    engine.expect_silence();
}

#[test]
fn says_nothing_about_a_search_that_never_started() {
    let (mut engine, _) = Engine::handshake();

    engine.send("stop");
    engine.send("isready");
    assert_eq!(engine.line(TIMEOUT).as_deref(), Some("readyok"));
}

#[test]
fn exits_while_searching() {
    let (mut engine, _) = Engine::handshake();

    engine.send("position startpos");
    engine.send("go infinite");
    engine.send("quit");
    assert_eq!(engine.wait(), Some(true));
}

#[test]
fn survives_input_it_cannot_use() {
    let (mut engine, _) = Engine::handshake();

    engine.send("");
    engine.send("nonsense");
    engine.send("position fen not/a/fen");
    engine.send("position startpos moves e2e5");
    engine.send(&format!("position fen {SHUFFLED}"));
    engine.send("setoption name Nonsense value 1");
    engine.send("setoption name Strategy value nonsense");
    engine.send("isready");
    assert_eq!(engine.line(TIMEOUT).as_deref(), Some("readyok"));

    engine.send("go movetime 50");
    assert!(OPENINGS.contains(&engine.best_move().as_str()));
}

#[test]
fn reports_its_thinking_when_searching() {
    let (mut engine, _) = Engine::handshake();

    engine.send("setoption name Strategy value two-ply");
    engine.send("position startpos");
    engine.send("go depth 2");
    let lines = engine.until("bestmove");

    assert!(lines.iter().any(|line| {
        line.starts_with("info depth 2 ") && line.contains(" score cp ") && line.contains(" pv ")
    }));
    engine.expect_silence();
}

#[test]
fn answers_a_search_for_a_mate() {
    let (mut engine, _) = Engine::handshake();

    engine.send("setoption name Strategy value two-ply");
    engine.send(&format!("position fen {MATE_IN_ONE}"));
    engine.send("go mate 1");
    let lines = engine.until_within(Duration::from_secs(60), "bestmove");

    assert_eq!(lines.last().map(String::as_str), Some("bestmove h5f7"));
    assert!(
        lines
            .iter()
            .any(|line| line.contains(" score mate 1 ") && line.contains(" pv h5f7")),
        "{lines:#?}"
    );
    engine.expect_silence();
}

#[test]
fn answers_with_one_of_the_searchmoves() {
    let (mut engine, _) = Engine::handshake();

    for strategy in ["random", "two-ply"] {
        engine.send(&format!("setoption name Strategy value {strategy}"));
        engine.send("position startpos");
        engine.send("go depth 2 searchmoves a2a3 h2h3");
        let bestmove = engine.best_move();
        assert!(
            ["a2a3", "h2h3"].contains(&bestmove.as_str()),
            "{strategy} answered {bestmove:?}"
        );
    }

    // Naming no usable move leaves every legal one allowed.
    engine.send("go depth 2 searchmoves e2e5 nonsense");
    assert!(OPENINGS.contains(&engine.best_move().as_str()));
}

#[test]
fn reports_a_score_even_when_picking_at_random() {
    let (mut engine, _) = Engine::handshake();

    engine.send("position startpos");
    engine.send("go movetime 50");
    let lines = engine.until("bestmove");

    assert!(lines.iter().any(|line| {
        line.starts_with("info depth ") && line.contains(" score cp ") && line.contains(" pv ")
    }));
}

#[test]
fn writes_castling_the_way_guis_do() {
    let (mut engine, _) = Engine::handshake();

    engine.send(&format!("position fen {CASTLING}"));
    engine.send("go movetime 50 searchmoves e1g1");
    assert_eq!(engine.best_move(), "e1g1");
}

#[test]
fn writes_castling_king_to_rook_under_chess960() {
    let (mut engine, _) = Engine::handshake();

    engine.send("setoption name UCI_Chess960 value true");
    engine.send(&format!("position fen {SHUFFLED}"));
    engine.send("go movetime 50 searchmoves b1c1");
    assert_eq!(engine.best_move(), "b1c1");

    // The two-square spelling names the same move, and is answered king-to-rook.
    engine.send(&format!("position fen {SHUFFLED}"));
    engine.send("go movetime 50 searchmoves b1g1");
    assert_eq!(engine.best_move(), "b1c1");
}

#[test]
fn plays_a_chess960_game_the_gui_hands_it() {
    let (mut engine, _) = Engine::handshake();

    engine.send("setoption name UCI_Chess960 value true");
    engine.send(&format!("position fen {SHUFFLED} moves b1c1"));
    engine.send("go movetime 50");
    assert!(["e8d7", "e8e7", "e8f7", "e8d8", "e8f8"].contains(&engine.best_move().as_str()));
}
