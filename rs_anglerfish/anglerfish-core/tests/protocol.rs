//! Protocol behaviour of the engine binary, driven over its stdin and stdout.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// How long an expected reply may take: a bound on the machine, not on the
/// engine, since a debug build searching on a loaded runner takes as long as it
/// takes. An engine that dies is caught by its closed output, not by this.
const TIMEOUT: Duration = Duration::from_secs(60);

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

/// What came out of the engine while waiting for a line.
#[derive(Debug)]
enum Reply {
    /// One line of output.
    Line(String),
    /// Nothing, for the whole wait.
    Silence,
    /// The engine closed its output, having exited or being about to.
    Ended,
}

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

    /// Whatever the engine says next, waiting up to `patience` for it.
    fn reply(&mut self, patience: Duration) -> Reply {
        match self.lines.recv_timeout(patience) {
            Ok(line) => Reply::Line(line),
            Err(RecvTimeoutError::Timeout) => Reply::Silence,
            Err(RecvTimeoutError::Disconnected) => Reply::Ended,
        }
    }

    /// The next line. Panics if the engine stays silent for `TIMEOUT` or ends.
    fn line(&mut self) -> String {
        match self.reply(TIMEOUT) {
            Reply::Line(line) => line,
            Reply::Silence => panic!("expected a line within {TIMEOUT:?}"),
            Reply::Ended => panic!("expected a line; the engine {}", self.ending()),
        }
    }

    /// The lines up to and including the first one starting with `prefix`.
    /// Panics if the engine stays silent for `TIMEOUT` or ends before it.
    fn until(&mut self, prefix: &str) -> Vec<String> {
        let mut lines = Vec::new();
        loop {
            match self.reply(TIMEOUT) {
                Reply::Line(line) => {
                    let found = line.starts_with(prefix);
                    lines.push(line);
                    if found {
                        return lines;
                    }
                }
                Reply::Silence => {
                    panic!("expected a {prefix:?} line within {TIMEOUT:?}, got {lines:#?}")
                }
                Reply::Ended => {
                    let ending = self.ending();
                    panic!("expected a {prefix:?} line; the engine {ending}, after {lines:#?}");
                }
            }
        }
    }

    /// The `bestmove` line of a search, with the keyword stripped.
    fn best_move(&mut self) -> String {
        let line = self.until("bestmove").pop().expect("a bestmove");
        line["bestmove ".len()..].to_owned()
    }

    /// Panics unless the engine is still running and says nothing more.
    fn expect_silence(&mut self) {
        match self.reply(SILENCE) {
            Reply::Silence => {}
            Reply::Line(line) => panic!("expected silence, got {line:?}"),
            Reply::Ended => panic!("expected silence; the engine {}", self.ending()),
        }
    }

    /// Whether the engine exited well, waiting up to `TIMEOUT` for it to end.
    fn wait(&mut self) -> Option<bool> {
        self.exit(TIMEOUT).map(|status| status.success())
    }

    /// The exit status, waiting up to `patience` for the process to end.
    fn exit(&mut self, patience: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + patience;
        loop {
            if let Some(status) = self.child.try_wait().expect("the engine to be waitable") {
                return Some(status);
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Why the engine's output ended, worded for a panic message.
    fn ending(&mut self) -> String {
        match self.exit(SILENCE) {
            Some(status) => format!("ended with {status}"),
            None => "closed its output while still running".to_owned(),
        }
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
    assert_eq!(engine.line(), "readyok");
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
    assert_eq!(engine.line(), "readyok");
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
    assert_eq!(engine.line(), "readyok");

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
    let lines = engine.until("bestmove");

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
