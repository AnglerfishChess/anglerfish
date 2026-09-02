//! Reading UCI command lines.

use std::time::Duration;

/// A command from the GUI.
#[derive(Debug, PartialEq)]
pub enum Command {
    /// `uci`.
    Uci,
    /// `isready`.
    IsReady,
    /// `setoption`, with the value it names when it names one.
    SetOption {
        /// The option's name, as written.
        name: String,
        /// The value, as written.
        value: Option<String>,
    },
    /// `ucinewgame`.
    NewGame,
    /// `position`.
    Position(Setup),
    /// `go`.
    Go(Go),
    /// `stop`.
    Stop,
    /// `quit`.
    Quit,
    /// A command that is understood and calls for nothing: `debug`, `ponderhit`, `register`.
    Nothing,
}

/// The position a `position` names: where to start, and what was played from there.
#[derive(Debug, Default, PartialEq)]
pub struct Setup {
    /// The FEN to start from; `None` for the start position, as `startpos` asks.
    pub fen: Option<String>,
    /// The moves played onto it, in UCI notation.
    pub moves: Vec<String>,
}

/// The limits named by a `go`.
#[derive(Debug, Default, PartialEq)]
pub struct Go {
    /// `movetime`.
    pub movetime: Option<Duration>,
    /// `wtime`.
    pub white_time: Option<Duration>,
    /// `btime`.
    pub black_time: Option<Duration>,
    /// `winc`.
    pub white_increment: Option<Duration>,
    /// `binc`.
    pub black_increment: Option<Duration>,
    /// `movestogo`.
    pub moves_to_go: Option<u32>,
    /// `depth`, in plies.
    pub depth: Option<u8>,
    /// `nodes`.
    pub nodes: Option<u64>,
    /// Moves to mate in, as `mate` asks.
    pub mate: Option<u8>,
    /// The moves the answer must come from, in UCI notation; empty allows every legal move.
    pub search_moves: Vec<String>,
    /// Whether the search runs until stopped, as `infinite` and `ponder` ask.
    pub infinite: bool,
}

/// Reads a command line, dropping unrecognised leading tokens as the UCI specification requires.
pub fn parse(line: &str) -> Option<Command> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    for (index, keyword) in tokens.iter().enumerate() {
        let rest = &tokens[index + 1..];
        let command = match *keyword {
            "uci" => Some(Command::Uci),
            "isready" => Some(Command::IsReady),
            "ucinewgame" => Some(Command::NewGame),
            "stop" => Some(Command::Stop),
            "quit" => Some(Command::Quit),
            "debug" | "ponderhit" | "register" => Some(Command::Nothing),
            "setoption" => set_option(rest),
            "position" => position(rest),
            "go" => Some(Command::Go(go(rest))),
            _ => None,
        };
        if command.is_some() {
            return command;
        }
    }
    None
}

/// Reads the tail of `setoption name <name> [value <value>]`.
fn set_option(tokens: &[&str]) -> Option<Command> {
    let (keyword, rest) = tokens.split_first()?;
    if *keyword != "name" {
        return None;
    }
    let (name, value) = match rest.iter().position(|token| *token == "value") {
        Some(at) => (&rest[..at], Some(rest[at + 1..].join(" "))),
        None => (rest, None),
    };
    if name.is_empty() {
        return None;
    }
    Some(Command::SetOption {
        name: name.join(" "),
        value,
    })
}

/// Reads the tail of `position startpos|fen <fields> [moves <move>...]`.
fn position(tokens: &[&str]) -> Option<Command> {
    let (keyword, rest) = tokens.split_first()?;
    let played = rest.iter().position(|token| *token == "moves");
    let (fields, rest) = match played {
        Some(at) => (&rest[..at], &rest[at + 1..]),
        None => (rest, &rest[rest.len()..]),
    };
    let fen = match *keyword {
        "startpos" if fields.is_empty() => None,
        "fen" if !fields.is_empty() => Some(fields.join(" ")),
        _ => return None,
    };
    Some(Command::Position(Setup {
        fen,
        moves: rest.iter().map(|token| (*token).to_owned()).collect(),
    }))
}

/// The keywords a `go` may name, each ending the argument list of the one before.
const GO_KEYWORDS: [&str; 12] = [
    "searchmoves",
    "ponder",
    "wtime",
    "btime",
    "winc",
    "binc",
    "movestogo",
    "depth",
    "nodes",
    "mate",
    "movetime",
    "infinite",
];

/// Reads the tail of a `go`, keeping the limits it names.
fn go(tokens: &[&str]) -> Go {
    let mut go = Go::default();
    for (index, keyword) in tokens.iter().enumerate() {
        let rest = &tokens[index + 1..];
        let argument = rest.first().copied().unwrap_or_default();
        match *keyword {
            "movetime" => go.movetime = millis(argument),
            "wtime" => go.white_time = millis(argument),
            "btime" => go.black_time = millis(argument),
            "winc" => go.white_increment = millis(argument),
            "binc" => go.black_increment = millis(argument),
            "movestogo" => go.moves_to_go = argument.parse().ok(),
            "depth" => go.depth = argument.parse().ok(),
            "nodes" => go.nodes = argument.parse().ok(),
            "mate" => go.mate = argument.parse().ok(),
            "searchmoves" => go.search_moves = search_moves(rest),
            "infinite" | "ponder" => go.infinite = true,
            _ => {}
        }
    }
    go
}

/// Reads the moves of a `searchmoves`, which run until the next keyword of the `go`.
fn search_moves(tokens: &[&str]) -> Vec<String> {
    tokens
        .iter()
        .take_while(|token| !GO_KEYWORDS.contains(*token))
        .map(|token| (*token).to_owned())
        .collect()
}

/// Reads a count of milliseconds.
fn millis(argument: &str) -> Option<Duration> {
    argument.parse().ok().map(Duration::from_millis)
}

#[cfg(test)]
mod tests {
    use esca::{Game, classic};
    use proptest::prelude::*;

    use super::*;
    use crate::search::Limits;

    /// The setup of a `position` command line.
    fn setup_of(line: &str) -> Option<Setup> {
        match parse(line) {
            Some(Command::Position(setup)) => Some(setup),
            _ => None,
        }
    }

    /// The limits of a `go` command line.
    fn go_of(line: &str) -> Go {
        match parse(line) {
            Some(Command::Go(go)) => go,
            other => panic!("expected a go, got {other:?}"),
        }
    }

    #[test]
    fn reads_a_plain_command() {
        assert_eq!(parse("  isready \n"), Some(Command::IsReady));
    }

    #[test]
    fn drops_unrecognised_leading_tokens() {
        assert_eq!(parse("joho debug on"), Some(Command::Nothing));
    }

    #[test]
    fn rejects_a_line_without_a_command() {
        assert_eq!(parse("what is this"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn reads_an_option_name_and_value_of_several_words() {
        assert_eq!(
            parse("setoption name Clear Hash value two words"),
            Some(Command::SetOption {
                name: "Clear Hash".to_owned(),
                value: Some("two words".to_owned()),
            })
        );
        assert_eq!(
            parse("setoption name Clear Hash"),
            Some(Command::SetOption {
                name: "Clear Hash".to_owned(),
                value: None,
            })
        );
        assert_eq!(parse("setoption value 1"), None);
    }

    #[test]
    fn reads_a_position_and_the_moves_played_onto_it() {
        assert_eq!(
            setup_of("position startpos moves e2e4 e7e5"),
            Some(Setup {
                fen: None,
                moves: ["e2e4", "e7e5"].map(str::to_owned).into(),
            })
        );
        assert_eq!(
            setup_of("position fen 4k3/8/8/8/8/8/8/4K2R w K - 0 1"),
            Some(Setup {
                fen: Some("4k3/8/8/8/8/8/8/4K2R w K - 0 1".to_owned()),
                moves: Vec::new(),
            })
        );
    }

    /// Four-field FEN, as an EPD without operations, is a position too.
    #[test]
    fn reads_a_position_without_clocks() {
        assert_eq!(
            setup_of("position fen 4k3/8/8/8/8/8/8/4K2R w K - moves e1g1")
                .and_then(|setup| setup.fen),
            Some("4k3/8/8/8/8/8/8/4K2R w K -".to_owned())
        );
    }

    #[test]
    fn rejects_a_position_command_it_cannot_read() {
        assert_eq!(setup_of("position"), None);
        assert_eq!(setup_of("position fen"), None);
        assert_eq!(setup_of("position startpos e2e4"), None);
        assert_eq!(setup_of("position elsewhere"), None);
    }

    #[test]
    fn reads_the_clock_of_a_go() {
        assert_eq!(
            go_of("go wtime 300000 btime 299000 winc 2000 binc 2000 movestogo 40"),
            Go {
                white_time: Some(Duration::from_secs(300)),
                black_time: Some(Duration::from_secs(299)),
                white_increment: Some(Duration::from_secs(2)),
                black_increment: Some(Duration::from_secs(2)),
                moves_to_go: Some(40),
                ..Go::default()
            }
        );
    }

    #[test]
    fn reads_the_other_limits_of_a_go() {
        assert_eq!(
            go_of("go movetime 500").movetime,
            Some(Duration::from_millis(500))
        );
        assert_eq!(go_of("go depth 7 nodes 900").depth, Some(7));
        assert_eq!(go_of("go depth 7 nodes 900").nodes, Some(900));
        assert_eq!(go_of("go mate 2").mate, Some(2));
        assert!(go_of("go infinite").infinite);
        assert!(go_of("go ponder").infinite);
        assert_eq!(go_of("go"), Go::default());
    }

    #[test]
    fn reads_the_moves_a_go_restricts_itself_to() {
        assert_eq!(
            go_of("go searchmoves e2e4 d2d4 depth 3"),
            Go {
                search_moves: vec!["e2e4".to_owned(), "d2d4".to_owned()],
                depth: Some(3),
                ..Go::default()
            }
        );
        assert!(go_of("go searchmoves").search_moves.is_empty());
    }

    #[test]
    fn keeps_the_limits_it_understands_out_of_a_go() {
        assert_eq!(go_of("go movetime nonsense mate nonsense"), Go::default());
    }

    proptest! {
        /// Whatever a GUI sends, reading it and turning it into limits answers rather than
        /// panics.
        #[test]
        fn limits_of_any_line_are_readable(line in r"[a-z0-9 /-]{0,64}") {
            let game = Game::new(classic());
            if let Some(Command::Go(go)) = parse(&line) {
                Limits::new(&go, &game);
            }
        }

        /// The same, over the `go` keywords and numbers that shape the limits.
        #[test]
        fn limits_of_any_go_are_readable(
            words in proptest::collection::vec(
                prop_oneof![
                    proptest::sample::select(GO_KEYWORDS.as_slice()).prop_map(str::to_owned),
                    any::<i64>().prop_map(|number| number.to_string()),
                    "[a-h][1-8][a-h][1-8]q?",
                ],
                0..12,
            ),
        ) {
            let game = Game::new(classic());
            let line = format!("go {}", words.join(" "));
            let Some(Command::Go(go)) = parse(&line) else {
                unreachable!("a go is a go");
            };
            Limits::new(&go, &game);
        }
    }
}
