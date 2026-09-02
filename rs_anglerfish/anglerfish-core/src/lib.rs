//! The Anglerfish engine.
//!
//! [`uci`] serves the protocol, [`search`] bounds and runs one `go`, and
//! [`strategy`] picks the move. What a position is worth, and which of its
//! moves deserve looking at, are asked of the [`eval::Evaluator`] and
//! [`eval::Policy`] traits; [`eval::Material`] and [`eval::Uniform`] are the
//! implementations that need no net.
//!
//! Board, moves and game state come from [`esca`].

#![deny(missing_docs)]

pub mod eval;
pub mod search;
pub mod strategy;
pub mod uci;
