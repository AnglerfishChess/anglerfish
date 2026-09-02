//! Anglerfish, a UCI chess engine.

fn main() {
    env_logger::init();
    anglerfish_core::uci::run();
}
