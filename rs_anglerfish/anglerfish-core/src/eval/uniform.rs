//! The uniform policy.

use esca::{Facts, Move, Position};

use super::Policy;

/// Spreads its weight evenly over the moves it is given.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Uniform;

impl Policy for Uniform {
    fn priors(&self, _position: &Position, _facts: &Facts, moves: &[Move], out: &mut [f32]) {
        assert!(out.len() >= moves.len(), "an output slot per move");
        let prior = if moves.is_empty() {
            0.0
        } else {
            1.0 / moves.len() as f32
        };
        out[..moves.len()].fill(prior);
    }
}

#[cfg(test)]
mod tests {
    use esca::{Game, classic};

    use super::*;

    #[test]
    fn spreads_its_weight_over_every_move() {
        let game = Game::new(classic());
        let facts = game.facts();
        let moves = game.legal_moves();
        let mut priors = [0.0f32; 20];
        Uniform.priors(game.position(), &facts, &moves, &mut priors);

        assert_eq!(moves.len(), 20);
        assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(priors.iter().all(|&prior| prior == priors[0]));
    }

    #[test]
    fn leaves_no_weight_when_there_is_no_move() {
        let game = Game::from_fen(classic(), "7k/5KQ1/8/8/8/8/8/8 b - - 0 1").expect("a position");
        let facts = game.facts();
        let mut priors = [1.0f32; 1];
        Uniform.priors(game.position(), &facts, &game.legal_moves(), &mut priors);
        assert_eq!(priors, [1.0]);
    }
}
