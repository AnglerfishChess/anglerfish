//! The material evaluator.

use esca::{Facts, Position, Role, Score, Side};

use super::Evaluator;

/// The roles `MaterialFacts::count` counts, in its order.
const COUNTED: [Role; 5] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
];

/// Counts material and nothing else.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Material;

impl Material {
    /// The centipawn value of a role: 100, 320, 330, 500, 900, and nothing
    /// for a king.
    pub fn value(role: Role) -> i32 {
        match role {
            Role::Pawn => 100,
            Role::Knight => 320,
            Role::Bishop => 330,
            Role::Rook => 500,
            Role::Queen => 900,
            Role::King => 0,
        }
    }

    /// The material balance in centipawns, from the side to move's point of
    /// view.
    pub fn balance(facts: &Facts) -> i32 {
        let counts = &facts.material.count;
        COUNTED
            .into_iter()
            .map(|role| {
                let ours = i32::from(counts[Side::Us.index()][role.index()]);
                let theirs = i32::from(counts[Side::Them.index()][role.index()]);
                Material::value(role) * (ours - theirs)
            })
            .sum()
    }
}

impl Evaluator for Material {
    fn value(&self, _position: &Position, facts: &Facts) -> Score {
        Score::Cp(Material::balance(facts))
    }
}

#[cfg(test)]
mod tests {
    use esca::{Position, classic};

    use super::*;

    /// The material evaluation of the position `fen`.
    fn value_of(fen: &str) -> Score {
        let position = Position::from_fen(fen).expect("a legal position");
        let facts = position.facts(classic().as_ref());
        Material.value(&position, &facts)
    }

    #[test]
    fn counts_material_for_the_side_to_move() {
        assert_eq!(value_of("4k3/8/8/8/8/8/8/3QK3 b - - 0 1"), Score::Cp(-900));
        assert_eq!(value_of("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"), Score::Cp(900));
    }

    #[test]
    fn calls_the_initial_position_level() {
        assert_eq!(
            value_of("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Score::Cp(0)
        );
    }

    #[test]
    fn scores_a_batch_as_it_scores_one() {
        let position =
            Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").expect("a legal position");
        let facts = position.facts(classic().as_ref());
        let items = [(position.clone(), facts.clone()), (position, facts)];
        let mut out = [Score::Cp(0); 2];
        Material.batch(&items, &mut out);
        assert_eq!(out, [Score::Cp(900); 2]);
    }
}
