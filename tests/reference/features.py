"""The v1 feature vector, written out from `docs/features.md`.

One function per group, in schema order, each appending to a list of values.
Everything is a plain loop over squares and over the legal move list; the point
is to be readable and independent, not fast.
"""

from __future__ import annotations

import struct
from pathlib import Path

from . import board as b
from .board import (
    ORDER,
    TARGET,
    VALUE,
    Move,
    Position,
    attacks_of,
    between,
    chebyshev,
    file_of,
    line_through,
    occupancy,
    other,
    rank_of,
    relative_rank,
    relative_square,
    role_at,
    square_at,
    view,
)

CANONICAL = Path(__file__).resolve().parents[2] / "rs_anglerfish" / "esca" / "tests" / "data" / "schema_v1.txt"

#: The features whose definitions assume the classic starting squares.
CLASSIC_ONLY = {
    ("pieces", "minors_undeveloped"),
    ("pieces", "queen_developed"),
    ("king", "king_on_home_square"),
    ("king", "king_castled_zone"),
}

MINOR_ROLES = ("n", "b", "r", "q")
COUNTED_ROLES = ("p", "n", "b", "r", "q")


def read_schema() -> list[tuple[str, int, list[tuple[str, int, int]]]]:
    """The canonical text as groups of ``(name, width, [(feature, offset, width)])``."""
    groups: list[tuple[str, int, list[tuple[str, int, int]]]] = []
    for line in CANONICAL.read_text().splitlines():
        if line.startswith("  "):
            name, width, _encoding = line.strip().split(":", 2)
            features = groups[-1][2]
            offset = features[-1][1] + features[-1][2] if features else 0
            features.append((name, offset, int(width)))
        else:
            name, _version, width = line.split(":")
            groups.append((name, int(width), []))
    return groups


SCHEMA = read_schema()


def f32(value: float) -> float:
    """``value`` rounded to the nearest `f32`, as the Rust side stores it."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


class Writer:
    """A cursor that appends one group's values."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def value(self, value: float) -> None:
        self.values.append(f32(value))

    def bit(self, flag: bool) -> None:
        self.value(1.0 if flag else 0.0)

    def count(self, n: float, scale: float) -> None:
        self.value(min(max(n, 0.0), scale) / scale)

    def diff(self, d: float, scale: float) -> None:
        self.value(max(-1.0, min(1.0, d / scale)))

    def one_hot(self, index: int | None, width: int) -> None:
        for slot in range(width):
            self.bit(index == slot)

    def mask8(self, files: set[int]) -> None:
        for file in range(8):
            self.bit(file in files)

    def plane(self, squares: set[int], us: str) -> None:
        for square in range(64):
            self.bit(view(square, us) in squares)


class Scan:
    """The shared pass: placement and attacks, indexed by side (0 = us)."""

    def __init__(self, position: Position) -> None:
        self.position = position
        self.us = position.side_to_move
        self.colour = (self.us, other(self.us))
        self.occupied = occupancy(position)
        self.units = tuple(set(position.squares_of(colour)) for colour in self.colour)
        self.role_units = tuple(
            {role: set(position.squares_of(colour, role)) for role in b.ROLES} for colour in self.colour
        )
        self.kings = tuple(position.king_of(colour) for colour in self.colour)
        self.attacks_from: dict[int, set[int]] = {}
        self.by_role: tuple[dict[str, set[int]], dict[str, set[int]]] = ({}, {})
        self.by: list[set[int]] = [set(), set()]
        for side in (0, 1):
            for role in b.ROLES:
                union: set[int] = set()
                for square in self.role_units[side][role]:
                    reach = set(attacks_of(role, square, self.colour[side], self.occupied))
                    self.attacks_from[square] = reach
                    union |= reach
                self.by_role[side][role] = union
                self.by[side] |= union

    def relative_rank(self, square: int, side: int) -> int:
        return relative_rank(square, self.colour[side])

    def relative_square(self, file: int, rank: int, side: int) -> int:
        return relative_square(file, rank, self.colour[side])

    def own_half(self, side: int) -> set[int]:
        return {square for square in range(64) if self.relative_rank(square, side) <= 4}

    def light(self) -> set[int]:
        """The squares that are light in the mover's view."""
        return {square for square in range(64) if not b.is_dark(view(square, self.us))}

    def dark(self) -> set[int]:
        """The squares that are dark in the mover's view."""
        return {square for square in range(64) if b.is_dark(view(square, self.us))}


def files_of(squares: set[int]) -> set[int]:
    return {file_of(square) for square in squares}


def adjacent_files(file: int) -> list[int]:
    return [neighbour for neighbour in (file - 1, file + 1) if 0 <= neighbour <= 7]


# --------------------------------------------------------------------------
# placement


def placement(scan: Scan, w: Writer) -> None:
    for side in (0, 1):
        for role in b.ROLES:
            w.plane(scan.role_units[side][role], scan.us)


# --------------------------------------------------------------------------
# state


def state(scan: Scan, w: Writer) -> None:
    position = scan.position
    w.bit(b.in_check(position))
    w.bit(len(b.checkers(position)) >= 2)
    for side in (0, 1):
        short, long = position.rights(scan.colour[side])
        w.bit(short is not None)
        w.bit(long is not None)
    ep = position.ep_square
    w.bit(ep is not None)
    w.one_hot(None if ep is None else file_of(ep), 8)
    w.bit(any(move.en_passant for move in b.legal_moves(position)))


def history(scan: Scan, w: Writer) -> None:
    position = scan.position
    w.one_hot(halfmove_bucket(position.halfmove_clock), 8)
    w.bit(position.clocks_known)
    # A position carries no history: repetition, the recent plies and the last
    # move are all unknown.
    w.bit(False)
    w.bit(False)
    w.count(0, 8.0)
    w.count(0, 8.0)
    w.count(0, 16.0)
    w.diff(0, 9.0)
    w.one_hot(None, 5)
    w.one_hot(None, 6)
    w.bit(False)


def halfmove_bucket(clock: int) -> int:
    for index, top in enumerate((0, 3, 9, 19, 39, 69, 89)):
        if clock <= top:
            return index
    return 7


# --------------------------------------------------------------------------
# material


def material_value(scan: Scan, side: int) -> int:
    return sum(VALUE[role] * len(scan.role_units[side][role]) for role in COUNTED_ROLES)


def phase_points(scan: Scan) -> int:
    weights = {"q": 4, "r": 2, "b": 1, "n": 1}
    return sum(weight * len(scan.role_units[side][role]) for side in (0, 1) for role, weight in weights.items())


def insufficient(scan: Scan, side: int) -> bool:
    units = scan.role_units[side]
    if units["p"] or units["r"] or units["q"]:
        return False
    knights = units["n"]
    bishops = units["b"]
    if len(knights) + len(bishops) <= 1:
        return True
    return not knights and (
        all(b.is_dark(square) for square in bishops) or all(not b.is_dark(square) for square in bishops)
    )


def material(scan: Scan, w: Writer) -> None:
    counts = [[len(scan.role_units[side][role]) for role in COUNTED_ROLES] for side in (0, 1)]
    for side in (0, 1):
        for index, role in enumerate(COUNTED_ROLES):
            w.count(counts[side][index], 8.0 if role == "p" else 4.0)
    for index in range(5):
        w.diff(counts[0][index] - counts[1][index], 4.0)
    for side in (0, 1):
        w.count(
            sum(VALUE[role] * len(scan.role_units[side][role]) for role in MINOR_ROLES),
            62.0,
        )
    w.diff(material_value(scan, 0) - material_value(scan, 1), 20.0)
    phase = f32(min(phase_points(scan), 24) / 24.0)
    w.value(phase)
    w.one_hot(0 if phase > 0.75 else (1 if phase >= 0.25 else 2), 3)
    w.bit(bool(scan.role_units[0]["q"]) and bool(scan.role_units[1]["q"]))
    w.bit(all(not scan.role_units[side][role] for side in (0, 1) for role in MINOR_ROLES))
    for side in (0, 1):
        w.bit(insufficient(scan, side))


# --------------------------------------------------------------------------
# pawns


class PawnFacts:
    """The pawn structure, worked out once and read by three groups."""

    def __init__(self, scan: Scan) -> None:
        self.scan = scan
        self.pawns = [set(scan.role_units[side]["p"]) for side in (0, 1)]
        self.passed: list[set[int]] = [set(), set()]
        self.candidates: list[set[int]] = [set(), set()]
        self.doubled: list[set[int]] = [set(), set()]
        self.isolated: list[set[int]] = [set(), set()]
        self.backward: list[set[int]] = [set(), set()]
        self.defended: list[set[int]] = [set(), set()]
        self.levers = [0, 0]
        self.islands = [0, 0]
        self.count_by_file = [[0] * 8 for _ in (0, 1)]
        self.count_by_rank = [[0] * 8 for _ in (0, 1)]
        self.open_files: set[int] = set()
        self.semi_open: list[set[int]] = [set(), set()]
        self.lead_rank: list[int | None] = [None, None]
        self.protected = [0, 0]
        self.connected = [False, False]
        self.unstoppable = [False, False]

        for side in (0, 1):
            ours = self.pawns[side]
            theirs = self.pawns[1 - side]
            for square in ours:
                self.count_by_file[side][file_of(square)] += 1
                self.count_by_rank[side][scan.relative_rank(square, side) - 1] += 1

            for square in ours:
                file = file_of(square)
                rank = scan.relative_rank(square, side)
                neighbours = adjacent_files(file)

                if any(file_of(mate) == file for mate in ours if mate != square):
                    self.doubled[side].add(square)
                if not any(file_of(mate) in neighbours for mate in ours):
                    self.isolated[side].add(square)

                blockers = [
                    enemy
                    for enemy in theirs
                    if file_of(enemy) in [file, *neighbours] and scan.relative_rank(enemy, side) > rank
                ]
                if not blockers:
                    self.passed[side].add(square)
                elif not any(file_of(enemy) == file and scan.relative_rank(enemy, side) > rank for enemy in theirs):
                    support = sum(
                        1 for mate in ours if file_of(mate) in neighbours and scan.relative_rank(mate, side) <= rank
                    )
                    opposition = sum(
                        1 for enemy in theirs if file_of(enemy) in neighbours and scan.relative_rank(enemy, side) > rank
                    )
                    if support >= opposition:
                        self.candidates[side].add(square)

                stop = scan.relative_square(file, rank + 1, side)
                if (
                    square not in self.passed[side]
                    and not any(file_of(mate) in neighbours and scan.relative_rank(mate, side) <= rank for mate in ours)
                    and stop in scan.by_role[1 - side]["p"]
                ):
                    self.backward[side].add(square)

                if any(target in theirs for target in b.pawn_attacks(square, scan.colour[side])):
                    self.levers[side] += 1

            self.defended[side] = ours & scan.by_role[side]["p"]

            previous = False
            for file in range(8):
                present = self.count_by_file[side][file] > 0
                if present and not previous:
                    self.islands[side] += 1
                previous = present

            passers = self.passed[side]
            if passers:
                self.lead_rank[side] = max(scan.relative_rank(square, side) for square in passers)
            self.protected[side] = len(passers & self.defended[side])
            self.connected[side] = any(
                any(file_of(mate) in adjacent_files(file_of(square)) for mate in passers) for square in passers
            )
            self.unstoppable[side] = any(self.is_unstoppable(square, side) for square in passers)

        for file in range(8):
            ours = self.count_by_file[0][file] > 0
            theirs = self.count_by_file[1][file] > 0
            if not ours and not theirs:
                self.open_files.add(file)
            if not ours and theirs:
                self.semi_open[0].add(file)
            if not theirs and ours:
                self.semi_open[1].add(file)

        self.rams = sum(
            1
            for square in self.pawns[0]
            if scan.relative_square(file_of(square), scan.relative_rank(square, 0) + 1, 0) in self.pawns[1]
        )

    def is_unstoppable(self, square: int, side: int) -> bool:
        scan = self.scan
        defender = 1 - side
        if any(scan.role_units[defender][role] for role in MINOR_ROLES):
            return False
        rank = scan.relative_rank(square, side)
        promotion = scan.relative_square(file_of(square), 8, side)
        tempo = 1 if defender == 0 else 0
        return max(chebyshev(scan.kings[defender], promotion) - tempo, 0) > 8 - rank


def pawns(scan: Scan, facts: PawnFacts, w: Writer) -> None:
    for side in (0, 1):
        for file in range(8):
            w.count(facts.count_by_file[side][file], 3.0)
    for side in (0, 1):
        for rank in range(8):
            w.count(facts.count_by_rank[side][rank], 8.0)
    for group in (
        facts.doubled,
        facts.isolated,
        facts.backward,
        facts.passed,
        facts.candidates,
    ):
        for side in (0, 1):
            w.mask8(files_of(group[side]))
    for side in (0, 1):
        lead = facts.lead_rank[side]
        w.one_hot(None if lead is None else lead - 1, 8)
    for side in (0, 1):
        w.count(facts.protected[side], 4.0)
    for side in (0, 1):
        w.bit(facts.connected[side])
    for side in (0, 1):
        w.bit(facts.unstoppable[side])
    w.mask8(facts.open_files)
    w.mask8(facts.semi_open[0])
    w.mask8(facts.semi_open[1])
    for side in (0, 1):
        w.count(facts.islands[side], 4.0)
    for side in (0, 1):
        w.count(len(facts.defended[side]), 8.0)
    for side in (0, 1):
        w.count(facts.levers[side], 4.0)
    w.count(facts.rams, 8.0)


# --------------------------------------------------------------------------
# pieces


def outposts(scan: Scan, facts: PawnFacts, side: int) -> set[int]:
    theirs = facts.pawns[1 - side]
    out = set()
    for square in scan.by_role[side]["p"]:
        rank = scan.relative_rank(square, side)
        if not 4 <= rank <= 6:
            continue
        attackable = any(
            file_of(enemy) in adjacent_files(file_of(square)) and scan.relative_rank(enemy, side) >= rank
            for enemy in theirs
        )
        if not attackable:
            out.add(square)
    return out


def behind_a_passer(scan: Scan, rook: int, passers: set[int], owner: int) -> bool:
    return any(
        file_of(passer) == file_of(rook) and scan.relative_rank(rook, owner) < scan.relative_rank(passer, owner)
        for passer in passers
    )


def trapped_rook(scan: Scan, side: int) -> bool:
    short, long = scan.position.rights(scan.colour[side])
    if short is not None or long is not None:
        return False
    king = scan.kings[side]
    for rook in scan.role_units[side]["r"]:
        outside = file_of(rook) > file_of(king) if file_of(king) >= 4 else file_of(rook) < file_of(king)
        quiet = len(scan.attacks_from[rook] - scan.occupied)
        if outside and quiet <= 2:
            return True
    return False


def pieces(scan: Scan, facts: PawnFacts, w: Writer) -> None:
    light = scan.light()
    dark = scan.dark()
    bishops_light = [len(scan.role_units[side]["b"] & light) for side in (0, 1)]
    bishops_dark = [len(scan.role_units[side]["b"] & dark) for side in (0, 1)]

    for side in (0, 1):
        w.bit(bishops_light[side] > 0 and bishops_dark[side] > 0)
    for side in (0, 1):
        w.count(bishops_light[side], 2.0)
        w.count(bishops_dark[side], 2.0)
    w.bit(
        bishops_light[0] + bishops_dark[0] == 1
        and bishops_light[1] + bishops_dark[1] == 1
        and bishops_light[0] != bishops_light[1]
    )
    for side in (0, 1):
        colours: set[int] = set()
        if bishops_light[side]:
            colours |= light
        if bishops_dark[side]:
            colours |= dark
        w.count(len(facts.pawns[side] & colours), 8.0)

    connected_rank = [False, False]
    connected_file = [False, False]
    for side in (0, 1):
        rooks = sorted(scan.role_units[side]["r"])
        for index, a in enumerate(rooks):
            for c in rooks[index + 1 :]:
                if set(between(a, c)) & scan.occupied:
                    continue
                if rank_of(a) == rank_of(c):
                    connected_rank[side] = True
                if file_of(a) == file_of(c):
                    connected_file[side] = True
    for side in (0, 1):
        w.bit(connected_rank[side])
    for side in (0, 1):
        w.bit(connected_file[side])

    counters = {name: [0, 0] for name in ("open", "semi", "seventh", "own", "enemy")}
    for side in (0, 1):
        for rook in scan.role_units[side]["r"]:
            if file_of(rook) in facts.open_files:
                counters["open"][side] += 1
            if file_of(rook) in facts.semi_open[side]:
                counters["semi"][side] += 1
            if scan.relative_rank(rook, side) == 7:
                counters["seventh"][side] += 1
            if behind_a_passer(scan, rook, facts.passed[side], side):
                counters["own"][side] += 1
            if behind_a_passer(scan, rook, facts.passed[1 - side], 1 - side):
                counters["enemy"][side] += 1
    for name in ("open", "semi", "seventh", "own", "enemy"):
        for side in (0, 1):
            w.count(counters[name][side], 2.0)

    for side in (0, 1):
        w.bit(trapped_rook(scan, side))

    squares = [outposts(scan, facts, side) for side in (0, 1)]
    for side in (0, 1):
        minors = scan.role_units[side]["n"] | scan.role_units[side]["b"]
        w.count(len(minors & squares[side]), 2.0)
    for side in (0, 1):
        w.count(len(squares[side] - scan.occupied), 4.0)
    for side in (0, 1):
        w.count(
            sum(
                1
                for knight in scan.role_units[side]["n"]
                if file_of(knight) in (0, 7) or scan.relative_rank(knight, side) in (1, 8)
            ),
            2.0,
        )
    for side in (0, 1):
        home = {scan.relative_square(file, 1, side) for file in (1, 2, 5, 6)}
        minors = scan.role_units[side]["n"] | scan.role_units[side]["b"]
        w.count(len(minors & home), 4.0)
    for side in (0, 1):
        queen_home = scan.relative_square(3, 1, side)
        w.bit(bool(scan.role_units[side]["q"] - {queen_home}))


# --------------------------------------------------------------------------
# king


def shield_files(king_file: int) -> list[int]:
    centre = min(max(king_file, 1), 6)
    return [centre - 1, centre, centre + 1]


def nearest_ahead(scan: Scan, pawns_on_file: set[int], rank: int, side: int) -> int | None:
    ranks = [scan.relative_rank(square, side) for square in pawns_on_file if scan.relative_rank(square, side) > rank]
    return min(ranks) - rank if ranks else None


def shelter_bucket(distance: int | None) -> int:
    if distance is None:
        return 3
    return {1: 0, 2: 1}.get(distance, 2)


def storm_bucket(distance: int | None) -> int:
    if distance is None or distance >= 5:
        return 3
    if distance <= 2:
        return 0
    return distance - 2


def king(scan: Scan, w: Writer) -> None:
    kings = scan.kings
    files = [shield_files(file_of(kings[side])) for side in (0, 1)]

    for side in (0, 1):
        w.one_hot(file_of(kings[side]), 8)
    for side in (0, 1):
        w.one_hot(scan.relative_rank(kings[side], side) - 1, 8)
    for side in (0, 1):
        w.bit(file_of(kings[side]) == 4 and scan.relative_rank(kings[side], side) == 1)
    for side in (0, 1):
        w.bit(file_of(kings[side]) <= 2)
        w.bit(file_of(kings[side]) >= 5)

    shields = []
    storms = []
    for side in (0, 1):
        rank = scan.relative_rank(kings[side], side)
        ours = scan.role_units[side]["p"]
        theirs = scan.role_units[1 - side]["p"]
        shields.append(
            [nearest_ahead(scan, {p for p in ours if file_of(p) == file}, rank, side) for file in files[side]]
        )
        storms.append(
            [nearest_ahead(scan, {p for p in theirs if file_of(p) == file}, rank, side) for file in files[side]]
        )

    for side in (0, 1):
        for slot in range(3):
            w.one_hot(shelter_bucket(shields[side][slot]), 4)
    for side in (0, 1):
        ours = scan.role_units[side]["p"]
        theirs = scan.role_units[1 - side]["p"]
        for file in files[side]:
            own = any(file_of(p) == file for p in ours)
            enemy = any(file_of(p) == file for p in theirs)
            w.bit(not own and not enemy)
            w.bit(not enemy and own)
    for side in (0, 1):
        for slot in range(3):
            w.one_hot(storm_bucket(storms[side][slot]), 4)

    rings = [set(b.step_attacks(kings[side], b.KING_STEPS)) for side in (0, 1)]
    weights = {"n": 1, "b": 1, "r": 2, "q": 4}
    attackers = [0, 0]
    weight = [0, 0]
    for side in (0, 1):
        for role in MINOR_ROLES:
            for square in scan.role_units[1 - side][role]:
                if scan.attacks_from[square] & rings[side]:
                    attackers[side] += 1
                    weight[side] += weights[role]
    for side in (0, 1):
        w.count(attackers[side], 6.0)
    for side in (0, 1):
        w.count(weight[side], 16.0)
    guarded = [set().union(*(scan.by_role[side][role] for role in COUNTED_ROLES)) for side in (0, 1)]
    for side in (0, 1):
        w.count(len(rings[side] & guarded[side]), 8.0)
    for side in (0, 1):
        w.count(len((rings[side] & scan.by[1 - side]) - guarded[side]), 8.0)
    for side in (0, 1):
        w.count(len(rings[side] - scan.units[side] - scan.by[1 - side]), 8.0)
    for side in (0, 1):
        rank = scan.relative_rank(kings[side], side)
        ahead = {
            scan.relative_square(file, 2, side)
            for file in [file_of(kings[side]), *adjacent_files(file_of(kings[side]))]
        }
        w.bit(rank == 1 and ahead <= scan.units[side])
    distance = chebyshev(kings[0], kings[1])
    w.one_hot(distance - 2 if 2 <= distance <= 7 else None, 6)
    for side in (0, 1):
        enemies = [square for role in MINOR_ROLES for square in scan.role_units[1 - side][role]]
        mean = sum(chebyshev(square, kings[side]) for square in enemies) / len(enemies) if enemies else 0.0
        w.count(f32(mean), 8.0)
    for side in (0, 1):
        w.count(len(attacks_of("q", kings[side], scan.colour[side], scan.occupied)), 27.0)


# --------------------------------------------------------------------------
# mobility and attacks

CENTRE = {square_at(file, rank) for file in (3, 4) for rank in (3, 4)}
EXTENDED_CENTRE = {square_at(file, rank) for file in range(2, 6) for rank in range(2, 6)}


def mobility_counts(scan: Scan) -> tuple[list[list[int]], list[list[int]], list[int]]:
    by_role = [[0] * 5 for _ in (0, 1)]
    safe = [[0] * 5 for _ in (0, 1)]
    total = [0, 0]
    for side in (0, 1):
        enemy_pawns = scan.by_role[1 - side]["p"]
        for index, role in enumerate(COUNTED_ROLES):
            free = scan.by_role[side][role] - scan.units[side]
            by_role[side][index] = len(free)
            safe[side][index] = len(free - enemy_pawns)
            total[side] += len(free)
    return by_role, safe, total


def mobility(scan: Scan, w: Writer) -> None:
    by_role, safe, total = mobility_counts(scan)
    both = total[0] + total[1]
    w.value(0.0 if both == 0 else f32(total[0] / both))
    for side in (0, 1):
        for index in range(5):
            w.count(by_role[side][index], 16.0)
    for side in (0, 1):
        for index in range(5):
            w.count(safe[side][index], 16.0)
    for index in range(5):
        w.diff(by_role[0][index] - by_role[1][index], 16.0)
    for side in (0, 1):
        w.count(len(scan.by[side] & scan.own_half(1 - side)), 32.0)
    w.count(len(scan.by[0]), 48.0)
    w.count(len(scan.by[1]), 48.0)
    w.diff(len(scan.by[0]) - len(scan.by[1]), 48.0)
    for side in (0, 1):
        w.count(len(scan.by[side] & CENTRE), 4.0)
    for side in (0, 1):
        w.count(len(scan.by[side] & EXTENDED_CENTRE), 16.0)
    for side in (0, 1):
        movable = scan.units[side] - scan.role_units[side]["p"] - scan.role_units[side]["k"]
        w.count(
            sum(1 for square in movable if not scan.attacks_from[square] - scan.units[side]),
            4.0,
        )
    for side in (0, 1):
        w.count(total[side], 96.0)


class AttackFacts:
    """Hanging, en prise, pinned and skewered, per side."""

    def __init__(self, scan: Scan) -> None:
        self.scan = scan
        self.attacked: list[set[int]] = [set(), set()]
        self.hanging: list[set[int]] = [set(), set()]
        self.en_prise: list[set[int]] = [set(), set()]
        self.pinned: list[set[int]] = [set(), set()]
        self.defended: list[set[int]] = [set(), set()]
        self.attacked_value = [0, 0]
        self.hanging_value = [0, 0]
        self.en_prise_value = [0, 0]
        self.en_prise_max = [0, 0]
        self.pinned_value = [0, 0]
        self.skewers = [0, 0]

        position = scan.position
        for side in (0, 1):
            self.defended[side] = scan.units[side] & scan.by[side]
            for square in scan.units[side] - scan.role_units[side]["k"]:
                role = role_at(position, square)
                attackers = b.attackers_of(position, square, scan.colour[1 - side])
                if not attackers:
                    continue
                self.attacked[side].add(square)
                self.attacked_value[side] += VALUE[role]
                defended = square in scan.by[side]
                cheaper = any(ORDER[role_at(position, origin)] < ORDER[role] for origin in attackers)
                if not defended:
                    self.hanging[side].add(square)
                    self.hanging_value[side] += VALUE[role]
                if not defended or cheaper:
                    self.en_prise[side].add(square)
                    self.en_prise_value[side] += VALUE[role]
                    self.en_prise_max[side] = max(self.en_prise_max[side], VALUE[role])
            self.pinned[side] = self.absolute_pins(side)
            self.pinned_value[side] = sum(VALUE[role_at(position, square)] for square in self.pinned[side])
            self.skewers[side] = self.count_skewers(side)

    def absolute_pins(self, side: int) -> set[int]:
        scan = self.scan
        king = scan.kings[side]
        pinned = set()
        for role, rays in (("b", b.BISHOP_RAYS), ("r", b.ROOK_RAYS)):
            reach = set(b.ray_attacks(king, rays, frozenset()))
            for slider in reach & (scan.role_units[1 - side][role] | scan.role_units[1 - side]["q"]):
                blockers = set(between(slider, king)) & scan.occupied
                if len(blockers) == 1 and blockers <= scan.units[side]:
                    pinned |= blockers
        return pinned

    def count_skewers(self, side: int) -> int:
        scan = self.scan
        position = scan.position
        count = 0
        for role in ("b", "r", "q"):
            for slider in scan.role_units[side][role]:
                reach = scan.attacks_from[slider]
                for front in reach & scan.units[1 - side]:
                    front_role = role_at(position, front)
                    xray = set(attacks_of(role, slider, scan.colour[side], scan.occupied - {front}))
                    behind = (xray - reach) & set(line_through(slider, front)) & scan.units[1 - side]
                    for back in behind:
                        if ORDER[role_at(position, back)] <= ORDER[front_role]:
                            count += 1
        return count


def attacks(scan: Scan, facts: AttackFacts, w: Writer) -> None:
    w.count(len(scan.by[0]), 48.0)
    w.count(len(scan.by[1]), 48.0)
    w.diff(len(scan.by[0]) - len(scan.by[1]), 48.0)
    for side in (0, 1):
        w.count(len(facts.attacked[side]), 8.0)
    for side in (0, 1):
        w.count(facts.attacked_value[side], 20.0)
    for side in (0, 1):
        w.count(len(facts.hanging[side]), 4.0)
    for side in (0, 1):
        w.count(facts.hanging_value[side], 20.0)
    for side in (0, 1):
        w.count(len(facts.en_prise[side]), 4.0)
    for side in (0, 1):
        w.count(facts.en_prise_value[side], 20.0)
    for side in (0, 1):
        w.count(facts.en_prise_max[side], 9.0)
    for side in (0, 1):
        w.count(len(facts.pinned[side]), 4.0)
    for side in (0, 1):
        w.count(facts.pinned_value[side], 20.0)
    for side in (0, 1):
        w.count(facts.skewers[side], 4.0)
    for side in (0, 1):
        w.count(len(facts.defended[side]), 16.0)


# --------------------------------------------------------------------------
# exchange


def least_valuable_attacker(
    position: Position, square: int, colour: str, occupied: frozenset[int]
) -> tuple[int, str] | None:
    """The cheapest unit of ``colour`` still on ``occupied`` that attacks ``square``."""
    best: tuple[int, str] | None = None
    for origin in sorted(occupied):
        piece = position.board[origin]
        if piece is None or piece[0] != colour:
            continue
        if square not in attacks_of(piece[1], origin, colour, occupied):
            continue
        if best is None or ORDER[piece[1]] < ORDER[best[1]]:
            best = (origin, piece[1])
    return best


def swap(position: Position, square: int, colour: str, occupant: int, occupied: frozenset[int]) -> int:
    """What ``colour`` wins by capturing the unit worth ``occupant`` on ``square``."""
    found = least_valuable_attacker(position, square, colour, occupied)
    if found is None:
        return 0
    origin, role = found
    left = occupied - {origin}
    # A king captures only what the other side has stopped defending.
    if role == "k" and least_valuable_attacker(position, square, other(colour), left) is not None:
        return 0
    promotes = role == "p" and relative_rank(square, colour) == 8
    landed = "q" if promotes else role
    gain = (
        occupant
        + (VALUE["q"] - VALUE["p"] if promotes else 0)
        - swap(position, square, other(colour), VALUE[landed], left)
    )
    return max(gain, 0)


def see_capture(position: Position, move: Move) -> int:
    """The static exchange evaluation of ``move``."""
    if move.castling:
        return 0
    us = position.side_to_move
    mover = role_at(position, move.frm)
    taken = victim_square(move)
    victim = position.board[taken]
    captured = 0 if victim is None else VALUE[victim[1]]
    landed = move.promotion or mover
    promoted = VALUE[landed] - VALUE["p"] if move.promotion else 0
    occupied = (occupancy(position) - {move.frm, taken}) | {move.to}
    return captured + promoted - swap(position, move.to, other(us), VALUE[landed], occupied)


def see_of_captures(position: Position | None) -> list[int]:
    """The SEE of every capture the side to move has."""
    if position is None:
        return []
    return [see_capture(position, move) for move in b.legal_moves(position) if move.capture]


def exchange_block(sees: list[int], w: Writer) -> None:
    w.diff(max(sees, default=0), 9.0)
    w.count(sum(1 for see in sees if see > 0), 8.0)
    w.count(sum(1 for see in sees if see == 0), 8.0)
    w.count(sum(see for see in sees if see > 0), 20.0)


# --------------------------------------------------------------------------
# tactics


def landing(move: Move) -> int:
    if move.castling:
        file = 6 if file_of(move.to) > file_of(move.frm) else 2
        return square_at(file, rank_of(move.frm))
    return move.to


def moved_to(move: Move) -> set[int]:
    out = {landing(move)}
    if move.castling:
        file = 5 if file_of(move.to) > file_of(move.frm) else 3
        out.add(square_at(file, rank_of(move.frm)))
    return out


def victim_square(move: Move) -> int:
    if move.en_passant:
        return square_at(file_of(move.to), rank_of(move.frm))
    return move.to


class Tactics:
    """One side's one-ply options, over the legal moves of ``position``."""

    def __init__(
        self,
        scan: Scan,
        position: Position | None,
        mover: int,
        attack_facts: AttackFacts,
    ) -> None:
        self.available = position is not None
        self.check_count = 0
        self.check_by_role = [False] * 5
        self.safe_check_count = 0
        self.safe_check_by_role = [False] * 5
        self.double_check = False
        self.discovered_check = False
        self.mate = False
        self.stalemate = False
        self.promotion_files: set[int] = set()
        self.promotion_roles = [False] * 4
        self.safe_promotion_files: set[int] = set()
        self.capture_count = 0
        self.winning_capture = False
        self.winning_gain = 0
        self.captures_hanging = False
        self.hanging_max = 0
        self.equal_captures = 0
        self.losing_captures = 0
        self.fork_count = 0
        self.fork_max = 0
        self.knight_fork = False
        self.royal_fork = False
        self.pin_creations = 0
        self.skewer_creation = False
        self.discovered_attack = False
        self.legal_moves = 0
        self.annotated: list[tuple[Move, dict[str, object]]] = []
        if position is None:
            return

        mover_colour = scan.colour[mover]
        enemy_colour = scan.colour[1 - mover]
        moves = b.legal_moves(position)
        self.legal_moves = len(moves)

        for move in moves:
            after = b.play(position, move)
            to = landing(move)
            mover_role = role_at(position, move.frm)
            landed_role = move.promotion or mover_role
            occupied = occupancy(after)

            pawn_attacked = any(
                after.board[origin] == (enemy_colour, "p") for origin in b.pawn_attacks(to, mover_colour)
            )
            enemy_attackers = b.attackers_of(after, to, enemy_colour)
            defenders = b.attackers_of(after, to, mover_colour)
            cheaper = any(ORDER[role_at(after, origin)] < ORDER[landed_role] for origin in enemy_attackers)
            is_safe = not pawn_attacked and not cheaper and (not enemy_attackers or bool(defenders))

            gives_check = b.is_attacked(after, after.king_of(enemy_colour), mover_colour)
            victim = after_victim(position, move) if move.capture else None
            hanging = attack_facts.hanging[1 - mover]
            captures_hanging = move.capture and victim_square(move) in hanging

            self.annotated.append(
                (
                    move,
                    {
                        "victim": victim,
                        "mover": mover_role,
                        "promotion": move.promotion,
                        "gives_check": gives_check,
                        "gives_safe_check": gives_check and is_safe,
                        "is_safe": is_safe,
                        "captures_hanging": captures_hanging,
                        "escapes_attack": move.frm in scan.by[1 - mover] and is_safe,
                        "to_attacked_by_pawn": pawn_attacked,
                        "is_castling": move.castling,
                        "is_en_passant": move.en_passant,
                    },
                )
            )

            if gives_check:
                self.check_count += 1
                if mover_role != "k":
                    self.check_by_role[COUNTED_ROLES.index(mover_role)] = True
                if is_safe:
                    self.safe_check_count += 1
                    if mover_role != "k":
                        self.safe_check_by_role[COUNTED_ROLES.index(mover_role)] = True
                givers = b.attackers_of(after, after.king_of(enemy_colour), mover_colour)
                if len(givers) >= 2:
                    self.double_check = True
                if set(givers) - moved_to(move):
                    self.discovered_check = True

            if not b.legal_moves(after):
                if gives_check:
                    self.mate = True
                else:
                    self.stalemate = True

            if move.promotion:
                self.promotion_files.add(file_of(move.to))
                self.promotion_roles["qrbn".index(move.promotion)] = True
                if is_safe:
                    self.safe_promotion_files.add(file_of(move.to))

            if victim is not None:
                self.capture_count += 1
                see = see_capture(position, move)
                self.winning_capture = self.winning_capture or see > 0
                self.winning_gain = max(self.winning_gain, max(see, 0))
                if captures_hanging:
                    self.captures_hanging = True
                    self.hanging_max = max(self.hanging_max, VALUE[victim])
                if see == 0:
                    self.equal_captures += 1
                elif see < 0:
                    self.losing_captures += 1

            enemy_units = set(after.squares_of(enemy_colour))
            reach = set(attacks_of(landed_role, to, mover_colour, occupied))
            targets = reach & enemy_units
            forked = 0
            fork_value = 0
            royal = False
            for target in targets:
                role = role_at(after, target)
                undefended = not b.attackers_of(after, target, enemy_colour)
                if ORDER[role] > ORDER[landed_role] or undefended:
                    forked += 1
                    fork_value = max(fork_value, TARGET[role])
                    royal = royal or role == "k"
            if forked >= 2:
                self.fork_count += 1
                self.fork_max = max(self.fork_max, fork_value)
                self.knight_fork = self.knight_fork or landed_role == "n"
                self.royal_fork = self.royal_fork or royal

            if landed_role in ("b", "r", "q"):
                pins = False
                for front in targets:
                    front_role = role_at(after, front)
                    xray = set(attacks_of(landed_role, to, mover_colour, occupied - {front}))
                    behind = (xray - reach) & set(line_through(to, front)) & enemy_units
                    for back in behind:
                        back_role = role_at(after, back)
                        if back_role == "k" or ORDER[back_role] > ORDER[front_role]:
                            pins = True
                        if ORDER[back_role] <= ORDER[front_role]:
                            self.skewer_creation = True
                if pins:
                    self.pin_creations += 1

            if not self.discovered_attack:
                stationary = moved_to(move) | {move.frm, move.to}
                for role in ("b", "r", "q"):
                    for origin in scan.role_units[mover][role] - stationary:
                        gained = set(attacks_of(role, origin, mover_colour, occupied)) - scan.attacks_from[origin]
                        if any(VALUE[role_at(after, target)] >= 3 for target in gained & enemy_units):
                            self.discovered_attack = True
                            break
                    if self.discovered_attack:
                        break


def after_victim(position: Position, move: Move) -> str | None:
    piece = position.board[victim_square(move)]
    return None if piece is None else piece[1]


def tactics_block(t: Tactics, w: Writer) -> None:
    w.bit(t.check_count > 0)
    w.count(t.check_count, 8.0)
    for flag in t.check_by_role:
        w.bit(flag)
    w.bit(t.safe_check_count > 0)
    w.count(t.safe_check_count, 8.0)
    for flag in t.safe_check_by_role:
        w.bit(flag)
    w.bit(t.double_check)
    w.bit(t.discovered_check)
    w.bit(t.mate)
    w.bit(t.stalemate)
    w.bit(bool(t.promotion_files))
    w.mask8(t.promotion_files)
    for flag in t.promotion_roles:
        w.bit(flag)
    w.bit(bool(t.safe_promotion_files))
    w.mask8(t.safe_promotion_files)
    w.bit(t.capture_count > 0)
    w.count(t.capture_count, 16.0)
    w.bit(t.winning_capture)
    w.count(t.winning_gain, 9.0)
    w.bit(t.captures_hanging)
    w.count(t.hanging_max, 9.0)
    w.count(t.equal_captures, 8.0)
    w.count(t.losing_captures, 8.0)
    w.bit(t.fork_count > 0)
    w.count(t.fork_count, 4.0)
    w.count(t.fork_max, 9.0)
    w.bit(t.knight_fork)
    w.bit(t.royal_fork)
    w.bit(t.pin_creations > 0)
    w.count(t.pin_creations, 4.0)
    w.bit(t.skewer_creation)
    w.bit(t.discovered_attack)
    w.count(t.legal_moves, 64.0)
    w.bit(t.available and t.legal_moves <= 2)
    w.bit(t.available)


# --------------------------------------------------------------------------
# endgame

#: d4, e4, d5 and e5.
CENTRE_SQUARES = (27, 28, 35, 36)

#: The race plies of a side with no passer.
NO_RACE = 8


def centralisation(square: int) -> int:
    """The Chebyshev distance from ``square`` to the nearest central square."""
    return min(chebyshev(square, centre) for centre in CENTRE_SQUARES)


def race_plies(lead_rank: int | None, side: int) -> int:
    """The plies the most advanced passer needs, or ``NO_RACE`` without one."""
    if lead_rank is None:
        return NO_RACE
    return max(8 - lead_rank - (1 if side == 0 else 0), 0)


def opposition(scan: Scan) -> int | None:
    """0 for a direct opposition, 1 for a distant one, ``None`` for neither."""
    corridor = between(scan.kings[0], scan.kings[1])
    if set(corridor) & scan.occupied:
        return None
    if len(corridor) == 1:
        return 0
    return 1 if len(corridor) in (3, 5) else None


def key_squares(scan: Scan, pawn: int, side: int) -> set[int]:
    """The key squares of one pawn; a rook pawn has none."""
    file = file_of(pawn)
    if file in (0, 7):
        return set()
    rank = scan.relative_rank(pawn, side)
    ahead = rank + 2 if rank <= 4 else rank + 1
    return {scan.relative_square(neighbour, ahead, side) for neighbour in (file - 1, file, file + 1)}


def wrong_colour_bishop(scan: Scan, side: int) -> bool:
    bishops = scan.role_units[side]["b"]
    pawns = scan.role_units[side]["p"]
    if not bishops or not pawns or any(file_of(pawn) not in (0, 7) for pawn in pawns):
        return False
    dark = all(b.is_dark(square) for square in bishops)
    if not dark and any(b.is_dark(square) for square in bishops):
        return False
    return all(b.is_dark(scan.relative_square(file_of(pawn), 8, side)) != dark for pawn in pawns)


def bare_king(scan: Scan, side: int) -> bool:
    return not any(scan.role_units[side][role] for role in COUNTED_ROLES)


def drawish_material(scan: Scan) -> int | None:
    """0 for two knights, 1 for a wrong bishop, 2 for opposite bishops."""
    for side in (0, 1):
        if not bare_king(scan, 1 - side):
            continue
        units = scan.role_units[side]
        if len(units["n"]) == 2 and not (units["p"] | units["b"] | units["r"] | units["q"]):
            return 0
        if wrong_colour_bishop(scan, side) and not (units["n"] | units["r"] | units["q"]):
            return 1
    bishops = [scan.role_units[side]["b"] for side in (0, 1)]
    opposite = (
        len(bishops[0]) == 1
        and len(bishops[1]) == 1
        and b.is_dark(next(iter(bishops[0]))) != b.is_dark(next(iter(bishops[1])))
    )
    if opposite and not any(scan.role_units[side][role] for side in (0, 1) for role in ("n", "r", "q")):
        return 2
    return None


def endgame(scan: Scan, facts: PawnFacts, w: Writer) -> None:
    for side in (0, 1):
        w.count(centralisation(scan.kings[side]), 3.0)
    plies = [race_plies(facts.lead_rank[side], side) for side in (0, 1)]
    for side in (0, 1):
        w.count(plies[side], 8.0)
    w.diff(plies[0] - plies[1], 8.0)
    stand = opposition(scan)
    w.one_hot(2 if stand is None else stand, 3)
    for side in (0, 1):
        w.bit(any(scan.kings[side] in key_squares(scan, pawn, side) for pawn in facts.passed[side]))
    for side in (0, 1):
        w.bit(wrong_colour_bishop(scan, side))
    w.one_hot(drawish_material(scan), 3)


# --------------------------------------------------------------------------
# planes


def planes(scan: Scan, facts: AttackFacts, w: Writer) -> None:
    for squares in (
        scan.by[0],
        scan.by[1],
        scan.by_role[0]["p"],
        scan.by_role[1]["p"],
        facts.hanging[0],
        facts.hanging[1],
        facts.pinned[0],
        facts.pinned[1],
    ):
        w.plane(squares, scan.us)


# --------------------------------------------------------------------------


def encode(fen: str, variant: str = "chess") -> list[float]:
    """The v1 values of ``fen`` under ``variant``."""
    position = b.parse_fen(fen)
    scan = Scan(position)
    pawn_facts = PawnFacts(scan)
    attack_facts = AttackFacts(scan)

    writers: dict[str, Writer] = {name: Writer() for name, _width, _features in SCHEMA}

    placement(scan, writers["placement"])
    state(scan, writers["state"])
    history(scan, writers["history"])
    material(scan, writers["material"])
    pawns(scan, pawn_facts, writers["pawns"])
    pieces(scan, pawn_facts, writers["pieces"])
    king(scan, writers["king"])
    mobility(scan, writers["mobility"])
    attacks(scan, attack_facts, writers["attacks"])
    null = b.null_move(position)
    exchange_block(see_of_captures(position), writers["exchange"])
    exchange_block(see_of_captures(null), writers["exchange"])
    tactics_block(Tactics(scan, position, 0, attack_facts), writers["tactics"])
    tactics_block(Tactics(scan, null, 1, attack_facts), writers["tactics"])
    endgame(scan, pawn_facts, writers["endgame"])
    planes(scan, attack_facts, writers["planes"])

    values: list[float] = []
    for name, width, features in SCHEMA:
        group = writers[name].values
        assert len(group) == width, f"{name} wrote {len(group)} of {width}"
        for feature, offset, feature_width in features:
            if variant != "chess" and (name, feature) in CLASSIC_ONLY:
                group[offset : offset + feature_width] = [0.0] * feature_width
        values.extend(group)
    return values
