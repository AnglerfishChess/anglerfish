"""A slow, explicit chess model: FEN parsing, attacks and legal moves.

Squares are 0-63 with a1 = 0 and h8 = 63. Colours are ``"w"`` and ``"b"``,
roles are the lower-case FEN letters. Nothing here is optimised and nothing
here is shared with the Rust implementation it is used to check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

WHITE = "w"
BLACK = "b"
ROLES = "pnbrqk"

KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_STEPS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
BISHOP_RAYS = ((1, 1), (1, -1), (-1, -1), (-1, 1))
ROOK_RAYS = ((0, 1), (1, 0), (0, -1), (-1, 0))
QUEEN_RAYS = BISHOP_RAYS + ROOK_RAYS

VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
ORDER = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
TARGET = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 9}


def file_of(square: int) -> int:
    return square % 8


def rank_of(square: int) -> int:
    return square // 8


def square_at(file: int, rank: int) -> int:
    return rank * 8 + file


def relative_rank(square: int, colour: str) -> int:
    """The rank of ``square`` counted from ``colour``'s own back rank, from 1."""
    return rank_of(square) + 1 if colour == WHITE else 8 - rank_of(square)


def relative_square(file: int, rank: int, colour: str) -> int:
    """The square on ``file`` at ``colour``'s relative ``rank``, from 1."""
    return square_at(file, rank - 1 if colour == WHITE else 8 - rank)


def view(square: int, us: str) -> int:
    """``square`` as the side to move sees it: ranks flipped when Black moves."""
    return square if us == WHITE else square_at(file_of(square), 7 - rank_of(square))


def is_dark(square: int) -> bool:
    return file_of(square) % 2 == rank_of(square) % 2


def chebyshev(a: int, b: int) -> int:
    return max(abs(file_of(a) - file_of(b)), abs(rank_of(a) - rank_of(b)))


def other(colour: str) -> str:
    return BLACK if colour == WHITE else WHITE


@dataclass(frozen=True)
class Move:
    """Origin, destination, promotion role and kind.

    For castling the destination is the rook's own square, as in esca.
    """

    frm: int
    to: int
    promotion: str | None = None
    castling: bool = False
    en_passant: bool = False
    capture: bool = False


@dataclass(frozen=True)
class Position:
    """Placement, side to move, castling rights, en passant and the clocks."""

    board: tuple[tuple[str, str] | None, ...]
    side_to_move: str
    # colour -> (short rook file or None, long rook file or None)
    castling: tuple[tuple[int | None, int | None], tuple[int | None, int | None]]
    ep_square: int | None
    halfmove_clock: int
    fullmove_number: int
    clocks_known: bool

    def piece_at(self, square: int) -> tuple[str, str] | None:
        return self.board[square]

    def squares_of(self, colour: str, role: str | None = None) -> list[int]:
        return [
            square
            for square, piece in enumerate(self.board)
            if piece is not None and piece[0] == colour and (role is None or piece[1] == role)
        ]

    def king_of(self, colour: str) -> int:
        squares = self.squares_of(colour, "k")
        assert len(squares) == 1, "a position has one king per colour"
        return squares[0]

    def rights(self, colour: str) -> tuple[int | None, int | None]:
        return self.castling[0 if colour == WHITE else 1]


def role_at(position: Position, square: int) -> str:
    """The role standing on ``square``; the caller has already checked it is there."""
    piece = position.board[square]
    assert piece is not None, "a unit stands on its own square"
    return piece[1]


def parse_fen(text: str) -> Position:
    """Reads a six-field FEN, or a four-field one, which leaves the clocks unknown."""
    fields = text.split()
    assert len(fields) in (4, 6), f"a FEN has four or six fields: {text!r}"
    board: list[tuple[str, str] | None] = [None] * 64

    rank = 7
    file = 0
    for character in fields[0]:
        if character == "/":
            rank -= 1
            file = 0
        elif character.isdigit():
            file += int(character)
        else:
            colour = WHITE if character.isupper() else BLACK
            board[square_at(file, rank)] = (colour, character.lower())
            file += 1

    side = fields[1]
    castling = parse_castling(board, fields[2])
    ep_square = None if fields[3] == "-" else parse_square(fields[3])
    halfmove = int(fields[4]) if len(fields) == 6 else 0
    fullmove = int(fields[5]) if len(fields) == 6 else 1
    return Position(
        board=tuple(board),
        side_to_move=side,
        castling=castling,
        ep_square=ep_square,
        halfmove_clock=halfmove,
        fullmove_number=fullmove,
        clocks_known=len(fields) == 6,
    )


def parse_square(text: str) -> int:
    return square_at(ord(text[0]) - ord("a"), int(text[1]) - 1)


def parse_castling(
    board: list[tuple[str, str] | None], field: str
) -> tuple[tuple[int | None, int | None], tuple[int | None, int | None]]:
    """Reads both dialects: ``KQkq`` names the outermost rook, ``AHah`` its file."""
    rights: list[list[int | None]] = [[None, None], [None, None]]
    if field == "-":
        return ((None, None), (None, None))
    for character in field:
        colour = WHITE if character.isupper() else BLACK
        index = 0 if colour == WHITE else 1
        back = 0 if colour == WHITE else 7
        rooks = [file_of(square) for square in range(back * 8, back * 8 + 8) if board[square] == (colour, "r")]
        kings = [file_of(square) for square in range(back * 8, back * 8 + 8) if board[square] == (colour, "k")]
        letter = character.lower()
        if letter == "k":
            king = kings[0]
            rights[index][0] = max(file for file in rooks if file > king)
        elif letter == "q":
            king = kings[0]
            rights[index][1] = min(file for file in rooks if file < king)
        else:
            file = ord(letter) - ord("a")
            king = kings[0]
            rights[index][0 if file > king else 1] = file
    return (
        (rights[0][0], rights[0][1]),
        (rights[1][0], rights[1][1]),
    )


def pawn_attacks(square: int, colour: str) -> list[int]:
    """The squares a pawn of ``colour`` on ``square`` attacks."""
    step = 1 if colour == WHITE else -1
    rank = rank_of(square) + step
    if not 0 <= rank <= 7:
        return []
    out = []
    for file in (file_of(square) - 1, file_of(square) + 1):
        if 0 <= file <= 7:
            out.append(square_at(file, rank))
    return out


def step_attacks(square: int, steps: tuple[tuple[int, int], ...]) -> list[int]:
    out = []
    for dfile, drank in steps:
        file = file_of(square) + dfile
        rank = rank_of(square) + drank
        if 0 <= file <= 7 and 0 <= rank <= 7:
            out.append(square_at(file, rank))
    return out


def ray_attacks(square: int, rays: tuple[tuple[int, int], ...], occupied: frozenset[int]) -> list[int]:
    out = []
    for dfile, drank in rays:
        file = file_of(square) + dfile
        rank = rank_of(square) + drank
        while 0 <= file <= 7 and 0 <= rank <= 7:
            target = square_at(file, rank)
            out.append(target)
            if target in occupied:
                break
            file += dfile
            rank += drank
    return out


def attacks_of(role: str, square: int, colour: str, occupied: frozenset[int]) -> list[int]:
    """The squares a unit of ``role`` and ``colour`` on ``square`` attacks."""
    if role == "p":
        return pawn_attacks(square, colour)
    if role == "n":
        return step_attacks(square, KNIGHT_STEPS)
    if role == "k":
        return step_attacks(square, KING_STEPS)
    if role == "b":
        return ray_attacks(square, BISHOP_RAYS, occupied)
    if role == "r":
        return ray_attacks(square, ROOK_RAYS, occupied)
    return ray_attacks(square, QUEEN_RAYS, occupied)


def occupancy(position: Position) -> frozenset[int]:
    return frozenset(square for square, piece in enumerate(position.board) if piece is not None)


def attackers_of(position: Position, square: int, colour: str) -> list[int]:
    """The units of ``colour`` that attack ``square``."""
    occupied = occupancy(position)
    out = []
    for origin, piece in enumerate(position.board):
        if piece is None or piece[0] != colour:
            continue
        if square in attacks_of(piece[1], origin, colour, occupied):
            out.append(origin)
    return out


def is_attacked(position: Position, square: int, colour: str) -> bool:
    return bool(attackers_of(position, square, colour))


def between(a: int, b: int) -> list[int]:
    """The squares strictly between two squares on a common line, else empty."""
    dfile = file_of(b) - file_of(a)
    drank = rank_of(b) - rank_of(a)
    if not (dfile == 0 or drank == 0 or abs(dfile) == abs(drank)):
        return []
    stepf = (dfile > 0) - (dfile < 0)
    stepr = (drank > 0) - (drank < 0)
    out = []
    file = file_of(a) + stepf
    rank = rank_of(a) + stepr
    while (file, rank) != (file_of(b), rank_of(b)):
        out.append(square_at(file, rank))
        file += stepf
        rank += stepr
    return out


def line_through(a: int, b: int) -> list[int]:
    """Every square of the line two squares share, or empty when they share none."""
    dfile = file_of(b) - file_of(a)
    drank = rank_of(b) - rank_of(a)
    if a == b or not (dfile == 0 or drank == 0 or abs(dfile) == abs(drank)):
        return []
    stepf = (dfile > 0) - (dfile < 0)
    stepr = (drank > 0) - (drank < 0)
    out = [a]
    for direction in (1, -1):
        file = file_of(a) + stepf * direction
        rank = rank_of(a) + stepr * direction
        while 0 <= file <= 7 and 0 <= rank <= 7:
            out.append(square_at(file, rank))
            file += stepf * direction
            rank += stepr * direction
    return out


def pseudo_moves(position: Position) -> list[Move]:
    """Every move legal by movement and occupancy, check ignored."""
    us = position.side_to_move
    them = other(us)
    occupied = occupancy(position)
    moves: list[Move] = []

    for origin, piece in enumerate(position.board):
        if piece is None or piece[0] != us:
            continue
        role = piece[1]
        if role == "p":
            moves.extend(pawn_moves(position, origin, us, occupied))
            continue
        for target in attacks_of(role, origin, us, occupied):
            occupant = position.board[target]
            if occupant is not None and occupant[0] == us:
                continue
            moves.append(Move(origin, target, capture=occupant is not None))

    moves.extend(castling_moves(position, us, them, occupied))
    return moves


def pawn_moves(position: Position, origin: int, us: str, occupied: frozenset[int]) -> list[Move]:
    moves = []
    step = 8 if us == WHITE else -8
    ahead = origin + step
    promotes = relative_rank(ahead, us) == 8
    roles = ("q", "r", "b", "n") if promotes else (None,)
    if 0 <= ahead <= 63 and ahead not in occupied:
        for role in roles:
            moves.append(Move(origin, ahead, promotion=role))
        double = ahead + step
        if relative_rank(origin, us) == 2 and double not in occupied:
            moves.append(Move(origin, double))
    for target in pawn_attacks(origin, us):
        occupant = position.board[target]
        if occupant is not None and occupant[0] != us:
            for role in roles:
                moves.append(Move(origin, target, promotion=role, capture=True))
        elif target == position.ep_square:
            moves.append(Move(origin, target, en_passant=True, capture=True))
    return moves


def castling_moves(position: Position, us: str, them: str, occupied: frozenset[int]) -> list[Move]:
    """King-to-rook castlings, destinations c and g as in classic geometry."""
    moves = []
    king = position.king_of(us)
    back = rank_of(king)
    if relative_rank(king, us) != 1 or is_attacked(position, king, them):
        return moves
    for wing, king_file, rook_target_file in ((0, 6, 5), (1, 2, 3)):
        rook = position.rights(us)[wing]
        if rook is None:
            continue
        rook_square = square_at(rook, back)
        if position.board[rook_square] != (us, "r"):
            continue
        king_target = square_at(king_file, back)
        rook_target = square_at(rook_target_file, back)
        path = set(between(king, king_target)) | {king_target}
        path |= set(between(rook_square, rook_target)) | {rook_target}
        path -= {king, rook_square}
        if path & occupied:
            continue
        walk = set(between(king, king_target)) | {king_target, king}
        if any(is_attacked(position, square, them) for square in walk):
            continue
        moves.append(Move(king, rook_square, castling=True))
    return moves


def play(position: Position, move: Move) -> Position:
    """The position after ``move``; the clocks follow the FIDE rules."""
    board = list(position.board)
    us = position.side_to_move
    them = other(us)
    piece = board[move.frm]
    assert piece is not None, "a move starts on an occupied square"
    role = piece[1]

    captures = move.capture
    if move.castling:
        back = rank_of(move.frm)
        king_file = 6 if file_of(move.to) > file_of(move.frm) else 2
        rook_file = 5 if king_file == 6 else 3
        board[move.frm] = None
        board[move.to] = None
        board[square_at(king_file, back)] = (us, "k")
        board[square_at(rook_file, back)] = (us, "r")
    else:
        board[move.frm] = None
        if move.en_passant:
            board[square_at(file_of(move.to), rank_of(move.frm))] = None
        board[move.to] = (us, move.promotion or role)

    rights = [list(position.castling[0]), list(position.castling[1])]
    index = 0 if us == WHITE else 1
    if role == "k":
        rights[index] = [None, None]
    if role == "r":
        for wing in (0, 1):
            if rights[index][wing] == file_of(move.frm) and relative_rank(move.frm, us) == 1:
                rights[index][wing] = None
    other_index = 1 - index
    for wing in (0, 1):
        file = rights[other_index][wing]
        if file is not None and move.to == relative_square(file, 1, them):
            rights[other_index][wing] = None

    ep_square = None
    if role == "p" and abs(rank_of(move.to) - rank_of(move.frm)) == 2:
        ep_square = (move.frm + move.to) // 2

    clock = 0 if role == "p" or captures else position.halfmove_clock + 1
    return replace(
        position,
        board=tuple(board),
        side_to_move=them,
        castling=((rights[0][0], rights[0][1]), (rights[1][0], rights[1][1])),
        ep_square=ep_square,
        halfmove_clock=clock,
        fullmove_number=position.fullmove_number + (1 if us == BLACK else 0),
    )


def legal_moves(position: Position) -> list[Move]:
    """Every pseudo-legal move that does not leave its own king attacked."""
    us = position.side_to_move
    them = other(us)
    out = []
    for move in pseudo_moves(position):
        after = play(position, move)
        if not is_attacked(after, after.king_of(us), them):
            out.append(move)
    return out


def checkers(position: Position) -> list[int]:
    us = position.side_to_move
    return attackers_of(position, position.king_of(us), other(us))


def in_check(position: Position) -> bool:
    return bool(checkers(position))


def null_move(position: Position) -> Position | None:
    """The same placement with the other side to move; ``None`` while in check."""
    if in_check(position):
        return None
    return replace(
        position,
        side_to_move=other(position.side_to_move),
        ep_square=None,
    )
