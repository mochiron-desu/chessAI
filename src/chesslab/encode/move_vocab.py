"""Deterministic UCI move vocabulary for standard chess.

Maps every geometrically possible move on an 8x8 chessboard to a fixed discrete index.
Guarantees 100% coverage of all legal moves in standard chess.
"""

from typing import Dict, List, Optional
import chess


def _generate_all_possible_uci_moves() -> List[str]:
    """Generates all geometrically possible UCI moves in standard chess.
    
    Includes:
    - All queen rays (straight and diagonal)
    - All knight moves
    - All pawn single and double pushes, and diagonal captures
    - All pawn promotions to Queen, Rook, Bishop, Knight (from rank 7 to 8)
    - King moves and castling (e1g1, e1c1, etc.)
    """
    moves = set()
    files = "abcdefgh"
    ranks = "12345678"

    for f1 in range(8):
        for r1 in range(8):
            sq1_name = f"{files[f1]}{ranks[r1]}"
            for f2 in range(8):
                for r2 in range(8):
                    if f1 == f2 and r1 == r2:
                        continue
                    sq2_name = f"{files[f2]}{ranks[r2]}"
                    df = abs(f1 - f2)
                    dr = abs(r1 - r2)

                    # Ray moves (horizontal, vertical, diagonal)
                    is_ray = (f1 == f2) or (r1 == r2) or (df == dr)
                    # Knight moves
                    is_knight = (df == 1 and dr == 2) or (df == 2 and dr == 1)

                    if is_ray or is_knight:
                        # Regular piece move (King, Queen, Rook, Bishop, Knight, or non-promoting pawn)
                        moves.add(f"{sq1_name}{sq2_name}")
                        # Pawn promotion moves (from player's perspective, rank 7 to rank 8, df <= 1)
                        if r1 == 6 and r2 == 7 and df <= 1:
                            for promo in ["q", "r", "b", "n"]:
                                moves.add(f"{sq1_name}{sq2_name}{promo}")

    return sorted(list(moves))


ALL_UCI_MOVES: List[str] = _generate_all_possible_uci_moves()
UCI_TO_IDX: Dict[str, int] = {move: idx for idx, move in enumerate(ALL_UCI_MOVES)}
IDX_TO_UCI: List[str] = ALL_UCI_MOVES
VOCAB_SIZE: int = len(ALL_UCI_MOVES)
MASK_NUM_BYTES: int = (VOCAB_SIZE + 7) // 8


def move_to_perspective(move: chess.Move, color: chess.Color) -> chess.Move:
    """Transforms a move into the current player's perspective.
    
    If color is White, the move is unchanged.
    If color is Black, squares are flipped vertically (sq ^ 56), so rank 8 -> 1, rank 7 -> 2.
    """
    if color == chess.WHITE:
        return move
    
    from_sq = move.from_square ^ 56
    to_sq = move.to_square ^ 56
    return chess.Move(from_sq, to_sq, promotion=move.promotion)


def move_from_perspective(move: chess.Move, color: chess.Color) -> chess.Move:
    """Inverts a perspective move back to the standard board frame."""
    return move_to_perspective(move, color)


def encode_move(move: chess.Move, color: chess.Color) -> int:
    """Encodes a chess.Move from the player's perspective into an integer index."""
    persp_move = move_to_perspective(move, color)
    uci = persp_move.uci()
    if uci not in UCI_TO_IDX:
        raise ValueError(f"Move {uci} (original {move.uci()}) not found in vocabulary")
    return UCI_TO_IDX[uci]


def decode_move(idx: int, color: chess.Color) -> chess.Move:
    """Decodes an integer index back to a chess.Move in the standard board frame."""
    if idx < 0 or idx >= VOCAB_SIZE:
        raise ValueError(f"Index {idx} out of bounds for vocab size {VOCAB_SIZE}")
    uci = IDX_TO_UCI[idx]
    persp_move = chess.Move.from_uci(uci)
    return move_from_perspective(persp_move, color)
