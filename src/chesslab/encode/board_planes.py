"""Encodes chess.Board into 12 spatial planes (8x8) and 7 rule-state scalars.

Always flips board perspective to the side to move (White or Black), ensuring
the neural network operates on perspective-invariant representations.
"""

from typing import Tuple
import numpy as np
import chess


PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def encode_board(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes a chess board into (planes, rule_state).
    
    Args:
        board: A python-chess Board object.
        
    Returns:
        planes: np.ndarray of shape (12, 8, 8), float32.
                Planes 0..5: Friendly pieces (P, N, B, R, Q, K).
                Planes 6..11: Opponent pieces (P, N, B, R, Q, K).
                Oriented such that rank 0 is the current player's back rank.
        rule_state: np.ndarray of shape (7,), float32.
                [friendly_k_castle, friendly_q_castle,
                 opp_k_castle, opp_q_castle,
                 ep_available, ep_file_norm, halfmove_clock_norm]
    """
    turn = board.turn
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    friendly_color = turn
    opponent_color = not turn

    # Populate 12 planes
    for p_idx, piece_type in enumerate(PIECE_TYPES):
        # Friendly pieces (0..5)
        for sq in board.pieces(piece_type, friendly_color):
            view_sq = sq if turn == chess.WHITE else (sq ^ 56)
            rank = view_sq // 8
            file = view_sq % 8
            planes[p_idx, rank, file] = 1.0

        # Opponent pieces (6..11)
        for sq in board.pieces(piece_type, opponent_color):
            view_sq = sq if turn == chess.WHITE else (sq ^ 56)
            rank = view_sq // 8
            file = view_sq % 8
            planes[p_idx + 6, rank, file] = 1.0

    # Rule state vector
    friendly_k_castle = 1.0 if board.has_kingside_castling_rights(friendly_color) else 0.0
    friendly_q_castle = 1.0 if board.has_queenside_castling_rights(friendly_color) else 0.0
    opp_k_castle = 1.0 if board.has_kingside_castling_rights(opponent_color) else 0.0
    opp_q_castle = 1.0 if board.has_queenside_castling_rights(opponent_color) else 0.0

    ep_sq = board.ep_square
    if ep_sq is not None:
        view_ep_sq = ep_sq if turn == chess.WHITE else (ep_sq ^ 56)
        ep_available = 1.0
        ep_file_norm = float(view_ep_sq % 8) / 7.0
    else:
        ep_available = 0.0
        ep_file_norm = 0.0

    halfmove_norm = min(board.halfmove_clock, 100) / 100.0

    rule_state = np.array([
        friendly_k_castle,
        friendly_q_castle,
        opp_k_castle,
        opp_q_castle,
        ep_available,
        ep_file_norm,
        halfmove_norm
    ], dtype=np.float32)

    return planes, rule_state
