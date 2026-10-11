import pytest
import chess
import numpy as np
from chesslab.encode.board_planes import encode_board
from chesslab.encode.move_vocab import (
    VOCAB_SIZE,
    UCI_TO_IDX,
    encode_move,
    decode_move,
    move_to_perspective,
    move_from_perspective,
)


def test_vocab_size_and_uniqueness():
    assert VOCAB_SIZE > 1800
    assert len(UCI_TO_IDX) == VOCAB_SIZE


def test_initial_board_encoding():
    board = chess.Board()
    planes, rule_state = encode_board(board)

    assert planes.shape == (12, 8, 8)
    assert planes.dtype == np.float32
    assert rule_state.shape == (7,)
    assert rule_state.dtype == np.float32

    # In starting position, White has 16 pieces, Black has 16 pieces
    assert np.sum(planes[:6]) == 16.0
    assert np.sum(planes[6:]) == 16.0

    # Friendly pawns (plane 0) are on rank 1 (0-indexed)
    assert np.all(planes[0, 1, :] == 1.0)
    # Opponent pawns (plane 6) are on rank 6
    assert np.all(planes[6, 6, :] == 1.0)

    # Castling rights: all 4 are present
    assert rule_state[0] == 1.0  # Friendly K
    assert rule_state[1] == 1.0  # Friendly Q
    assert rule_state[2] == 1.0  # Opponent K
    assert rule_state[3] == 1.0  # Opponent Q

    # No en passant
    assert rule_state[4] == 0.0
    # Halfmove clock is 0
    assert rule_state[6] == 0.0


def test_perspective_flip_black_to_move():
    board = chess.Board()
    board.push_san("e4")  # Now Black to move

    planes, rule_state = encode_board(board)

    # Friendly pieces are now Black pieces
    # Black's pawns are on rank 1 from Black's perspective
    assert np.sum(planes[0]) == 8.0  # 8 friendly pawns
    assert np.all(planes[0, 1, :] == 1.0)

    # White's e4 pawn is on e5 from Black's perspective (rank 4, file 4)
    # In standard view, e4 is rank 3. Flipped: 3 ^ 7 = 4!
    assert planes[6, 4, 4] == 1.0

    # En passant: e3 is the ep square in standard (rank 2, file 4).
    # Flipped: rank 2 ^ 7 = 5. ep_available should be 1.0
    assert rule_state[4] == 1.0
    assert rule_state[5] == 4.0 / 7.0  # File 4 (e-file)


def test_move_encoding_roundtrip():
    # White move
    m_white = chess.Move.from_uci("e2e4")
    idx_w = encode_move(m_white, chess.WHITE)
    decoded_w = decode_move(idx_w, chess.WHITE)
    assert decoded_w == m_white

    # Black move: e7e5 from Black perspective is e2e4
    m_black = chess.Move.from_uci("e7e5")
    idx_b = encode_move(m_black, chess.BLACK)
    decoded_b = decode_move(idx_b, chess.BLACK)
    assert decoded_b == m_black

    # Check that Black's e7e5 has the exact same index as White's e2e4!
    # (Because e7e5 is e2e4 from Black's perspective)
    assert idx_w == idx_b

    # Promotion move
    m_promo = chess.Move.from_uci("a7a8n")
    idx_p = encode_move(m_promo, chess.WHITE)
    decoded_p = decode_move(idx_p, chess.WHITE)
    assert decoded_p == m_promo


def test_legal_moves_coverage():
    """Verify that all legal moves in arbitrary game states exist in the vocabulary."""
    board = chess.Board()
    # Play out a standard game line
    moves_san = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
    for san in moves_san:
        for legal_move in board.legal_moves:
            # Must not raise ValueError
            idx = encode_move(legal_move, board.turn)
            assert 0 <= idx < VOCAB_SIZE
        board.push_san(san)
