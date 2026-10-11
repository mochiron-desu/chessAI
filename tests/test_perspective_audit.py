"""Empirical audit verifying policy vocabulary, legal masking, and WDL perspective."""

import pytest
import chess
import torch
import torch.nn.functional as F

from chesslab.models.resnet import ChessResNet
from chesslab.encode.board_planes import encode_board
from chesslab.encode.move_vocab import encode_move, decode_move, ALL_UCI_MOVES, VOCAB_SIZE
from chesslab.inference.legal_mask import compute_legal_mask, mask_logits, sample_legal_move


def test_vocab_coverage_and_bijection():
    """Verify that every legal move in arbitrary positions round-trips through vocabulary."""
    board = chess.Board()
    for _ in range(50):
        for move in board.legal_moves:
            idx = encode_move(move, board.turn)
            assert 0 <= idx < VOCAB_SIZE
            decoded = decode_move(idx, board.turn)
            assert decoded == move
        # Make a move
        board.push(list(board.legal_moves)[0])


def test_perspective_reversal_symmetry():
    """Verify that vertical board flipping produces identical perspective encodings for White and Black."""
    # White e4 vs Black e5
    board_w = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    board_b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    planes_w, rule_w = encode_board(board_w)
    planes_b, rule_b = encode_board(board_b)

    # In board_w: Black is to move, White has e4 pawn (opponent e4 -> friendly perspective rank 4)
    # In board_b: White is to move, Black has e5 pawn (opponent e5 -> friendly perspective rank 4)
    # Both boards have identical relative perspective!
    assert (planes_w == planes_b).all()
    assert (rule_w == rule_b).all()


def test_wdl_perspective_empirical():
    """Empirically verify that WDL value head outputs are from the perspective of the side to move."""
    checkpoint_path = "checkpoints/run_1m/final_ema_model.pt"
    device = torch.device("cpu")
    model = ChessResNet(num_blocks=6, channels=64)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    # Position 1: White has Queen advantage, White to move
    # White: Kh1, Qd1, pawns a2..h2; Black: Kh8, pawns a7..h7 (Black has NO queen)
    fen_white_winning_white_turn = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1"
    b_w_win_w_turn = chess.Board(fen_white_winning_white_turn)

    # Position 2: Same board, but Black to move (Black is down a Queen)
    fen_white_winning_black_turn = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R b KQkq - 0 1"
    b_w_win_b_turn = chess.Board(fen_white_winning_black_turn)

    # Position 3: Black has Queen advantage, Black to move
    fen_black_winning_black_turn = "rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    b_b_win_b_turn = chess.Board(fen_black_winning_black_turn)

    # Position 4: Black has Queen advantage, White to move (White is down a Queen)
    fen_black_winning_white_turn = "rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
    b_b_win_white_turn = chess.Board(fen_black_winning_white_turn)

    def get_val(board):
        planes, rule = encode_board(board)
        pt = torch.from_numpy(planes).unsqueeze(0)
        rt = torch.from_numpy(rule).unsqueeze(0)
        with torch.no_grad():
            _, wdl = model(pt, rt)
            probs = F.softmax(wdl, dim=-1).squeeze(0)
            p_win = probs[0].item()
            p_loss = probs[2].item()
            return p_win - p_loss

    val_w_win_w_turn = get_val(b_w_win_w_turn)
    val_w_win_b_turn = get_val(b_w_win_b_turn)
    val_b_win_b_turn = get_val(b_b_win_b_turn)
    val_b_win_w_turn = get_val(b_b_win_white_turn)

    print(f"White winning, White turn: {val_w_win_w_turn:+.4f} (expected > 0)")
    print(f"White winning, Black turn: {val_w_win_b_turn:+.4f} (expected < 0)")
    print(f"Black winning, Black turn: {val_b_win_b_turn:+.4f} (expected > 0)")
    print(f"Black winning, White turn: {val_b_win_w_turn:+.4f} (expected < 0)")

    # When player to move has massive advantage, value must be positive
    assert val_w_win_w_turn > 0.0, "White turn with Queen up should be positive"
    assert val_b_win_b_turn > 0.0, "Black turn with Queen up should be positive"

    # When player to move is down a Queen, value must be negative
    assert val_w_win_b_turn < 0.0, "Black turn when down a Queen should be negative"
    assert val_b_win_w_turn < 0.0, "White turn when down a Queen should be negative"
