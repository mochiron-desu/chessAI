import pytest
import chess
import numpy as np
import torch
from chesslab.inference.legal_mask import compute_legal_mask, mask_logits, sample_legal_move
from chesslab.encode.move_vocab import decode_move, VOCAB_SIZE


def test_initial_position_legal_mask():
    board = chess.Board()
    mask = compute_legal_mask(board)

    assert mask.shape == (VOCAB_SIZE,)
    assert mask.dtype == bool
    # Starting position has exactly 20 legal moves (16 pawn pushes + 4 knight hops)
    assert mask.sum() == 20
    assert len(list(board.legal_moves)) == 20


def test_mask_logits_probabilities():
    board = chess.Board()
    mask_np = compute_legal_mask(board)
    mask_tensor = torch.from_numpy(mask_np)

    # Random raw logits
    torch.manual_seed(42)
    raw_logits = torch.randn(VOCAB_SIZE)

    masked = mask_logits(raw_logits, mask_tensor)
    probs = torch.softmax(masked, dim=-1)

    # All illegal moves must have probability exactly 0.0
    illegal_indices = np.where(~mask_np)[0]
    legal_indices = np.where(mask_np)[0]

    assert torch.all(probs[illegal_indices] == 0.0)
    assert torch.all(probs[legal_indices] > 0.0)
    assert pytest.approx(probs.sum().item(), abs=1e-5) == 1.0


def test_sample_legal_move_guarantee():
    board = chess.Board()
    mask_tensor = torch.from_numpy(compute_legal_mask(board))
    raw_logits = torch.randn(VOCAB_SIZE)

    # Greedy
    greedy_idx = sample_legal_move(raw_logits, mask_tensor, greedy=True)
    greedy_move = decode_move(greedy_idx, board.turn)
    assert greedy_move in board.legal_moves

    # Sampled (run 50 times across varied temperatures)
    for temp in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(25):
            idx = sample_legal_move(raw_logits, mask_tensor, temperature=temp)
            move = decode_move(idx, board.turn)
            assert move in board.legal_moves
