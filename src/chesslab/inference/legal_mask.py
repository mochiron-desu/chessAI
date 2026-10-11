"""Legal move generation and logit masking for policy inference.

Guarantees that the policy head can only select legal moves.
"""

from typing import Set
import numpy as np
import torch
import chess

from chesslab.encode.move_vocab import VOCAB_SIZE, encode_move


def compute_legal_mask(board: chess.Board) -> np.ndarray:
    """Returns a boolean array of length VOCAB_SIZE indicating legal moves."""
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    turn = board.turn
    for move in board.legal_moves:
        idx = encode_move(move, turn)
        mask[idx] = True
    return mask


def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Masks policy logits with -inf on all illegal moves.
    
    Args:
        logits: Float tensor of shape (..., VOCAB_SIZE).
        legal_mask: Bool tensor of shape (..., VOCAB_SIZE).
        
    Returns:
        Masked logits where illegal moves are -inf.
    """
    return logits.masked_fill(~legal_mask, float("-inf"))


def sample_legal_move(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False
) -> int:
    """Selects an action index from policy logits constrained to legal moves.
    
    Args:
        logits: 1D Float tensor of shape (VOCAB_SIZE,).
        legal_mask: 1D Bool tensor of shape (VOCAB_SIZE,).
        temperature: Sampling temperature (lower = more deterministic).
        greedy: If True, returns argmax over legal moves.
        
    Returns:
        int action index.
    """
    masked = mask_logits(logits, legal_mask)
    if greedy or temperature <= 1e-4:
        return int(torch.argmax(masked).item())
    
    probs = torch.softmax(masked / temperature, dim=-1)
    # Check for NaN / terminal state with 0 legal moves
    if torch.isnan(probs).any() or probs.sum() == 0:
        return int(torch.argmax(masked).item())
    
    action = torch.multinomial(probs, num_samples=1)
    return int(action.item())
