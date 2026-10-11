"""Match evaluation harness and Elo rating estimation.

Conducts controlled matches with balanced openings and color-swapped pairs,
computing score, draw rate, and estimated Elo with 95% confidence intervals.
"""

import math
from typing import Dict, Any, List, Optional
import chess

from chesslab.inference.engine_uci import UCIEngine


def play_game(
    white_engine: UCIEngine,
    black_engine: UCIEngine,
    opening_moves: Optional[List[chess.Move]] = None,
    max_plies: int = 150
) -> str:
    """Plays a single game between two UCIEngines.
    
    Returns:
        "1-0" if White wins, "0-1" if Black wins, "1/2-1/2" if Draw.
    """
    board = chess.Board()

    # Apply opening moves
    if opening_moves:
        for m in opening_moves:
            board.push(m)

    white_engine.board = board.copy()
    black_engine.board = board.copy()

    while not board.is_game_over() and len(board.move_stack) < max_plies:
        turn = board.turn
        if turn == chess.WHITE:
            move = white_engine.select_best_move(temperature=0.0)
        else:
            move = black_engine.select_best_move(temperature=0.0)

        if move == chess.Move.null() or move not in board.legal_moves:
            # Forfeit if illegal or null
            return "0-1" if turn == chess.WHITE else "1-0"

        board.push(move)
        white_engine.board.push(move)
        black_engine.board.push(move)

    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    else:
        return "1/2-1/2"


def compute_elo_with_ci(wins: int, draws: int, losses: int) -> Dict[str, float]:
    """Computes empirical score, estimated delta Elo, and 95% confidence interval."""
    total = wins + draws + losses
    if total == 0:
        return {"total_games": 0, "score": 0.5, "delta_elo": 0.0, "ci_95": 0.0}

    score = (wins + 0.5 * draws) / total
    draw_rate = draws / total

    # Avoid log(0) at extreme scores
    clipped_score = max(0.01, min(0.99, score))
    delta_elo = -400.0 * math.log10(1.0 / clipped_score - 1.0)

    # Variance and standard error
    variance = (1.0 - draw_rate) / 4.0
    se_score = math.sqrt(variance / total)
    slope = 400.0 / (math.log(10.0) * clipped_score * (1.0 - clipped_score))
    ci_95 = 1.96 * se_score * slope

    return {
        "total_games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "draw_rate": draw_rate,
        "delta_elo": delta_elo,
        "ci_95": ci_95,
    }


def run_match(
    engine_a: UCIEngine,
    engine_b: UCIEngine,
    opening_pairs: List[List[chess.Move]],
    max_plies: int = 120
) -> Dict[str, Any]:
    """Runs a paired match with swapped colors to eliminate first-move bias."""
    wins_a = 0
    draws = 0
    losses_a = 0

    for pair_idx, openings in enumerate(opening_pairs):
        # Game 1: A is White, B is Black
        res1 = play_game(engine_a, engine_b, opening_moves=openings, max_plies=max_plies)
        if res1 == "1-0":
            wins_a += 1
        elif res1 == "0-1":
            losses_a += 1
        else:
            draws += 1

        # Game 2: B is White, A is Black (Colors swapped)
        res2 = play_game(engine_b, engine_a, opening_moves=openings, max_plies=max_plies)
        if res2 == "0-1":  # Black (A) won
            wins_a += 1
        elif res2 == "1-0":  # White (B) won
            losses_a += 1
        else:
            draws += 1

    stats = compute_elo_with_ci(wins_a, draws, losses_a)
    return stats
