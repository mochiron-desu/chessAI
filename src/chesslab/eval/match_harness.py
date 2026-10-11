"""Match evaluation harness and Elo rating estimation.

Conducts controlled matches with balanced openings and color-swapped pairs,
computing score, draw rate, and estimated Elo with 95% confidence intervals.
"""

import math
from typing import Dict, Any, List, Optional
import chess

from chesslab.inference.engine_uci import UCIEngine


import time
import numpy as np


def play_game(
    white_engine: UCIEngine,
    black_engine: UCIEngine,
    opening_moves: Optional[List[chess.Move]] = None,
    max_plies: int = 150,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Plays a single game between two UCIEngines with latency and node profiling.
    
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

    if getattr(white_engine, "searcher", None) is not None:
        white_engine.searcher.clear_caches()
    if getattr(black_engine, "searcher", None) is not None:
        black_engine.searcher.clear_caches()

    while not board.is_game_over() and len(board.move_stack) < max_plies:
        turn = board.turn
        curr_engine = white_engine if turn == chess.WHITE else black_engine
        engine_key = "white" if turn == chess.WHITE else "black"

        t0 = time.perf_counter()
        move = curr_engine.select_best_move(temperature=0.0)
        dt = (time.perf_counter() - t0) * 1000.0  # ms

        if perf_stats is not None:
            perf_stats[f"latency_{engine_key}"].append(dt)
            searcher = getattr(curr_engine, "searcher", None)
            if searcher is not None:
                perf_stats[f"nodes_{engine_key}"].append(searcher.nodes_visited)
                perf_stats[f"nn_evals_{engine_key}"].append(searcher.nn_evals)
            else:
                perf_stats[f"nodes_{engine_key}"].append(1)
                perf_stats[f"nn_evals_{engine_key}"].append(1)

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

    latencies_a: List[float] = []
    latencies_b: List[float] = []
    nodes_a: List[int] = []
    nodes_b: List[int] = []
    nn_evals_a: List[int] = []
    nn_evals_b: List[int] = []

    for pair_idx, openings in enumerate(opening_pairs):
        # Game 1: A is White, B is Black
        stats_g1 = {"latency_white": [], "latency_black": [], "nodes_white": [], "nodes_black": [], "nn_evals_white": [], "nn_evals_black": []}
        res1 = play_game(engine_a, engine_b, opening_moves=openings, max_plies=max_plies, perf_stats=stats_g1)
        if res1 == "1-0":
            wins_a += 1
        elif res1 == "0-1":
            losses_a += 1
        else:
            draws += 1

        latencies_a.extend(stats_g1["latency_white"])
        latencies_b.extend(stats_g1["latency_black"])
        nodes_a.extend(stats_g1["nodes_white"])
        nodes_b.extend(stats_g1["nodes_black"])
        nn_evals_a.extend(stats_g1["nn_evals_white"])
        nn_evals_b.extend(stats_g1["nn_evals_black"])

        # Game 2: B is White, A is Black (Colors swapped)
        stats_g2 = {"latency_white": [], "latency_black": [], "nodes_white": [], "nodes_black": [], "nn_evals_white": [], "nn_evals_black": []}
        res2 = play_game(engine_b, engine_a, opening_moves=openings, max_plies=max_plies, perf_stats=stats_g2)
        if res2 == "0-1":  # Black (A) won
            wins_a += 1
        elif res2 == "1-0":  # White (B) won
            losses_a += 1
        else:
            draws += 1

        latencies_a.extend(stats_g2["latency_black"])
        latencies_b.extend(stats_g2["latency_white"])
        nodes_a.extend(stats_g2["nodes_black"])
        nodes_b.extend(stats_g2["nodes_white"])
        nn_evals_a.extend(stats_g2["nn_evals_black"])
        nn_evals_b.extend(stats_g2["nn_evals_white"])

    stats = compute_elo_with_ci(wins_a, draws, losses_a)

    # Compute latency percentiles
    if latencies_a:
        stats["engine_a_perf"] = {
            "median_latency_ms": float(np.median(latencies_a)),
            "p95_latency_ms": float(np.percentile(latencies_a, 95)),
            "mean_nodes_per_move": float(np.mean(nodes_a)),
            "mean_nn_evals_per_move": float(np.mean(nn_evals_a)),
        }
    if latencies_b:
        stats["engine_b_perf"] = {
            "median_latency_ms": float(np.median(latencies_b)),
            "p95_latency_ms": float(np.percentile(latencies_b, 95)),
            "mean_nodes_per_move": float(np.mean(nodes_b)),
            "mean_nn_evals_per_move": float(np.mean(nn_evals_b)),
        }

    return stats
