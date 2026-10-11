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


class StockfishEngine:
    """External UCI engine (Stockfish) with a fixed node budget, same interface as UCIEngine.

    The node limit is sent as `go nodes N` to Stockfish only. `last_nodes` holds the actual
    node count Stockfish reported for the most recent move (it may slightly exceed N).
    """

    def __init__(self, binary_path: str, nodes: int, hash_mb: int = 16, threads: int = 1):
        import chess.engine
        self._chess_engine = chess.engine
        self.binary_path = binary_path
        self.nodes = nodes
        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.board = chess.Board()
        self.searcher = None
        self.last_nodes = 0
        self._game_token = object()
        self.version = self.engine.id.get("name", "unknown")

    def new_game(self):
        self._game_token = object()  # python-chess sends ucinewgame when token changes

    def select_best_move(self, temperature: float = 0.0) -> chess.Move:
        if self.board.is_game_over():
            return chess.Move.null()
        res = self.engine.play(
            self.board,
            self._chess_engine.Limit(nodes=self.nodes),
            info=self._chess_engine.INFO_BASIC,
            game=self._game_token,
        )
        self.last_nodes = int(res.info.get("nodes", 0))
        return res.move if res.move is not None else chess.Move.null()

    def close(self):
        self.engine.quit()


def play_game(
    white_engine: UCIEngine,
    black_engine: UCIEngine,
    opening_moves: Optional[List[chess.Move]] = None,
    max_plies: int = 150,
    perf_stats: Optional[Dict[str, Any]] = None,
    record: Optional[Dict[str, Any]] = None,
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

    for e in (white_engine, black_engine):
        if hasattr(e, "new_game"):
            e.new_game()

    def _finish(result: str, reason: str, illegal_by: Optional[str] = None) -> str:
        if record is not None:
            record.update({"result": result, "termination": reason, "plies": len(board.move_stack),
                           "illegal_by": illegal_by})
        return result

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
            elif hasattr(curr_engine, "last_nodes"):
                perf_stats[f"nodes_{engine_key}"].append(curr_engine.last_nodes)
                perf_stats[f"nn_evals_{engine_key}"].append(0)
            else:
                perf_stats[f"nodes_{engine_key}"].append(1)
                perf_stats[f"nn_evals_{engine_key}"].append(1)

        if move == chess.Move.null() or move not in board.legal_moves:
            # Forfeit if illegal or null
            res = "0-1" if turn == chess.WHITE else "1-0"
            return _finish(res, "illegal_move", engine_key)

        board.push(move)
        white_engine.board.push(move)
        black_engine.board.push(move)

    if board.is_checkmate():
        return _finish("0-1" if board.turn == chess.WHITE else "1-0", "checkmate")
    if board.is_game_over():
        return _finish("1/2-1/2", "draw_rule:" + str(board.outcome().termination.name))
    return _finish("1/2-1/2", "max_plies_adjudicated_draw")


def compute_paired_elo(pair_scores: List[float], z: float = 1.96) -> Dict[str, Any]:
    """Elo estimate with CI from per-pair scores (each in {0,.25,.5,.75,1}) via normal approx.

    Pairs (same opening, colors swapped) are the sampling unit. Score bounds are clamped away
    from 0/1 by 1/(2*games) so extreme results give a finite but flagged (saturated) interval.
    This is a rough interval, not BayesElo; with few pairs it is unreliable.
    """
    n = len(pair_scores)
    if n == 0:
        return {}
    mean = sum(pair_scores) / n
    var = sum((x - mean) ** 2 for x in pair_scores) / (n - 1) if n > 1 else 0.25
    se = math.sqrt(var / n)
    eps = 1.0 / (4.0 * n)

    def elo(s: float) -> float:
        s = min(1.0 - eps, max(eps, s))
        return -400.0 * math.log10(1.0 / s - 1.0)

    return {
        "pairs": n,
        "score": mean,
        "elo": elo(mean),
        "elo_lo": elo(mean - z * se),
        "elo_hi": elo(mean + z * se),
        "saturated": mean <= 0.0 or mean >= 1.0,
        "method": "paired normal approx, clamped; not BayesElo",
    }


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
