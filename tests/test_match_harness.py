import pytest
import chess
from chesslab.inference.engine_uci import UCIEngine
from chesslab.eval.match_harness import compute_elo_with_ci, run_match


def test_compute_elo_with_ci():
    # Equal score
    stats = compute_elo_with_ci(wins=50, draws=20, losses=50)
    assert stats["score"] == 0.5
    assert stats["delta_elo"] == 0.0
    assert stats["ci_95"] > 0.0

    # Winning score
    stats_win = compute_elo_with_ci(wins=75, draws=10, losses=15)
    assert stats_win["score"] > 0.5
    assert stats_win["delta_elo"] > 100.0


def test_paired_match_execution():
    engine_a = UCIEngine(checkpoint_path=None, device="cpu")
    engine_b = UCIEngine(checkpoint_path=None, device="cpu")

    # 1 opening pair: 1. e4 e5
    openings = [[chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]]

    stats = run_match(engine_a, engine_b, opening_pairs=openings, max_plies=20)
    assert stats["total_games"] == 2
    assert stats["wins"] + stats["draws"] + stats["losses"] == 2
