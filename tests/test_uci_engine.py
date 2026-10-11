import pytest
import chess
from chesslab.inference.engine_uci import UCIEngine


def test_uci_initialization_and_move_selection():
    """Gate 6: Verifies that the UCI engine generates legal moves on arbitrary positions."""
    engine = UCIEngine(checkpoint_path=None, device="cpu")

    # Initial position
    best_move = engine.select_best_move(temperature=0.0)
    assert best_move in engine.board.legal_moves

    # Push moves
    engine.board.push_san("e4")
    best_move_b = engine.select_best_move(temperature=0.0)
    assert best_move_b in engine.board.legal_moves


def test_simulated_game_playout():
    """Verify that the engine can complete a simulated 20-ply game without throwing exceptions."""
    engine = UCIEngine(checkpoint_path=None, device="cpu")

    for ply in range(20):
        if engine.board.is_game_over():
            break
        mv = engine.select_best_move(temperature=0.5)
        assert mv in engine.board.legal_moves
        engine.board.push(mv)

    assert len(engine.board.move_stack) > 0
