"""Comprehensive test suite for inference-time Alpha-Beta tree search harness."""

import pytest
import chess
import torch

from chesslab.models.resnet import ChessResNet
from chesslab.search.minimax import AlphaBetaSearcher, MATE_SCORE
from chesslab.inference.engine_uci import UCIEngine


@pytest.fixture
def dummy_model():
    """Instantiates a lightweight deterministic ChessResNet for fast testing."""
    torch.manual_seed(42)
    model = ChessResNet(num_blocks=2, channels=32)
    model.eval()
    return model


def test_depth_semantics(dummy_model):
    """Verify that search depth in plies corresponds exactly to tree expansion depth."""
    board = chess.Board()
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_pruning=False, use_tt=False)

    # Depth 1: expands root moves (20 moves)
    _, _, stats1 = searcher.search_root(board, depth=1)
    # 1 root node + 20 children = 21 nodes visited
    assert stats1["nodes"] == 21
    assert stats1["depth"] == 1

    # Depth 2: expands root moves + replies
    _, _, stats2 = searcher.search_root(board, depth=2)
    assert stats2["depth"] == 2
    assert stats2["nodes"] > stats1["nodes"]


def test_minimax_vs_alphabeta_equivalence(dummy_model):
    """Verify that unpruned Minimax and Alpha-Beta return identical root score while pruning reduces nodes."""
    board = chess.Board()
    # Play a few moves to create an asymmetric middlegame position
    for mv in ["e2e4", "c7c5", "g1f3", "d7d6"]:
        board.push_uci(mv)

    # Unpruned Minimax
    searcher_minimax = AlphaBetaSearcher(
        dummy_model, device=torch.device("cpu"), use_pruning=False, use_move_ordering=False, use_tt=False
    )
    move_mm, score_mm, stats_mm = searcher_minimax.search_root(board, depth=2)

    # Alpha-Beta with move ordering
    searcher_ab = AlphaBetaSearcher(
        dummy_model, device=torch.device("cpu"), use_pruning=True, use_move_ordering=True, use_tt=False
    )
    move_ab, score_ab, stats_ab = searcher_ab.search_root(board, depth=2)

    # Scores must match within floating point precision
    assert abs(score_mm - score_ab) < 1e-5, f"Minimax score {score_mm} != AlphaBeta score {score_ab}"
    # Pruned search must visit fewer nodes than unpruned search
    assert stats_ab["nodes"] < stats_mm["nodes"], f"AlphaBeta nodes ({stats_ab['nodes']}) >= Minimax nodes ({stats_mm['nodes']})"
    reduction_pct = (1.0 - stats_ab["nodes"] / stats_mm["nodes"]) * 100
    print(f"\nNode reduction on Sicilian position: {reduction_pct:.1f}% ({stats_ab['nodes']} vs {stats_mm['nodes']} nodes)")


def test_mate_in_one_detection(dummy_model):
    """Immediate checkmate is selected and scored properly (> 90.0)."""
    # Scholar's mate: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
    board = chess.Board()
    for m in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]:
        board.push_uci(m)

    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), max_depth=1)
    best_move, score, stats = searcher.search_root(board, depth=1)

    assert best_move == chess.Move.from_uci("h5f7")
    assert score >= 90.0, f"Expected mate score >= 90.0, got {score}"


def test_forced_mate_in_two(dummy_model):
    """Forced mate in 3 plies (mate-in-2 moves) is found at search depth >= 3 plies.
    
    Verified Lichess Puzzle 000Zo:
    Black to move: 1... Re1+ 2. Kf2 Rf1# (3 plies to mate).
    """
    fen = "4r3/1k6/pp3P2/1b5p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 35"
    board = chess.Board(fen)
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), max_depth=3)
    best_move, score, stats = searcher.search_root(board, depth=3)

    assert best_move == chess.Move.from_uci("e8e1")
    assert score >= 90.0, f"Expected mate score >= 90.0, got {score}"


def test_terminal_states_scoring(dummy_model):
    """Checkmate, stalemate, and 3-fold repetition return correct scores."""
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), max_depth=1)

    # Stalemate
    stalemate_board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    assert stalemate_board.is_stalemate()
    best_move, score, _ = searcher.search_root(stalemate_board, depth=1)
    assert best_move == chess.Move.null()
    assert score == 0.0

    # 3-fold repetition along path
    rep_board = chess.Board()
    rep_moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
    for m in rep_moves:
        rep_board.push_uci(m)
    assert rep_board.is_repetition(3) or rep_board.can_claim_draw()
    score_draw = searcher.evaluate_leaf(rep_board)
    assert score_draw == 0.0


def test_tactical_blunder_avoidance(dummy_model):
    """Search at depth 2 avoids playing a move that immediately drops a major piece."""
    # Position: White to move. White Queen on e4. Black Rook on e8.
    # If White does not move or shield the Queen, Black plays Rxe4 on ply 2.
    fen = "4r1k1/ppp2ppp/8/8/4Q3/8/PPP2PPP/4R1K1 w - - 0 1"
    board = chess.Board(fen)

    # Search at depth 2 (root move + Black response)
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), max_depth=2)
    best_move, score, stats = searcher.search_root(board, depth=2)

    # White must NOT leave queen on e4 or play a blunder like a2a3
    # Safe moves: Qxe8+ (winning rook for queen), Qd3, Qd4, Qc4, etc.
    # Testing that best_move is legal and not a blunder:
    board.push(best_move)
    # If Black can immediately capture the Queen with Rxe4, that would be a blunder
    if chess.Move.from_uci("e8e4") in board.legal_moves:
        pytest.fail(f"Search selected blundering move {best_move.uci()} allowing e8xe4 capture")


def test_transposition_table_agreement(dummy_model):
    """Verify that search with TT enabled matches search with TT disabled."""
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")

    searcher_no_tt = AlphaBetaSearcher(
        dummy_model, device=torch.device("cpu"), max_depth=2, use_pruning=True, use_tt=False
    )
    move_no_tt, score_no_tt, _ = searcher_no_tt.search_root(board, depth=2)

    searcher_with_tt = AlphaBetaSearcher(
        dummy_model, device=torch.device("cpu"), max_depth=2, use_pruning=True, use_tt=True
    )
    move_with_tt, score_with_tt, _ = searcher_with_tt.search_root(board, depth=2)

    assert move_no_tt == move_with_tt
    assert abs(score_no_tt - score_with_tt) < 1e-5


def test_legal_moves_guarantee(dummy_model):
    """Every root move selected by search across diverse positions is guaranteed legal."""
    positions = [
        chess.Board(),
        chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5"),
        chess.Board("r2q1rk1/1pp2ppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8"),
        chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"),
    ]
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), max_depth=2)
    for b in positions:
        mv, _, _ = searcher.search_root(b, depth=2)
        assert mv in b.legal_moves


def test_cpu_performance_metrics(dummy_model):
    """Verify that node visits, NN evaluations, elapsed time, and NPS are properly recorded."""
    board = chess.Board()
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), max_depth=2)
    _, _, stats = searcher.search_root(board, depth=2)

    assert stats["nodes"] > 0
    assert stats["nn_evals"] > 0
    assert stats["time_sec"] > 0
    assert stats["nps"] > 0
    assert stats["cutoffs"] >= 0


# ---------------------------------------------------------------------------
# Regression tests: TT correctness (halfmove clock, repetition, mate scores)
# ---------------------------------------------------------------------------
from chesslab.search.minimax import (  # noqa: E402
    get_position_key,
    score_to_tt,
    score_from_tt,
    MATE_PLY_PENALTY,
)


def test_tt_key_includes_halfmove_clock():
    """Identical layouts with different halfmove clocks must get different TT keys."""
    fen = "8/8/4k3/8/8/4K3/4R3/8 w - - {} 60"
    assert get_position_key(chess.Board(fen.format(0))) != get_position_key(chess.Board(fen.format(99)))
    assert get_position_key(chess.Board(fen.format(10))) == get_position_key(chess.Board(fen.format(10)))


def test_tt_halfmove_clock_isolation(dummy_model):
    """A warm TT built at clock 0 must not change the result at clock 99 (fifty-move edge)."""
    fen = "4k3/8/8/8/8/8/4R3/4K3 w - - {} 60"
    b0, b99 = chess.Board(fen.format(0)), chess.Board(fen.format(99))

    fresh = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=True)
    _, score_fresh, _ = fresh.search_root(b99, depth=3)

    warm = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=True)
    warm.search_root(b0, depth=3)
    _, score_warm, _ = warm.search_root(b99, depth=3)

    no_tt = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=False)
    _, score_no_tt, _ = no_tt.search_root(b99, depth=3)

    assert abs(score_fresh - score_warm) < 1e-5
    assert abs(score_no_tt - score_warm) < 1e-5


def test_fifty_move_draw_scored_below_root(dummy_model):
    """At clock 99 any non-pawn/non-capture move yields a claimable draw (0.0) at ply 1."""
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 99 60")
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_pruning=False, use_tt=False)
    # Mating-capable material but every quiet root move triggers the claim -> exactly 0.0 at depth 1
    _, score, _ = searcher.search_root(board, depth=1)
    assert score == 0.0


def test_tt_not_stored_for_repetition_influenced_nodes(dummy_model):
    """Subtree touching a repetition draw must not be cached (path-dependent score)."""
    board = chess.Board()
    for m in ["g1f3", "g8f6", "f3g1"]:
        board.push_uci(m)
    # Black to move; f6g8 returns to the start position (2nd occurrence) -> repetition draw
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=True)
    searcher.search_root(board, depth=2)
    assert get_position_key(board) not in searcher.tt

    # Same layout reached without repetition history is cached normally
    clean = chess.Board(board.fen())
    searcher2 = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=True)
    searcher2.search_root(clean, depth=2)
    assert get_position_key(clean) in searcher2.tt


def test_repetition_history_does_not_leak_through_shared_searcher(dummy_model):
    """Search with repetition history, then identical FEN without history, must equal a fresh search."""
    hist = chess.Board()
    for m in ["g1f3", "g8f6", "f3g1"]:
        hist.push_uci(m)
    clean = chess.Board(hist.fen())

    shared = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=True)
    shared.search_root(hist, depth=3)
    _, s_shared, _ = shared.search_root(clean, depth=3)

    fresh = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=True)
    _, s_fresh, _ = fresh.search_root(clean, depth=3)
    assert abs(s_shared - s_fresh) < 1e-5


def test_root_with_claimable_draw_still_returns_legal_move(dummy_model):
    """Root position with claimable repetition must still search and return a legal move."""
    board = chess.Board()
    for m in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(m)
    assert board.can_claim_draw()
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), max_depth=1)
    mv, _, stats = searcher.search_root(board, depth=1)
    assert mv in board.legal_moves
    assert stats["nodes"] > 1


def test_mate_score_tt_roundtrip_and_ply_shift():
    """Mate scores round-trip at same ply and shift correctly across plies; non-mate unchanged."""
    win = 100.0 - 3 * MATE_PLY_PENALTY  # mate at absolute ply 3
    stored = score_to_tt(win, 1)  # node at ply 1 => 2 plies to mate
    assert abs(score_from_tt(stored, 1) - win) < 1e-9
    # Same node reached at ply 2 => mate at absolute ply 4
    assert abs(score_from_tt(stored, 2) - (100.0 - 4 * MATE_PLY_PENALTY)) < 1e-9
    loss = -100.0 + 2 * MATE_PLY_PENALTY
    assert abs(score_from_tt(score_to_tt(loss, 1), 1) - loss) < 1e-9
    assert score_to_tt(0.25, 5) == 0.25 and score_from_tt(-0.4, 3) == -0.4


def test_tt_matches_no_tt_on_forced_mate(dummy_model):
    """TT-enabled search reports the same mate score/move as TT-free search."""
    board = chess.Board("4r3/1k6/pp3P2/1b5p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 35")
    a = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=False)
    b = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"), use_tt=True)
    ma, sa, _ = a.search_root(board, depth=3)
    mb, sb, _ = b.search_root(board, depth=3)
    assert ma == mb and abs(sa - sb) < 1e-5


@pytest.mark.parametrize("fen", [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r2q1rk1/1pp2ppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
])
def test_alphabeta_tt_equals_plain_minimax_root_score(dummy_model, fen):
    """Full Alpha-Beta+ordering+TT root score equals unpruned minimax at depth 3."""
    board = chess.Board(fen)
    mm = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"),
                           use_pruning=False, use_move_ordering=False, use_tt=False)
    ab = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"))
    _, s_mm, _ = mm.search_root(board, depth=3)
    _, s_ab, _ = ab.search_root(board, depth=3)
    assert abs(s_mm - s_ab) < 1e-4, f"{s_mm} vs {s_ab}"


def test_move_ordering_keeps_all_legal_moves(dummy_model):
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5")
    searcher = AlphaBetaSearcher(dummy_model, device=torch.device("cpu"))
    ordered = searcher.order_moves(board)
    assert sorted(m.uci() for m in ordered) == sorted(m.uci() for m in board.legal_moves)
