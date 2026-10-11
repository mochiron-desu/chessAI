"""Negamax Alpha-Beta search with policy move-ordering, WDL leaf evaluation, and TT caching.

Designed for rigorous experimental comparison: supports unpruned Minimax,
Alpha-Beta pruning, policy ordering ablation, and transposition table ablation.
"""

import time
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
import chess

from chesslab.encode.board_planes import encode_board
from chesslab.encode.move_vocab import encode_move
from chesslab.inference.legal_mask import compute_legal_mask, mask_logits

# Transposition Table Entry Flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# Checkmate base score (large enough to dominate any non-terminal leaf evaluation in [-1, 1])
MATE_SCORE = 100.0
MATE_PLY_PENALTY = 0.1


MATE_THRESHOLD = MATE_SCORE / 2.0


def score_to_tt(score: float, ply: int) -> float:
    """Converts a root-relative mate score to a node-relative score for TT storage."""
    if score > MATE_THRESHOLD:
        return score + ply * MATE_PLY_PENALTY
    if score < -MATE_THRESHOLD:
        return score - ply * MATE_PLY_PENALTY
    return score


def score_from_tt(score: float, ply: int) -> float:
    """Converts a node-relative TT mate score back to a root-relative score at `ply`."""
    if score > MATE_THRESHOLD:
        return score - ply * MATE_PLY_PENALTY
    if score < -MATE_THRESHOLD:
        return score + ply * MATE_PLY_PENALTY
    return score


def get_position_key(board: chess.Board) -> Tuple[Any, ...]:
    """Generates a fast, robust, immutable tuple representing board state for transposition caching.
    
    Includes piece positions, side to move, castling rights, en passant square, and the
    halfmove clock (fifty-move-rule state; also an input feature of the neural network).
    Does NOT include fullmove number. Repetition history is path-dependent and is handled
    separately: subtrees influenced by repetition draws are never stored in the TT.
    """
    return (
        board.turn,
        board.castling_rights,
        board.ep_square,
        board.halfmove_clock,
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    )


class AlphaBetaSearcher:
    """Rigorous Negamax tree search supporting Minimax, Alpha-Beta, Move Ordering, and TT caching."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        max_depth: int = 2,
        use_pruning: bool = True,
        use_move_ordering: bool = True,
        use_tt: bool = True,
    ):
        self.model = model
        self.device = device
        self.max_depth = max_depth
        self.use_pruning = use_pruning
        self.use_move_ordering = use_move_ordering
        self.use_tt = use_tt

        # Transposition table: key -> (depth, score, flag, best_move)
        self.tt: Dict[Any, Tuple[int, float, int, Optional[chess.Move]]] = {}
        # Node cache for network forward pass: key -> (pol_logits_np, leaf_value)
        self.eval_cache: Dict[Any, Tuple[np.ndarray, float]] = {}

        # Search statistics
        self.nodes_visited = 0
        self.nn_evals = 0
        self.cutoffs = 0
        self.tt_hits = 0
        # Incremented whenever a path-dependent repetition draw score is returned.
        self._rep_taint = 0

    def clear_caches(self):
        """Clears transposition table and neural network evaluation cache."""
        self.tt.clear()
        self.eval_cache.clear()

    def _eval_network(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Runs the neural model once on the board, returning (policy_logits, leaf_value).
        
        Leaf value is P(win) - P(loss) from the perspective of board.turn, in range [-1.0, 1.0].
        """
        key = get_position_key(board)
        if key in self.eval_cache:
            return self.eval_cache[key]

        self.nn_evals += 1
        planes_np, rule_np = encode_board(board)
        mask_np = compute_legal_mask(board)

        planes_t = torch.from_numpy(planes_np).unsqueeze(0).to(self.device)
        rule_t = torch.from_numpy(rule_np).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).to(self.device)

        with torch.no_grad():
            pol_logits, wdl_logits = self.model(planes_t, rule_t)
            masked_pol = mask_logits(pol_logits, mask_t).squeeze(0)

            # Softmax on WDL: 0=Win, 1=Draw, 2=Loss from side to move's perspective
            wdl_probs = F.softmax(wdl_logits, dim=-1).squeeze(0)
            p_win = wdl_probs[0].item()
            p_loss = wdl_probs[2].item()
            leaf_val = float(p_win - p_loss)

            pol_np = masked_pol.cpu().numpy()

        res = (pol_np, leaf_val)
        self.eval_cache[key] = res
        return res

    def evaluate_leaf(self, board: chess.Board) -> float:
        """Evaluates static leaf position from the perspective of board.turn."""
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0

        _, leaf_val = self._eval_network(board)
        return leaf_val

    def order_moves(
        self, board: chess.Board, tt_move: Optional[chess.Move] = None
    ) -> List[chess.Move]:
        """Orders legal moves in descending order of policy prior probability."""
        legal_moves = list(board.legal_moves)
        if len(legal_moves) <= 1:
            return legal_moves

        if not self.use_move_ordering:
            # Without move ordering, return raw legal moves (or tt_move first if present)
            if tt_move is not None and tt_move in legal_moves:
                legal_moves.remove(tt_move)
                return [tt_move] + legal_moves
            return legal_moves

        pol_logits_np, _ = self._eval_network(board)
        turn = board.turn

        scored_moves: List[Tuple[chess.Move, float]] = []
        for mv in legal_moves:
            idx = encode_move(mv, turn)
            score = float(pol_logits_np[idx])
            # High priority for transposition table best move
            if tt_move is not None and mv == tt_move:
                score += 1000.0
            scored_moves.append((mv, score))

        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [mv for mv, _ in scored_moves]

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int = 0
    ) -> Tuple[float, Optional[chess.Move]]:
        """Recursive Negamax search with Alpha-Beta pruning and path repetition awareness."""
        self.nodes_visited += 1

        # 1. Terminal states
        if board.is_checkmate():
            # Side to move has lost. Prefer slower losses and faster wins
            return -MATE_SCORE + ply * MATE_PLY_PENALTY, None

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0, None

        if ply > 0:
            # Position-dependent (halfmove clock is in the TT key): fifty-move claim
            if board.can_claim_fifty_moves():
                return 0.0, None
            # Path-dependent: repetition along current game/search path. Taint ancestors
            # so their results are not stored as position-independent TT entries.
            if board.is_repetition(2) or board.can_claim_threefold_repetition():
                self._rep_taint += 1
                return 0.0, None

        # 2. Leaf node (configured ply depth reached)
        if depth <= 0:
            return self.evaluate_leaf(board), None

        # 3. Transposition Table Lookup
        alpha_orig = alpha
        taint_start = self._rep_taint
        tt_key = get_position_key(board) if self.use_tt else None
        tt_move = None
        if self.use_tt and tt_key in self.tt:
            tt_depth, tt_stored, tt_flag, tt_best_mv = self.tt[tt_key]
            tt_move = tt_best_mv
            if tt_depth >= depth:
                tt_score = score_from_tt(tt_stored, ply)
                self.tt_hits += 1
                if tt_flag == TT_EXACT:
                    return tt_score, tt_best_mv
                elif tt_flag == TT_LOWERBOUND:
                    alpha = max(alpha, tt_score)
                elif tt_flag == TT_UPPERBOUND:
                    beta = min(beta, tt_score)
                if self.use_pruning and alpha >= beta:
                    return tt_score, tt_best_mv

        # 4. Move Ordering
        ordered_moves = self.order_moves(board, tt_move=tt_move)
        if not ordered_moves:
            return 0.0, None

        best_score = -float("inf")
        best_move = ordered_moves[0]

        # 5. Search children
        for move in ordered_moves:
            board.push(move)
            if self.use_pruning:
                child_score, _ = self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Unpruned Minimax
                child_score, _ = self.negamax(board, depth - 1, -float("inf"), float("inf"), ply + 1)
            score = -child_score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if self.use_pruning:
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    self.cutoffs += 1
                    break

        # 6. Store in TT only if no repetition draw influenced this subtree
        if self.use_tt and self._rep_taint == taint_start:
            if best_score <= alpha_orig:
                flag = TT_UPPERBOUND
            elif best_score >= beta:
                flag = TT_LOWERBOUND
            else:
                flag = TT_EXACT
            self.tt[tt_key] = (depth, score_to_tt(best_score, ply), flag, best_move)

        return best_score, best_move

    def search_root(
        self, board: chess.Board, depth: Optional[int] = None
    ) -> Tuple[chess.Move, float, Dict[str, Any]]:
        """Finds the best move from the root position using Negamax search.
        
        Args:
            board: chess.Board position to search.
            depth: Number of plies to search (1, 2, 3, ...). If None, uses self.max_depth.
            
        Returns:
            best_move: Selected chess.Move.
            best_score: Evaluation from perspective of root player.
            stats: Dictionary with search statistics (depth, nodes, nn_evals, cutoffs, tt_hits, time_sec, nps).
        """
        target_depth = depth if depth is not None else self.max_depth
        if board.is_game_over() or not list(board.legal_moves):
            return chess.Move.null(), 0.0, {
                "depth": target_depth, "nodes": 0, "nn_evals": 0, "cutoffs": 0, "tt_hits": 0, "time_sec": 0.0, "nps": 0.0
            }

        self.nodes_visited = 0
        self.nn_evals = 0
        self.cutoffs = 0
        self.tt_hits = 0
        t0 = time.perf_counter()

        best_score, best_move = self.negamax(
            board, depth=target_depth, alpha=-float("inf"), beta=float("inf"), ply=0
        )

        elapsed = max(1e-6, time.perf_counter() - t0)
        nps = self.nodes_visited / elapsed

        stats = {
            "depth": target_depth,
            "score": best_score,
            "nodes": self.nodes_visited,
            "nn_evals": self.nn_evals,
            "cutoffs": self.cutoffs,
            "tt_hits": self.tt_hits,
            "time_sec": elapsed,
            "nps": nps,
        }

        if best_move is None or best_move not in board.legal_moves:
            best_move = list(board.legal_moves)[0]

        return best_move, best_score, stats
