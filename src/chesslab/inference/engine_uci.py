"""Universal Chess Interface (UCI) protocol wrapper for ChessLab neural models.

Enables automated tournament play against Stockfish and other engines via cutechess-cli
or direct Python subprocessing.
"""

import sys
import os
from typing import Optional
import torch
import chess

from chesslab.models.resnet import ChessResNet
from chesslab.encode.board_planes import encode_board
from chesslab.inference.legal_mask import compute_legal_mask, sample_legal_move
from chesslab.encode.move_vocab import decode_move
from chesslab.search.minimax import AlphaBetaSearcher


class UCIEngine:
    """Wraps ChessResNet in the standard UCI engine protocol."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        search_depth: int = 0,
    ):
        self.device = torch.device(device)
        self.model = ChessResNet(num_blocks=6, channels=64).to(self.device)
        self.search_depth = search_depth

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()

        self.board = chess.Board()
        self.searcher = (
            AlphaBetaSearcher(self.model, device=self.device, max_depth=self.search_depth)
            if self.search_depth > 0
            else None
        )

    def select_best_move(self, temperature: float = 0.05) -> chess.Move:
        """Selects the best legal move for the current position."""
        if self.board.is_game_over() or len(list(self.board.legal_moves)) == 0:
            return chess.Move.null()

        # If search is enabled and temperature is deterministic, use alpha-beta search
        if self.search_depth > 0 and temperature < 1e-3 and self.searcher is not None:
            best_move, _, _ = self.searcher.search_root(self.board, depth=self.search_depth)
            if best_move in self.board.legal_moves:
                return best_move

        planes_np, rule_np = encode_board(self.board)
        mask_np = compute_legal_mask(self.board)

        planes_t = torch.from_numpy(planes_np).unsqueeze(0).to(self.device)
        rule_t = torch.from_numpy(rule_np).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).to(self.device)

        with torch.no_grad():
            pol_logits, _ = self.model(planes_t, rule_t)
            action_idx = sample_legal_move(
                pol_logits.squeeze(0),
                mask_t,
                temperature=temperature,
                greedy=(temperature < 1e-3)
            )

        return decode_move(action_idx, self.board.turn)

    def handle_line(self, line: str):
        line = line.strip()
        if not line:
            return

        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            print("id name ChessLab-ResNet")
            print("id author Antigravity Pair")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd == "ucinewgame":
            self.board.reset()

        elif cmd == "position":
            if len(tokens) > 1:
                idx = 1
                if tokens[idx] == "startpos":
                    self.board.reset()
                    idx += 1
                elif tokens[idx] == "fen":
                    fen_tokens = tokens[idx + 1: idx + 7]
                    self.board.set_fen(" ".join(fen_tokens))
                    idx += 7

                if idx < len(tokens) and tokens[idx] == "moves":
                    for mv_str in tokens[idx + 1:]:
                        self.board.push_uci(mv_str)

        elif cmd == "setoption":
            # e.g., setoption name SearchDepth value 2
            if len(tokens) >= 5 and tokens[1] == "name" and tokens[2].lower() == "searchdepth" and tokens[3] == "value":
                try:
                    self.search_depth = int(tokens[4])
                    self.searcher = (
                        AlphaBetaSearcher(self.model, device=self.device, max_depth=self.search_depth)
                        if self.search_depth > 0
                        else None
                    )
                except ValueError:
                    pass

        elif cmd == "go":
            # Optional go depth N
            depth = self.search_depth
            if len(tokens) >= 3 and tokens[1] == "depth":
                try:
                    depth = int(tokens[2])
                except ValueError:
                    pass

            if depth > 0 and self.searcher is not None:
                best_move, _, _ = self.searcher.search_root(self.board, depth=depth)
            else:
                best_move = self.select_best_move(temperature=0.0)

            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()

        elif cmd == "quit":
            sys.exit(0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UCI interface for ChessLab model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--search-depth", type=int, default=0, help="Alpha-Beta search depth (0 = searchless policy)")
    args = parser.parse_args()

    engine = UCIEngine(
        checkpoint_path=args.checkpoint or None,
        device=args.device,
        search_depth=args.search_depth
    )

    for line in sys.stdin:
        engine.handle_line(line)


if __name__ == "__main__":
    main()
