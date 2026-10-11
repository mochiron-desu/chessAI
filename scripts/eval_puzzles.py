#!/usr/bin/env python3
"""Tactical puzzle evaluation benchmark for ChessLab models.

Evaluates model on Lichess puzzle subset, measuring tactical solution accuracy
overall and stratified by theme and puzzle rating.
"""

import os
import csv
import argparse
from typing import Dict, Any, List
import chess
import torch

from chesslab.inference.engine_uci import UCIEngine
from chesslab.encode.board_planes import encode_board
from chesslab.inference.legal_mask import compute_legal_mask, mask_logits
from chesslab.encode.move_vocab import decode_move, encode_move


def evaluate_puzzles(
    engine: UCIEngine,
    csv_path: str = "data/benchmarks/puzzles_1k.csv",
    max_puzzles: int = 1000,
    search_depth: int = 0,
) -> Dict[str, Any]:
    total = 0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    by_rating = {"<1500": [0, 0], "1500-1800": [0, 0], "1800+": [0, 0]}
    by_theme = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if total >= max_puzzles:
                break

            fen = row["FEN"]
            moves_str = row["Moves"]
            rating = int(row.get("Rating", 1500))
            themes = row.get("Themes", "").split()

            moves = moves_str.split()
            if len(moves) < 2:
                continue

            # In Lichess puzzles: FEN is before opponent's move.
            # moves[0] is opponent's blunder, moves[1] is the puzzle solution.
            board = chess.Board(fen)
            opponent_move = chess.Move.from_uci(moves[0])
            solution_move = chess.Move.from_uci(moves[1])

            if opponent_move not in board.legal_moves:
                continue
            board.push(opponent_move)

            if solution_move not in board.legal_moves:
                continue

            if search_depth > 0:
                engine.board = board.copy()
                pred_move = engine.select_best_move(temperature=0.0)
                is_top1 = (pred_move == solution_move)
                is_top3 = is_top1
                is_top5 = is_top1
            else:
                # Searchless policy evaluation
                planes_np, rule_np = encode_board(board)
                mask_np = compute_legal_mask(board)

                planes_t = torch.from_numpy(planes_np).unsqueeze(0).to(engine.device)
                rule_t = torch.from_numpy(rule_np).unsqueeze(0).to(engine.device)
                mask_t = torch.from_numpy(mask_np).to(engine.device)

                with torch.no_grad():
                    pol_logits, _ = engine.model(planes_t, rule_t)
                    masked_pol = mask_logits(pol_logits, mask_t)
                    topk = torch.topk(masked_pol.squeeze(0), k=min(5, len(list(board.legal_moves))))

                pred_moves = [decode_move(idx.item(), board.turn) for idx in topk.indices]

                is_top1 = (pred_moves[0] == solution_move)
                is_top3 = solution_move in pred_moves[:3]
                is_top5 = solution_move in pred_moves[:5]

            total += 1
            if is_top1:
                correct_top1 += 1
            if is_top3:
                correct_top3 += 1
            if is_top5:
                correct_top5 += 1

            # Stratification by rating
            r_bucket = "<1500" if rating < 1500 else ("1500-1800" if rating <= 1800 else "1800+")
            by_rating[r_bucket][1] += 1
            if is_top1:
                by_rating[r_bucket][0] += 1

            # Stratification by theme
            for th in themes[:3]:
                if th not in by_theme:
                    by_theme[th] = [0, 0]
                by_theme[th][1] += 1
                if is_top1:
                    by_theme[th][0] += 1

    return {
        "search_depth": search_depth,
        "total_evaluated": total,
        "total": total,
        "top1_accuracy": correct_top1 / max(1, total),
        "top3_accuracy": correct_top3 / max(1, total),
        "top5_accuracy": correct_top5 / max(1, total),
        "by_rating": {
            k: (v[0] / max(1, v[1]), v[1]) for k, v in by_rating.items() if v[1] > 0
        },
        "by_theme": {
            k: (v[0] / max(1, v[1]), v[1])
            for k, v in sorted(by_theme.items(), key=lambda x: -x[1][1])[:10]
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on tactical puzzles")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/run_1m/final_ema_model.pt")
    parser.add_argument("--puzzles", type=str, default="data/benchmarks/puzzles_1k.csv")
    parser.add_argument("--max-puzzles", type=int, default=1000)
    parser.add_argument("--search-depth", type=int, default=0, help="Search depth (0 = searchless policy)")
    args = parser.parse_args()

    engine = UCIEngine(checkpoint_path=args.checkpoint, device="cpu", search_depth=args.search_depth)
    print(f"Loaded checkpoint: {args.checkpoint} (Search Depth: {args.search_depth})")
    print(f"Evaluating on {args.max_puzzles} tactical puzzles from {args.puzzles}...")

    results = evaluate_puzzles(
        engine, csv_path=args.puzzles, max_puzzles=args.max_puzzles, search_depth=args.search_depth
    )

    print("\n=== Tactical Puzzle Benchmark Results ===")
    print(f"Search Depth: {results['search_depth']}")
    print(f"Total Puzzles: {results['total_evaluated']}")
    print(f"Top-1 Solution Accuracy: {results['top1_accuracy']*100:.2f}%")
    if args.search_depth == 0:
        print(f"Top-3 Solution Accuracy: {results['top3_accuracy']*100:.2f}%")
        print(f"Top-5 Solution Accuracy: {results['top5_accuracy']*100:.2f}%")

    print("\nAccuracy by Rating Tier:")
    for tier, (acc, count) in results["by_rating"].items():
        print(f"  {tier} ({count} puzzles): {acc*100:.1f}%")

    print("\nAccuracy by Tactical Theme (Top 10):")
    for theme, (acc, count) in results["by_theme"].items():
        print(f"  {theme} ({count} puzzles): {acc*100:.1f}%")


if __name__ == "__main__":
    main()
