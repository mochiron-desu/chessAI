#!/usr/bin/env python3
"""Tournament and match runner for evaluating trained Chess models."""

import os
import argparse
import json
import chess

from chesslab.inference.engine_uci import UCIEngine
from chesslab.eval.match_harness import run_match


STANDARD_OPENINGS = [
    [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")],
    [chess.Move.from_uci("d2d4"), chess.Move.from_uci("d7d5")],
    [chess.Move.from_uci("c2c4"), chess.Move.from_uci("e7e6")],
    [chess.Move.from_uci("e2e4"), chess.Move.from_uci("c7c5")],
    [chess.Move.from_uci("g1f3"), chess.Move.from_uci("g8f6")],
]


def main():
    parser = argparse.ArgumentParser(description="Run head-to-head chess engine match")
    parser.add_argument("--engine-a-ckpt", type=str, default="", help="Checkpoint path for Engine A")
    parser.add_argument("--engine-b-ckpt", type=str, default="", help="Checkpoint path for Engine B (empty = random init)")
    parser.add_argument("--pairs", type=int, default=5, help="Number of opening pairs (each pair = 2 games with swapped colors)")
    parser.add_argument("--max-plies", type=int, default=100, help="Maximum plies per game")
    parser.add_argument("--output", type=str, default="data/tournament_results.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Initializing Engine A: {args.engine_a_ckpt or 'random weights'}")
    engine_a = UCIEngine(checkpoint_path=args.engine_a_ckpt or None, device="cpu")

    print(f"Initializing Engine B: {args.engine_b_ckpt or 'random weights'}")
    engine_b = UCIEngine(checkpoint_path=args.engine_b_ckpt or None, device="cpu")

    # Select opening pairs
    openings = (STANDARD_OPENINGS * (args.pairs // len(STANDARD_OPENINGS) + 1))[:args.pairs]

    print(f"Running match: {len(openings)} opening pairs = {len(openings)*2} games total...")
    stats = run_match(engine_a, engine_b, opening_pairs=openings, max_plies=args.max_plies)

    print("\n=== Match Results ===")
    print(f"Total Games: {stats['total_games']}")
    print(f"Engine A Wins: {stats['wins']} | Draws: {stats['draws']} | Engine B Wins: {stats['losses']}")
    print(f"Score: {stats['score'] * 100:.1f}%")
    print(f"Draw Rate: {stats['draw_rate'] * 100:.1f}%")
    print(f"Delta Elo: {stats['delta_elo']:+.1f} ± {stats['ci_95']:.1f} (95% CI)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
