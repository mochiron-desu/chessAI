#!/usr/bin/env python3
"""Stockfish calibration ladder: one ChessLab variant vs Stockfish at fixed node budgets.

Paired games: each opening is played twice with colors swapped. Openings are `--opening-plies`
random legal plies drawn with a seeded RNG (identical across all configurations for a given seed).
Stockfish gets `go nodes N`; ChessLab uses ply-depth search (0 = searchless policy).
Results are written to one JSON per (depth, nodes) under --out-dir; existing files are never overwritten.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import chess
import numpy as np
import resource
import torch

from chesslab.inference.engine_uci import UCIEngine
from chesslab.eval.match_harness import StockfishEngine, play_game, compute_paired_elo, compute_elo_with_ci


def make_openings(n, plies, seed):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        b = chess.Board()
        moves = []
        for _ in range(plies):
            m = rng.choice(list(b.legal_moves))
            b.push(m)
            moves.append(m)
        key = b.board_fen()
        if b.is_game_over() or key in seen:
            continue
        seen.add(key)
        out.append(moves)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf-bin", required=True)
    ap.add_argument("--ckpt", default="checkpoints/run_1m/final_ema_model.pt")
    ap.add_argument("--depth", type=int, required=True, help="ChessLab ply depth (0 = searchless)")
    ap.add_argument("--nodes", type=int, nargs="+", default=[200])
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--out-dir", default="data/calibration")
    ap.add_argument("--tag", default="pilot")
    a = ap.parse_args()

    torch.set_num_threads(a.torch_threads)
    os.makedirs(a.out_dir, exist_ok=True)
    openings = make_openings(a.pairs, a.opening_plies, a.seed)
    cl = UCIEngine(checkpoint_path=a.ckpt, device="cpu", search_depth=a.depth)

    for nodes in a.nodes:
        path = os.path.join(a.out_dir, f"{a.tag}_d{a.depth}_sf{nodes}.json")
        if os.path.exists(path):
            print(f"SKIP existing {path}", file=sys.stderr)
            continue
        sf = StockfishEngine(a.sf_bin, nodes=nodes)
        games, pair_scores = [], []
        lat, cl_nodes, cl_nn, sf_nodes = [], [], [], []
        t_start, cpu_start = time.time(), time.process_time()
        for pi, op in enumerate(openings):
            ps = 0.0
            for cl_white in (True, False):
                perf = {k: [] for k in ("latency_white", "latency_black", "nodes_white", "nodes_black",
                                        "nn_evals_white", "nn_evals_black")}
                rec = {}
                w, b = (cl, sf) if cl_white else (sf, cl)
                res = play_game(w, b, opening_moves=op, max_plies=a.max_plies, perf_stats=perf, record=rec)
                cl_c, sf_c = ("white", "black") if cl_white else ("black", "white")
                cl_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
                if not cl_white:
                    cl_score = 1.0 - cl_score
                ps += cl_score
                lat += perf[f"latency_{cl_c}"]
                cl_nodes += perf[f"nodes_{cl_c}"]
                cl_nn += perf[f"nn_evals_{cl_c}"]
                sf_nodes += perf[f"nodes_{sf_c}"]
                games.append({"pair": pi, "opening": [m.uci() for m in op], "chesslab_color": cl_c,
                              "result": res, "chesslab_score": cl_score, **rec})
            pair_scores.append(ps / 2.0)
            print(f"d{a.depth} sf{nodes} pair {pi+1}/{len(openings)} score={ps/2:.2f}", flush=True)
        sf.close()

        w = sum(g["chesslab_score"] == 1.0 for g in games)
        d = sum(g["chesslab_score"] == 0.5 for g in games)
        l = sum(g["chesslab_score"] == 0.0 for g in games)
        by_color = {c: dict(Counter(g["chesslab_score"] for g in games if g["chesslab_color"] == c))
                    for c in ("white", "black")}
        illegal = [g for g in games if g["termination"] == "illegal_move"]
        out = {
            "config": {"chesslab_depth": a.depth, "stockfish_nodes": nodes, "pairs": a.pairs,
                       "max_plies": a.max_plies, "opening_plies": a.opening_plies, "seed": a.seed,
                       "checkpoint": a.ckpt, "stockfish_bin": a.sf_bin, "stockfish_version": sf.version,
                       "stockfish_options": {"Threads": 1, "Hash": 16, "UCI_LimitStrength": False,
                                             "Skill Level": 20, "limit": f"nodes={nodes}"},
                       "adjudication": "draw at max_plies; otherwise standard chess game-over rules",
                       "torch_threads": a.torch_threads, "torch": torch.__version__},
            "games_played": len(games), "wins": w, "draws": d, "losses": l,
            "score": (w + 0.5 * d) / len(games),
            "elo_paired": compute_paired_elo(pair_scores),
            "elo_game_level_legacy": compute_elo_with_ci(w, d, l),
            "score_by_color_counts": by_color,
            "pair_scores": pair_scores,
            "terminations": dict(Counter(g["termination"] for g in games)),
            "illegal_moves": [{"pair": g["pair"], "by": g["illegal_by"], "color": g["chesslab_color"]} for g in illegal],
            "chesslab_perf": {"median_latency_ms": float(np.median(lat)), "p95_latency_ms": float(np.percentile(lat, 95)),
                              "mean_nodes_per_move": float(np.mean(cl_nodes)), "mean_nn_evals_per_move": float(np.mean(cl_nn)),
                              "moves": len(lat)},
            "stockfish_actual_nodes": {"mean": float(np.mean(sf_nodes)), "min": int(np.min(sf_nodes)),
                                       "max": int(np.max(sf_nodes)), "moves": len(sf_nodes)},
            "resources": {"wall_sec": time.time() - t_start, "cpu_sec": time.process_time() - cpu_start,
                          "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0},
            "games": games,
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"WROTE {path}: W{w} D{d} L{l} score={out['score']:.3f}", flush=True)


if __name__ == "__main__":
    main()
