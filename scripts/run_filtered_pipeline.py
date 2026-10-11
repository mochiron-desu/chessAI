#!/usr/bin/env python3
"""Quality-filtered dataset pipeline (Phase B blunder curation).

Streams games from Lichess PGN archive, preserves deterministic train/val/test split
boundaries, evaluates human moves with local Stockfish 16, filters human moves with
centipawn loss > max_cpl (default 25 cp), and writes compact binary records.
Guaranteed zero disk overflow and zero overwriting of existing baseline shards.
"""

import argparse
import io
import json
import os
import random
import sys
import time
import hashlib
import urllib.request
from typing import Dict, Any, Optional, Tuple, List
from collections import Counter
import zstandard as zstd
import chess
import chess.pgn
import chess.engine
import numpy as np

from chesslab.data.stream_parser import (
    RECORD_DTYPE,
    encode_position_to_record,
    filter_game,
)
from chesslab.data.pipeline import (
    assign_split_hash,
    sample_game_position,
    PipelineConfig,
)


def compute_move_cpl(
    sf_engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move: chess.Move,
    limit: chess.engine.Limit,
) -> Tuple[float, float, float]:
    """Computes centipawn loss of `move` played in `board` from the perspective of board.turn.

    Returns (cpl, best_score_cp, played_score_cp).
    Perspective is guaranteed identical for both evaluations (board.turn before move).
    """
    turn = board.turn

    # 1. Best move evaluation from player's perspective
    info_best = sf_engine.analyse(board, limit)
    best_pv = info_best.get("pv", [])
    best_move = best_pv[0] if best_pv else None
    best_score_cp = float(info_best["score"].pov(turn).score(mate_score=10000))

    # Fast path: human move is engine's top move -> CPL is strictly 0.0
    if best_move is not None and move == best_move:
        return 0.0, best_score_cp, best_score_cp

    # 2. Played move evaluation from player's perspective
    board.push(move)
    try:
        if board.is_checkmate():
            played_score_cp = 10000.0
        elif board.is_game_over():
            played_score_cp = 0.0
        else:
            info_played = sf_engine.analyse(board, limit)
            played_score_cp = float(info_played["score"].pov(turn).score(mate_score=10000))
    finally:
        board.pop()

    cpl = max(0.0, best_score_cp - played_score_cp)
    return cpl, best_score_cp, played_score_cp


def run_filtered_pipeline(
    source: str,
    output_path: str,
    sf_bin: str,
    max_cpl: float = 25.0,
    max_kept: int = 500,
    sf_nodes: int = 500,
    report_json: Optional[str] = None,
    min_elo: int = 1500,
    min_ply: int = 12,
    salt: str = "chesslab_2026",
    seed: int = 20261011,
) -> Dict[str, Any]:
    """Streams games and filters human training moves by centipawn loss."""
    random.seed(seed)
    np.random.seed(seed)

    # Protect against accidental overwriting of baseline shards
    protected = ["train.bin", "val.bin", "test.bin"]
    out_basename = os.path.basename(output_path)
    if out_basename in protected and os.path.exists(output_path):
        raise ValueError(f"Refusing to overwrite baseline shard: {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Initializing Stockfish from {sf_bin} (nodes={sf_nodes})...")
    sf_engine = chess.engine.SimpleEngine.popen_uci(sf_bin)
    sf_engine.configure({"Threads": 1, "Hash": 16})
    sf_limit = chess.engine.Limit(nodes=sf_nodes)

    print(f"Connecting to stream: {source}...")
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "ChessLab-Filter/1.0"})
        raw_stream = urllib.request.urlopen(req, timeout=30)
    else:
        raw_stream = open(source, "rb")

    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(raw_stream)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="ignore")

    writer = open(output_path, "wb")

    seen_hashes = set()
    total_scanned = 0
    filtered_out_prd = 0
    duplicates = 0
    non_train_skipped = 0
    evaluated_positions = 0
    kept_count = 0
    blunders_rejected = 0
    engine_failures = 0

    cpl_values: List[float] = []
    eval_latencies_ms: List[float] = []
    kept_plies: List[int] = []

    t0 = time.time()

    try:
        while kept_count < max_kept:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                print("Stream ended.")
                break

            total_scanned += 1

            if not filter_game(game.headers):
                filtered_out_prd += 1
                continue

            # Deduplication
            moves = list(game.mainline_moves())
            move_str = " ".join([m.uci() for m in moves])
            move_hash = hashlib.sha256(move_str.encode()).hexdigest()[:16]
            if move_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(move_hash)

            # Preserve split boundaries: only train split gets filtered
            split = assign_split_hash(move_hash, salt=salt)
            if split != "train":
                non_train_skipped += 1
                continue

            # Sample 1 representative position after ply 10
            sample = sample_game_position(game, min_ply=min_ply)
            if sample is None:
                filtered_out_prd += 1
                continue

            board_sample, move_sample, result_str = sample

            # Evaluate CPL with Stockfish
            t_eval0 = time.perf_counter()
            try:
                cpl, best_score, played_score = compute_move_cpl(
                    sf_engine, board_sample, move_sample, sf_limit
                )
            except Exception as e:
                engine_failures += 1
                print(f"Engine evaluation warning on position: {e}", file=sys.stderr)
                continue
            dt_ms = (time.perf_counter() - t_eval0) * 1000.0

            evaluated_positions += 1
            eval_latencies_ms.append(dt_ms)
            cpl_values.append(cpl)

            if cpl <= max_cpl:
                # Retain position: sound move
                record = encode_position_to_record(board_sample, move_sample, result_str)
                writer.write(record.tobytes())
                kept_count += 1
                kept_plies.append(len(board_sample.move_stack))
            else:
                blunders_rejected += 1

            if evaluated_positions % 50 == 0 or kept_count == max_kept:
                elapsed = time.time() - t0
                filter_pct = (blunders_rejected / evaluated_positions) * 100.0
                rate = evaluated_positions / max(0.1, elapsed)
                print(
                    f"Scanned: {total_scanned} | Evaluated: {evaluated_positions} | "
                    f"Kept: {kept_count}/{max_kept} | Blunders: {blunders_rejected} ({filter_pct:.1f}%) | "
                    f"Speed: {rate:.1f} pos/s",
                    flush=True
                )
    finally:
        writer.close()
        sf_engine.quit()

    total_time = time.time() - t0

    # Bucket CPL distribution
    cpl_buckets = {
        "0_cp (exact best)": sum(1 for c in cpl_values if c == 0.0),
        "1-10_cp": sum(1 for c in cpl_values if 0.0 < c <= 10.0),
        "11-25_cp": sum(1 for c in cpl_values if 10.0 < c <= 25.0),
        "26-50_cp": sum(1 for c in cpl_values if 25.0 < c <= 50.0),
        "51-100_cp": sum(1 for c in cpl_values if 50.0 < c <= 100.0),
        ">100_cp": sum(1 for c in cpl_values if c > 100.0),
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "source": source,
            "output_path": output_path,
            "stockfish_bin": sf_bin,
            "stockfish_nodes": sf_nodes,
            "max_cpl_threshold": max_cpl,
            "max_kept": max_kept,
            "salt": salt,
            "seed": seed,
        },
        "counts": {
            "total_games_scanned": total_scanned,
            "prd_filtered_out": filtered_out_prd,
            "duplicates_skipped": duplicates,
            "non_train_split_skipped": non_train_skipped,
            "positions_evaluated": evaluated_positions,
            "positions_kept": kept_count,
            "blunders_rejected": blunders_rejected,
            "blunder_filter_rate": blunders_rejected / max(1, evaluated_positions),
            "retention_rate": kept_count / max(1, evaluated_positions),
            "engine_failures": engine_failures,
        },
        "performance": {
            "wall_clock_seconds": total_time,
            "positions_evaluated_per_second": evaluated_positions / max(0.1, total_time),
            "median_eval_latency_ms": float(np.median(eval_latencies_ms)) if eval_latencies_ms else 0.0,
            "mean_eval_latency_ms": float(np.mean(eval_latencies_ms)) if eval_latencies_ms else 0.0,
            "p95_eval_latency_ms": float(np.percentile(eval_latencies_ms, 95)) if eval_latencies_ms else 0.0,
        },
        "cpl_distribution": cpl_buckets,
        "kept_plies_summary": {
            "min": int(np.min(kept_plies)) if kept_plies else 0,
            "max": int(np.max(kept_plies)) if kept_plies else 0,
            "mean": float(np.mean(kept_plies)) if kept_plies else 0.0,
        },
        "output_shard": {
            "path": output_path,
            "size_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
            "sha256": hashlib.sha256(open(output_path, "rb").read()).hexdigest() if os.path.exists(output_path) else "",
        },
    }

    if report_json:
        os.makedirs(os.path.dirname(os.path.abspath(report_json)), exist_ok=True)
        with open(report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_json}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Quality-filtered dataset pipeline")
    parser.add_argument("--source", type=str, default="https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst")
    parser.add_argument("--output", type=str, default="data/shards/train_filtered_pilot.bin")
    parser.add_argument("--sf-bin", type=str, default="tools/stockfish16/stockfish")
    parser.add_argument("--max-cpl", type=float, default=25.0)
    parser.add_argument("--max-kept", type=int, default=200)
    parser.add_argument("--sf-nodes", type=int, default=500)
    parser.add_argument("--report-json", type=str, default="data/filtered_pipeline_pilot.json")
    args = parser.parse_args()

    report = run_filtered_pipeline(
        source=args.source,
        output_path=args.output,
        sf_bin=args.sf_bin,
        max_cpl=args.max_cpl,
        max_kept=args.max_kept,
        sf_nodes=args.sf_nodes,
        report_json=args.report_json,
    )
    print("\nPilot Execution Summary:")
    print(f"  Positions Evaluated: {report['counts']['positions_evaluated']}")
    print(f"  Kept (<= {args.max_cpl} cp): {report['counts']['positions_kept']} ({report['counts']['retention_rate']*100:.1f}%)")
    print(f"  Blunders Rejected (> {args.max_cpl} cp): {report['counts']['blunders_rejected']} ({report['counts']['blunder_filter_rate']*100:.1f}%)")
    print(f"  Eval Latency (Median): {report['performance']['median_eval_latency_ms']:.2f} ms")
    print(f"  Engine Failures: {report['counts']['engine_failures']}")


if __name__ == "__main__":
    main()
