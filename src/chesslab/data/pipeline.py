"""Streaming PGN ingestion pipeline for ChessLab.

Streams compressed PGN data (.pgn.zst) from remote HTTP or local files, applies
PRD quality filters, deduplicates move histories, extracts representative positions,
and writes compact binary shards.
"""

import io
import os
import sys
import time
import json
import random
import hashlib
import urllib.request
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import zstandard as zstd
import chess
import chess.pgn
import numpy as np

from chesslab.data.stream_parser import (
    RECORD_DTYPE,
    encode_position_to_record,
    filter_game,
)


@dataclass
class PipelineConfig:
    source: str = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
    output_dir: str = "data"
    max_valid_games: int = 10000
    report_every: int = 1000
    salt: str = "chesslab_2026"
    min_elo: int = 1500
    min_ply: int = 12
    train_pct: float = 0.95
    val_pct: float = 0.025
    test_pct: float = 0.025


def assign_split_hash(game_id: str, salt: str = "chesslab_2026") -> str:
    """Deterministically assigns a game to train (95%), val (2.5%), or test (2.5%)."""
    h = int(hashlib.sha256(f"{game_id}_{salt}".encode()).hexdigest(), 16) % 10000
    if h < 9500:
        return "train"
    elif h < 9750:
        return "val"
    else:
        return "test"


def sample_game_position(game: chess.pgn.Game, min_ply: int = 12) -> Optional[Tuple[chess.Board, chess.Move, str]]:
    """Samples 1 random representative position after ply 10 from a valid game."""
    board = game.board()
    moves = list(game.mainline_moves())
    total_plies = len(moves)

    if total_plies < min_ply:
        return None

    # Sample a ply between ply 10 and (total_plies - 2)
    sample_ply = random.randint(10, total_plies - 2)

    for ply_idx, move in enumerate(moves):
        if ply_idx == sample_ply:
            return board.copy(), move, game.headers.get("Result", "*")
        board.push(move)

    return None


def run_ingestion_pipeline(
    cfg: PipelineConfig,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """Runs the streaming PGN pipeline and writes compact binary shards."""
    shards_dir = os.path.join(cfg.output_dir, "shards")
    manifests_dir = os.path.join(cfg.output_dir, "manifests")
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    writers = {
        "train": open(os.path.join(shards_dir, "train.bin"), "wb"),
        "val": open(os.path.join(shards_dir, "val.bin"), "wb"),
        "test": open(os.path.join(shards_dir, "test.bin"), "wb"),
    }

    counts = {"train": 0, "val": 0, "test": 0}
    seen_hashes = set()
    total_scanned = 0
    filtered_out = 0
    duplicates = 0
    valid_count = 0

    t0 = time.time()
    print(f"Connecting to source: {cfg.source}")
    print(f"Target valid games: {cfg.max_valid_games:,}")

    # Open stream
    if cfg.source.startswith("http://") or cfg.source.startswith("https://"):
        req = urllib.request.Request(
            cfg.source,
            headers={"User-Agent": "ChessLab-Research-Bot/1.0"}
        )
        raw_stream = urllib.request.urlopen(req)
    else:
        raw_stream = open(cfg.source, "rb")

    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(raw_stream)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="ignore")

    try:
        while valid_count < cfg.max_valid_games:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                print("Stream completed or closed.")
                break

            total_scanned += 1

            if not filter_game(game.headers, min_elo=cfg.min_elo):
                filtered_out += 1
                continue

            # Deduplication on moves string
            move_str = " ".join([m.uci() for m in game.mainline_moves()])
            move_hash = hashlib.sha256(move_str.encode()).hexdigest()[:16]
            if move_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(move_hash)

            # Sample 1 representative position
            sample = sample_game_position(game, min_ply=cfg.min_ply)
            if sample is None:
                filtered_out += 1
                continue

            board_sample, move_sample, result_str = sample
            try:
                record = encode_position_to_record(board_sample, move_sample, result_str)
            except Exception:
                filtered_out += 1
                continue

            split = assign_split_hash(move_hash, salt=cfg.salt)
            writers[split].write(record.tobytes())
            counts[split] += 1
            valid_count += 1

            if valid_count % cfg.report_every == 0 or valid_count == cfg.max_valid_games:
                elapsed = time.time() - t0
                speed = valid_count / max(0.1, elapsed)
                stats = {
                    "valid_count": valid_count,
                    "target": cfg.max_valid_games,
                    "scanned": total_scanned,
                    "filtered": filtered_out,
                    "duplicates": duplicates,
                    "counts": counts,
                    "speed_valid_sec": speed,
                    "elapsed_sec": elapsed,
                }
                if progress_callback:
                    progress_callback(stats)
                else:
                    print(
                        f"Valid: {valid_count:,}/{cfg.max_valid_games:,} "
                        f"({counts['train']:,} train, {counts['val']:,} val, {counts['test']:,} test) | "
                        f"Scanned: {total_scanned:,} | "
                        f"Speed: {speed:.1f} valid/s | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

    finally:
        for w in writers.values():
            w.close()
        text_stream.close()
        raw_stream.close()

    total_time = time.time() - t0

    # Compute SHA-256 for each generated shard
    shard_hashes = {}
    shard_sizes = {}
    for split_name in ["train", "val", "test"]:
        path = os.path.join(shards_dir, f"{split_name}.bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            shard_hashes[split_name] = h
            shard_sizes[split_name] = os.path.getsize(path)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": asdict(cfg),
        "total_scanned": total_scanned,
        "filtered_out": filtered_out,
        "duplicates": duplicates,
        "valid_kept": valid_count,
        "counts_by_split": counts,
        "shard_sizes_bytes": shard_sizes,
        "shard_sha256": shard_hashes,
        "total_time_seconds": round(total_time, 2),
        "valid_games_per_second": round(valid_count / max(1.0, total_time), 2),
    }

    manifest_path = os.path.join(manifests_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
