"""Streaming parser for compressed PGN archives (.pgn.zst).

Streams games directly from an HTTP URL or local file, filters according to PRD criteria,
subsamples positions, and writes compact binary shards.
"""

import io
import os
import random
import hashlib
from typing import Generator, Dict, Any, Optional, Tuple, BinaryIO
from pathlib import Path
import zstandard as zstd
import chess
import chess.pgn
import numpy as np

from chesslab.encode.board_planes import PIECE_TYPES
from chesslab.encode.move_vocab import VOCAB_SIZE, MASK_NUM_BYTES, encode_move
from chesslab.inference.legal_mask import compute_legal_mask


RECORD_DTYPE = np.dtype([
    ("bitboards", np.uint64, (12,)),        # 96 bytes
    ("rule_state", np.float16, (7,)),       # 14 bytes
    ("move_idx", np.uint16),                # 2 bytes
    ("wdl", np.uint8),                      # 1 byte (0=win, 1=draw, 2=loss)
    ("legal_mask", np.uint8, (MASK_NUM_BYTES,)),  # packed bitset
])


def encode_position_to_record(
    board: chess.Board,
    move: chess.Move,
    game_result: str
) -> np.ndarray:
    """Encodes a single board state, played move, and outcome into a compact binary record."""
    turn = board.turn
    record = np.zeros(1, dtype=RECORD_DTYPE)[0]

    # 1. 12 Bitboards (perspective flipped)
    for p_idx, pt in enumerate(PIECE_TYPES):
        if turn == chess.WHITE:
            f_mask = board.pieces_mask(pt, chess.WHITE)
            o_mask = board.pieces_mask(pt, chess.BLACK)
        else:
            f_mask = chess.flip_vertical(board.pieces_mask(pt, chess.BLACK))
            o_mask = chess.flip_vertical(board.pieces_mask(pt, chess.WHITE))
        record["bitboards"][p_idx] = np.uint64(f_mask)
        record["bitboards"][p_idx + 6] = np.uint64(o_mask)

    # 2. Rule state (perspective flipped)
    f_k_castle = 1.0 if board.has_kingside_castling_rights(turn) else 0.0
    f_q_castle = 1.0 if board.has_queenside_castling_rights(turn) else 0.0
    o_k_castle = 1.0 if board.has_kingside_castling_rights(not turn) else 0.0
    o_q_castle = 1.0 if board.has_queenside_castling_rights(not turn) else 0.0

    ep_sq = board.ep_square
    if ep_sq is not None:
        view_ep_sq = ep_sq if turn == chess.WHITE else (ep_sq ^ 56)
        ep_avail = 1.0
        ep_file = float(view_ep_sq % 8) / 7.0
    else:
        ep_avail = 0.0
        ep_file = 0.0

    halfmove_norm = min(board.halfmove_clock, 100) / 100.0

    record["rule_state"][:] = [
        f_k_castle, f_q_castle, o_k_castle, o_q_castle,
        ep_avail, ep_file, halfmove_norm
    ]

    # 3. Move index
    record["move_idx"] = encode_move(move, turn)

    # 4. Result (from perspective of player who played the move)
    # 0 = Win, 1 = Draw, 2 = Loss
    if game_result == "1/2-1/2":
        wdl = 1
    elif (game_result == "1-0" and turn == chess.WHITE) or (game_result == "0-1" and turn == chess.BLACK):
        wdl = 0
    else:
        wdl = 2
    record["wdl"] = wdl

    # 5. Legal mask bitset
    mask_bool = compute_legal_mask(board)
    record["legal_mask"][:] = np.packbits(mask_bool)

    return record


def filter_game(headers: chess.pgn.Headers, min_elo: int = 1500) -> bool:
    """Evaluates if a game meets the PRD quality filters."""
    # Standard chess only
    variant = headers.get("Variant", "Standard")
    if variant != "Standard":
        return False

    # Event must be Blitz or Rapid
    event = headers.get("Event", "").lower()
    if not ("blitz" in event or "rapid" in event):
        return False

    # Minimum Elo threshold for both players
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
        if white_elo < min_elo or black_elo < min_elo:
            return False
    except ValueError:
        return False

    # Termination: normal or time forfeit
    termination = headers.get("Termination", "Normal")
    if termination not in ["Normal", "Time forfeit"]:
        return False

    # Valid result
    result = headers.get("Result", "*")
    if result not in ["1-0", "0-1", "1/2-1/2"]:
        return False

    return True


def decompress_stream(raw_stream: BinaryIO, chunk_size: int = 1024 * 1024) -> Generator[str, None, None]:
    """Streams decompressed text lines from a zstandard-compressed byte stream."""
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(raw_stream) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
        yield from text_stream
