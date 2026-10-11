import pytest
import io
import os
import tempfile
import zstandard as zstd
import chess
import chess.pgn

from chesslab.data.pipeline import (
    PipelineConfig,
    assign_split_hash,
    sample_game_position,
    run_ingestion_pipeline,
)
from chesslab.data.stream_parser import filter_game


def test_assign_split_hash_distribution():
    splits = [assign_split_hash(f"game_{i}") for i in range(1000)]
    train_c = splits.count("train")
    val_c = splits.count("val")
    test_c = splits.count("test")

    assert 920 <= train_c <= 970
    assert 10 <= val_c <= 45
    assert 10 <= test_c <= 45


def test_filter_game_rules():
    # Valid game
    h_valid = chess.pgn.Headers({
        "Variant": "Standard",
        "Event": "Rated Blitz game",
        "WhiteElo": "1800",
        "BlackElo": "1850",
        "Termination": "Normal",
        "Result": "1-0",
    })
    assert filter_game(h_valid) is True

    # Variant Chess960 -> False
    h_960 = h_valid.copy()
    h_960["Variant"] = "Chess960"
    assert filter_game(h_960) is False

    # Bullet event -> False
    h_bullet = h_valid.copy()
    h_bullet["Event"] = "Rated Bullet game"
    assert filter_game(h_bullet) is False

    # Low rating -> False (default min_elo=1500)
    h_low = h_valid.copy()
    h_low["WhiteElo"] = "1400"
    assert filter_game(h_low) is False
    # Custom min_elo allows lower threshold if specified
    assert filter_game(h_low, min_elo=1300) is True
    # Custom min_elo rejects if higher threshold specified
    assert filter_game(h_valid, min_elo=1900) is False


def test_sample_game_position():
    pgn_text = """[Event "Rated Blitz game"]
[Site "https://lichess.org/test"]
[Date "2024.01.01"]
[Round "-"]
[White "W"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None

    sample = sample_game_position(game, min_ply=10)
    assert sample is not None
    board, move, result = sample
    assert isinstance(board, chess.Board)
    assert isinstance(move, chess.Move)
    assert move in board.legal_moves
    assert result == "1-0"


def test_local_pipeline_execution():
    """Verify end-to-end pipeline on a small synthetic .pgn.zst archive."""
    game_lines = [
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 1-0",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 1-0",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 1-0",
        "1. c4 e5 2. Nc3 Nf6 3. g3 d5 4. cxd5 Nxd5 5. Bg2 Nb6 6. Nf3 Nc6 7. O-O Be7 8. d3 O-O 1-0",
        "1. Nf3 d5 2. g3 Nf6 3. Bg2 e6 4. O-O Be7 5. d3 O-O 6. Nbd2 c5 7. e4 Nc6 8. Re1 b5 1-0",
    ]
    pgn_sample = ""
    for moves in game_lines:
        pgn_sample += f"""[Event "Rated Blitz game"]
[Variant "Standard"]
[WhiteElo "1800"]
[BlackElo "1900"]
[Result "1-0"]

{moves}

"""

    # Compress to temporary .pgn.zst
    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(pgn_sample.encode("utf-8"))

    with tempfile.TemporaryDirectory() as tmpdir:
        zst_path = os.path.join(tmpdir, "sample.pgn.zst")
        with open(zst_path, "wb") as f:
            f.write(compressed)

        out_dir = os.path.join(tmpdir, "data_out")
        cfg = PipelineConfig(
            source=zst_path,
            output_dir=out_dir,
            max_valid_games=5,
            report_every=2,
        )

        manifest = run_ingestion_pipeline(cfg)

        assert manifest["valid_kept"] == 5
        assert os.path.exists(os.path.join(out_dir, "shards", "train.bin"))
        assert os.path.exists(os.path.join(out_dir, "manifests", "dataset_manifest.json"))
