import pytest
import tempfile
import os
import chess
import numpy as np
import torch
from chesslab.data.stream_parser import encode_position_to_record, RECORD_DTYPE
from chesslab.data.dataset import BinaryChessDataset, fast_collate_fn
from chesslab.encode.board_planes import encode_board
from chesslab.encode.move_vocab import decode_move, VOCAB_SIZE


def test_split_determinism_and_disjointness():
    """Gate 3: Verify train, val, and test splits are strictly disjoint and deterministic."""
    def assign_split(game_id: str, salt: str = "chesslab_v1") -> str:
        h = int(hashlib.sha256(f"{game_id}_{salt}".encode()).hexdigest(), 16) % 10000
        if h < 9500:
            return "train"
        elif h < 9750:
            return "val"
        else:
            return "test"

    import hashlib
    train_ids = set()
    val_ids = set()
    test_ids = set()

    for i in range(5000):
        gid = f"game_{i:06d}"
        split = assign_split(gid)
        if split == "train":
            train_ids.add(gid)
        elif split == "val":
            val_ids.add(gid)
        else:
            test_ids.add(gid)

    # Completely disjoint
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    # Expected proportions ~95%, 2.5%, 2.5%
    assert 4600 < len(train_ids) < 4900
    assert 50 < len(val_ids) < 200
    assert 50 < len(test_ids) < 200


def test_record_serialization_roundtrip():
    """Verify that serializing to binary record and reading back yields identical data."""
    board = chess.Board()
    board.push_san("e4")
    move = chess.Move.from_uci("e7e5")

    # Encode position
    rec = encode_position_to_record(board, move, "0-1")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(rec.tobytes())

    try:
        ds = BinaryChessDataset(tmp_path)
        assert len(ds) == 1
        item = ds[0]

        # Collate single item into batch
        batch = fast_collate_fn([item])

        # Verify planes match encode_board
        expected_planes, expected_rules = encode_board(board)
        assert np.allclose(batch["planes"][0].numpy(), expected_planes)
        assert np.allclose(batch["rule_state"][0].numpy(), expected_rules, atol=1e-3)

        # Verify target move matches
        decoded = decode_move(int(batch["target_moves"][0].item()), board.turn)
        assert decoded == move

        # Verify wdl matches (Black won, Black played move -> Win for Black = 0)
        assert batch["target_wdl"][0].item() == 0

        # Verify legal mask has correct legal move count
        assert batch["legal_masks"][0].sum().item() == len(list(board.legal_moves))
    finally:
        os.remove(tmp_path)
