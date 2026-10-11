"""High-throughput memory-mapped PyTorch Dataset for Chess binary shards."""

from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset

from chesslab.data.stream_parser import RECORD_DTYPE
from chesslab.encode.move_vocab import VOCAB_SIZE


class BinaryChessDataset(Dataset):
    """Memory-mapped dataset reading packed binary records with zero copy."""

    def __init__(self, shard_path: str):
        self.shard_path = shard_path
        self.data = np.memmap(shard_path, dtype=RECORD_DTYPE, mode="r")
        self.length = len(self.data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.data[idx]
        return {
            "bitboards": rec["bitboards"],
            "rule_state": rec["rule_state"].astype(np.float32),
            "move_idx": int(rec["move_idx"]),
            "wdl": int(rec["wdl"]),
            "legal_mask_bits": rec["legal_mask"],
        }


def fast_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Vectorized batch collation that unpacks bitboards and legal masks in parallel."""
    b_size = len(batch)

    # 1. Unpack 12 bitboards for all samples in batch
    # Shape: (B, 12) uint64 -> (B, 12, 8) uint8 -> (B, 12, 8, 8) float32
    bbs = np.stack([item["bitboards"] for item in batch], axis=0)  # (B, 12)
    unpacked_planes = np.unpackbits(
        bbs.view(np.uint8).reshape(b_size, 12, 8),
        axis=-1,
        bitorder="little"
    ).reshape(b_size, 12, 8, 8).astype(np.float32)

    # 2. Rule states
    rule_states = np.stack([item["rule_state"] for item in batch], axis=0)

    # 3. Targets
    moves = np.array([item["move_idx"] for item in batch], dtype=np.int64)
    wdls = np.array([item["wdl"] for item in batch], dtype=np.int64)

    # 4. Unpack legal masks: (B, 233) -> (B, 1864) -> truncate to VOCAB_SIZE
    masks_packed = np.stack([item["legal_mask_bits"] for item in batch], axis=0)
    masks_bool = np.unpackbits(masks_packed, axis=-1)[:, :VOCAB_SIZE].astype(bool)

    return {
        "planes": torch.from_numpy(unpacked_planes),
        "rule_state": torch.from_numpy(rule_states),
        "target_moves": torch.from_numpy(moves),
        "target_wdl": torch.from_numpy(wdls),
        "legal_masks": torch.from_numpy(masks_bool),
    }
