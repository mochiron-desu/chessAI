#!/usr/bin/env python3
"""Training entrypoint for ChessLab.

Loads binary shards, initializes ChessResNet, and runs training loop on CPU.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from chesslab.models.resnet import ChessResNet
from chesslab.data.dataset import BinaryChessDataset, fast_collate_fn
from chesslab.training.train_loop import train_chess_model
from chesslab.eval.predictive import evaluate_predictive


def main():
    parser = argparse.ArgumentParser(description="Train ChessResNet on binary shards")
    parser.add_argument("--data-dir", type=str, default="data/shards", help="Directory containing train.bin and val.bin")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--channels", type=int, default=64, help="Model channels (64 = ~960k params, 32 = ~350k params)")
    parser.add_argument("--blocks", type=int, default=6, help="Number of residual blocks")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoints output directory")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--train-shard", type=str, default="", help="Path to training shard (overrides --data-dir)")
    parser.add_argument("--val-shard", type=str, default="", help="Path to val shard (overrides --data-dir)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model init and data shuffling")
    parser.add_argument("--num-threads", type=int, default=8, help="PyTorch CPU intra-op threads")
    args = parser.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")

    train_path = args.train_shard if args.train_shard else os.path.join(args.data_dir, "train.bin")
    val_path = args.val_shard if args.val_shard else os.path.join(args.data_dir, "val.bin")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} does not exist!")
        return

    print(f"Loading training shard from {train_path}...")
    train_dataset = BinaryChessDataset(train_path)
    print(f"Train dataset size: {len(train_dataset):,} positions")

    val_dataset = BinaryChessDataset(val_path) if os.path.exists(val_path) and os.path.getsize(val_path) > 0 else None
    if val_dataset:
        print(f"Val dataset size: {len(val_dataset):,} positions")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=fast_collate_fn,
        num_workers=0,  # memmap zero-copy works fastest in main process on Linux
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=fast_collate_fn,
        num_workers=0,
    ) if val_dataset else None

    # Instantiate model
    model = ChessResNet(num_blocks=args.blocks, channels=args.channels)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: ChessResNet({args.blocks} blocks, {args.channels} channels) | {total_params:,} parameters")

    # Run training
    results = train_chess_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        eval_every_steps=args.eval_every,
    )

    # Final evaluation on val set
    if val_loader:
        final_metrics = evaluate_predictive(model, val_loader, device=device)
        print("\n=== Final Validation Metrics ===")
        for k, v in final_metrics.items():
            if "accuracy" in k or "rate" in k:
                print(f"  {k}: {v*100:.2f}%")
            else:
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
