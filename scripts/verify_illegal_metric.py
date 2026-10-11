#!/usr/bin/env python3
"""Experimentally verify the explanation for the ~100% raw (unmasked) illegal-move rate.

Hypothesis: training loss masks illegal logits to -inf before softmax, so illegal logits
receive exactly zero gradient and are never pushed down; raw argmax is therefore illegal.

Experiments:
  E1. Raw / masked argmax legality of the real checkpoint on N held-out test positions.
  E2. Gradient on illegal logits under masked CE (must be exactly 0) vs full CE (nonzero).
  E3. Controlled: same seed/data/steps, small model trained with masked CE vs full (unmasked) CE;
      compare raw argmax illegal rate on held-out positions.
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from chesslab.models.resnet import ChessResNet
from chesslab.data.dataset import BinaryChessDataset, fast_collate_fn
from chesslab.inference.legal_mask import mask_logits


def raw_stats(model, loader, max_n):
    model.eval()
    n = raw_illegal = masked_illegal = 0
    leg_mean, ill_max = [], []
    with torch.no_grad():
        for b in loader:
            if n >= max_n:
                break
            pol, _ = model(b["planes"], b["rule_state"])
            m = b["legal_masks"]
            raw_arg = pol.argmax(-1)
            raw_illegal += (~m.gather(1, raw_arg[:, None]).squeeze(1)).sum().item()
            mk = mask_logits(pol, m)
            masked_illegal += (~m.gather(1, mk.argmax(-1)[:, None]).squeeze(1)).sum().item()
            leg_mean.append(pol.masked_fill(~m, 0).sum().item() / m.sum().item())
            ill_max.append(pol.masked_fill(m, float("-inf")).max(-1).values.mean().item())
            n += pol.size(0)
    return {
        "samples": n,
        "raw_argmax_illegal_rate": raw_illegal / n,
        "masked_argmax_illegal_rate": masked_illegal / n,
        "mean_logit_of_legal_moves": sum(leg_mean) / len(leg_mean),
        "mean_of_per_position_max_illegal_logit": sum(ill_max) / len(ill_max),
    }


def train_small(masked, train_ds, steps, seed, bs=128):
    torch.manual_seed(seed)
    model = ChessResNet(num_blocks=2, channels=32)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=fast_collate_fn, generator=g)
    model.train()
    step = 0
    while step < steps:
        for b in loader:
            pol, _ = model(b["planes"], b["rule_state"])
            logits = mask_logits(pol, b["legal_masks"]) if masked else pol
            loss = F.cross_entropy(logits, b["target_moves"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= steps:
                break
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/run_1m/final_ema_model.pt")
    ap.add_argument("--test", default="data/shards/test.bin")
    ap.add_argument("--train", default="data/shards/train.bin")
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--train-subset", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/verify_illegal_metric.json")
    a = ap.parse_args()
    torch.set_num_threads(6)

    test_ds = BinaryChessDataset(a.test)
    test_loader = DataLoader(Subset(test_ds, range(min(a.n_eval, len(test_ds)))),
                             batch_size=256, shuffle=False, collate_fn=fast_collate_fn)
    out = {"seed": a.seed, "torch": torch.__version__}

    # E1
    model = ChessResNet(num_blocks=6, channels=64)
    ck = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
    out["E1_checkpoint"] = raw_stats(model, test_loader, a.n_eval)
    out["E1_checkpoint"]["checkpoint"] = a.ckpt

    # E2
    b = next(iter(DataLoader(test_ds, batch_size=256, collate_fn=fast_collate_fn)))
    model.eval()
    res = {}
    for name, masked in (("masked_ce", True), ("full_ce", False)):
        model.zero_grad()
        pol, _ = model(b["planes"], b["rule_state"])
        pol.retain_grad()
        logits = mask_logits(pol, b["legal_masks"]) if masked else pol
        F.cross_entropy(logits, b["target_moves"]).backward()
        ill = ~b["legal_masks"]
        res[name] = {"max_abs_grad_on_illegal_logits": pol.grad[ill].abs().max().item(),
                     "mean_abs_grad_on_illegal_logits": pol.grad[ill].abs().mean().item()}
    out["E2_gradients"] = res

    # E3
    train_ds = Subset(BinaryChessDataset(a.train), range(a.train_subset))
    out["E3_controlled"] = {"train_subset": a.train_subset, "steps": a.steps, "arch": "ChessResNet(2 blocks, 32 ch)"}
    for name, masked in (("masked_ce", True), ("full_ce", False)):
        m = train_small(masked, train_ds, a.steps, a.seed)
        out["E3_controlled"][name] = raw_stats(m, test_loader, a.n_eval)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
