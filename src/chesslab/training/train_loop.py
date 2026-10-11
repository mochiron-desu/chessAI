"""Training engine with learning rate scheduling, EMA, and checkpointing."""

import os
import time
import math
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chesslab.inference.legal_mask import mask_logits
from chesslab.eval.predictive import evaluate_predictive


class ModelEMA:
    """Maintains an exponential moving average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        """Copies shadow weights into the model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name])


def train_chess_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lam_wdl: float = 0.5,
    ema_decay: float = 0.999,
    checkpoint_dir: str = "checkpoints",
    device: torch.device = torch.device("cpu"),
    eval_every_steps: int = 2000,
) -> Dict[str, Any]:
    """Runs full training loop on CPU or GPU."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    def get_lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)
    ema = ModelEMA(model, decay=ema_decay)

    global_step = 0
    history = []
    best_val_top1 = 0.0

    print(f"Starting training: {epochs} epochs, {total_steps} total steps, device={device}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            planes = batch["planes"].to(device)
            rule_state = batch["rule_state"].to(device)
            target_moves = batch["target_moves"].to(device)
            target_wdl = batch["target_wdl"].to(device)
            legal_masks = batch["legal_masks"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pol_logits, wdl_logits = model(planes, rule_state)

            masked_pol = mask_logits(pol_logits, legal_masks)
            loss_pol = F.cross_entropy(masked_pol, target_moves)
            loss_wdl = F.cross_entropy(wdl_logits, target_wdl)
            total_loss = loss_pol + lam_wdl * loss_wdl

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_loss += total_loss.item()
            global_step += 1

            if global_step % eval_every_steps == 0 and val_loader is not None:
                val_metrics = evaluate_predictive(model, val_loader, device=device, max_samples=1000)
                print(
                    f"Step {global_step}/{total_steps} | "
                    f"Train Loss: {total_loss.item():.4f} | "
                    f"Val Top-1: {val_metrics['top1_accuracy']*100:.2f}% | "
                    f"Val Top-5: {val_metrics['top5_accuracy']*100:.2f}% | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                if val_metrics["top1_accuracy"] > best_val_top1:
                    best_val_top1 = val_metrics["top1_accuracy"]
                    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_top1": best_val_top1,
                    }, ckpt_path)

        t1 = time.time()
        elapsed = t1 - t0
        avg_loss = epoch_loss / len(train_loader)
        throughput = len(train_loader.dataset) / elapsed
        print(f"Epoch {epoch}/{epochs} complete in {elapsed:.1f}s ({throughput:.1f} pos/sec) | Avg Loss: {avg_loss:.4f}")

        if val_loader is not None:
            full_val = evaluate_predictive(model, val_loader, device=device)
            print(
                f"--> Epoch {epoch} Full Validation ({full_val['samples']} positions): "
                f"Top-1: {full_val['top1_accuracy']*100:.2f}% | "
                f"Top-5: {full_val['top5_accuracy']*100:.2f}% | "
                f"WDL Acc: {full_val['wdl_accuracy']*100:.2f}%"
            )

    # Save final model with EMA weights applied
    ema.apply_shadow(model)
    final_path = os.path.join(checkpoint_dir, "final_ema_model.pt")
    torch.save({
        "step": global_step,
        "epochs": epochs,
        "model_state": model.state_dict(),
    }, final_path)
    print(f"Training complete. Final EMA weights saved to {final_path}")

    return {
        "final_step": global_step,
        "final_checkpoint": final_path,
        "best_val_top1": best_val_top1,
    }
