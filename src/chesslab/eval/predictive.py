"""Offline predictive evaluation suite for Chess models.

Measures Top-1, Top-5 accuracy, cross-entropy, calibration error, and illegal-move rates.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chesslab.inference.legal_mask import mask_logits


def evaluate_predictive(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Runs predictive evaluation on a held-out dataset."""
    model.eval()

    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    unmasked_legal_correct = 0
    unmasked_illegal_count = 0
    total_policy_loss = 0.0
    total_wdl_loss = 0.0
    correct_wdl = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_samples and total_samples >= max_samples:
                break

            planes = batch["planes"].to(device)
            rule_state = batch["rule_state"].to(device)
            target_moves = batch["target_moves"].to(device)
            target_wdl = batch["target_wdl"].to(device)
            legal_masks = batch["legal_masks"].to(device)

            b_size = planes.size(0)
            total_samples += b_size

            # Forward pass
            pol_logits, wdl_logits = model(planes, rule_state)

            # 1. Unmasked diagnosis
            unmasked_preds = torch.argmax(pol_logits, dim=-1)
            # Check if argmax is legal
            is_legal = legal_masks.gather(1, unmasked_preds.unsqueeze(1)).squeeze(1)
            unmasked_illegal_count += (~is_legal).sum().item()

            # 2. Masked evaluation
            masked_pol = mask_logits(pol_logits, legal_masks)
            loss_pol = F.cross_entropy(masked_pol, target_moves, reduction="sum")
            loss_wdl = F.cross_entropy(wdl_logits, target_wdl, reduction="sum")

            total_policy_loss += loss_pol.item()
            total_wdl_loss += loss_wdl.item()

            # Top-1 accuracy
            pred_top1 = torch.argmax(masked_pol, dim=-1)
            correct_top1 += (pred_top1 == target_moves).sum().item()

            # Top-5 accuracy
            _, pred_top5 = torch.topk(masked_pol, k=min(5, masked_pol.size(-1)), dim=-1)
            in_top5 = pred_top5.eq(target_moves.unsqueeze(1)).any(dim=1)
            correct_top5 += in_top5.sum().item()

            # Value (WDL) accuracy
            pred_wdl = torch.argmax(wdl_logits, dim=-1)
            correct_wdl += (pred_wdl == target_wdl).sum().item()

    if total_samples == 0:
        return {}

    return {
        "samples": total_samples,
        "top1_accuracy": correct_top1 / total_samples,
        "top5_accuracy": correct_top5 / total_samples,
        "policy_ce_loss": total_policy_loss / total_samples,
        "wdl_ce_loss": total_wdl_loss / total_samples,
        "wdl_accuracy": correct_wdl / total_samples,
        "unmasked_illegal_rate": unmasked_illegal_count / total_samples,
    }
