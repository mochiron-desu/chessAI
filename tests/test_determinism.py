import pytest
import tempfile
import os
import torch
import torch.nn.functional as F
from chesslab.models.resnet import ChessResNet
from chesslab.inference.legal_mask import mask_logits
from chesslab.encode.move_vocab import VOCAB_SIZE


def test_checkpoint_resume_determinism():
    """Gate 5: Verifies that saving, resuming, and retraining produces deterministic bit-for-bit results."""
    device = torch.device("cpu")
    batch_size = 8

    # Fixed synthetic inputs
    torch.manual_seed(100)
    planes = torch.randn(batch_size, 12, 8, 8, device=device)
    rules = torch.randn(batch_size, 7, device=device)
    targets_pol = torch.randint(0, VOCAB_SIZE, (batch_size,), device=device)
    targets_wdl = torch.randint(0, 3, (batch_size,), device=device)
    masks = torch.ones(batch_size, VOCAB_SIZE, dtype=torch.bool, device=device)

    # Run A: Train 10 steps, save, train 10 more steps
    torch.manual_seed(42)
    model_a = ChessResNet(num_blocks=2, channels=32).to(device)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

    for step in range(10):
        opt_a.zero_grad()
        p, w = model_a(planes, rules)
        loss = F.cross_entropy(mask_logits(p, masks), targets_pol) + 0.5 * F.cross_entropy(w, targets_wdl)
        loss.backward()
        opt_a.step()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name
        torch.save({
            "model_state": model_a.state_dict(),
            "opt_state": opt_a.state_dict(),
        }, ckpt_path)

    # Complete 10 more steps on Model A
    for step in range(10):
        opt_a.zero_grad()
        p, w = model_a(planes, rules)
        loss = F.cross_entropy(mask_logits(p, masks), targets_pol) + 0.5 * F.cross_entropy(w, targets_wdl)
        loss.backward()
        opt_a.step()

    # Run B: Resume from checkpoint and train the exact same 10 steps
    model_b = ChessResNet(num_blocks=2, channels=32).to(device)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

    ckpt_loaded = torch.load(ckpt_path, map_location=device)
    model_b.load_state_dict(ckpt_loaded["model_state"])
    opt_b.load_state_dict(ckpt_loaded["opt_state"])

    for step in range(10):
        opt_b.zero_grad()
        p, w = model_b(planes, rules)
        loss = F.cross_entropy(mask_logits(p, masks), targets_pol) + 0.5 * F.cross_entropy(w, targets_wdl)
        loss.backward()
        opt_b.step()

    # Clean up checkpoint file
    os.remove(ckpt_path)

    # Weights must match bit-for-bit
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.all(param_a == param_b), f"Parameter {name_a} diverged after checkpoint resume!"
