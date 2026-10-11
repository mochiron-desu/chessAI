import pytest
import torch
import torch.nn as nn
from chesslab.models.resnet import ChessResNet, ResBlock
from chesslab.encode.move_vocab import VOCAB_SIZE
from chesslab.inference.legal_mask import mask_logits


def test_resblock_shape_and_residual_connection():
    channels = 32
    block = ResBlock(channels)
    x = torch.randn(4, channels, 8, 8)

    out = block(x)
    assert out.shape == (4, channels, 8, 8)

    # Test gradient propagation through residual
    loss = out.sum()
    loss.backward()
    assert block.conv1.weight.grad is not None
    assert block.conv2.weight.grad is not None


def test_chess_resnet_shapes_and_parameter_counts():
    # Small test model
    model_small = ChessResNet(num_blocks=2, channels=32)
    params_small = sum(p.numel() for p in model_small.parameters())
    assert 300_000 < params_small < 600_000

    # Baseline 6-block model
    model_base = ChessResNet(num_blocks=6, channels=64)
    params_base = sum(p.numel() for p in model_base.parameters())
    assert 800_000 < params_base < 1_200_000

    # Test forward with multiple batch sizes
    for b_size in [1, 4, 16]:
        planes = torch.randn(b_size, 12, 8, 8)
        rules = torch.randn(b_size, 7)
        with torch.no_grad():
            pol, wdl = model_base(planes, rules)
        assert pol.shape == (b_size, VOCAB_SIZE)
        assert wdl.shape == (b_size, 3)


def test_value_head_probabilities():
    model = ChessResNet(num_blocks=2, channels=32)
    model.eval()

    planes = torch.randn(5, 12, 8, 8)
    rules = torch.randn(5, 7)

    with torch.no_grad():
        _, wdl = model(planes, rules)
        probs = torch.softmax(wdl, dim=-1)

    assert probs.shape == (5, 3)
    # Sum over [Win, Draw, Loss] should equal 1.0
    for p in probs:
        assert pytest.approx(p.sum().item(), abs=1e-5) == 1.0


def test_policy_head_with_legal_masking():
    model = ChessResNet(num_blocks=2, channels=32)
    model.eval()

    planes = torch.randn(2, 12, 8, 8)
    rules = torch.randn(2, 7)
    mask = torch.zeros(2, VOCAB_SIZE, dtype=torch.bool)
    mask[0, 10] = True
    mask[0, 20] = True
    mask[1, 50] = True

    with torch.no_grad():
        pol, _ = model(planes, rules)
        masked = mask_logits(pol, mask)
        probs = torch.softmax(masked, dim=-1)

    # In row 0, only moves 10 and 20 are legal
    assert probs[0, 10] > 0
    assert probs[0, 20] > 0
    assert pytest.approx((probs[0, 10] + probs[0, 20]).item(), abs=1e-5) == 1.0
    assert probs[0, 0] == 0.0

    # In row 1, only move 50 is legal
    assert pytest.approx(probs[1, 50].item(), abs=1e-5) == 1.0


def test_backward_gradient_flow():
    """Verify gradients flow back to all layers in the trunk and heads."""
    model = ChessResNet(num_blocks=4, channels=32)
    planes = torch.randn(2, 12, 8, 8, requires_grad=True)
    rules = torch.randn(2, 7, requires_grad=True)

    pol, wdl = model(planes, rules)
    loss = pol.sum() + wdl.sum()
    loss.backward()

    # Stem gradients
    assert model.conv_in.weight.grad is not None
    assert model.rule_proj.weight.grad is not None
    # Blocks gradients
    for idx, block in enumerate(model.blocks):
        assert block.conv1.weight.grad is not None, f"Block {idx} conv1 has no gradient"
        assert block.conv2.weight.grad is not None, f"Block {idx} conv2 has no gradient"
    # Heads gradients
    assert model.policy_fc.weight.grad is not None
    assert model.value_fc2.weight.grad is not None
