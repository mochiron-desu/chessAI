import pytest
import chess
import torch
import torch.nn.functional as F
from chesslab.models.resnet import ChessResNet
from chesslab.encode.board_planes import encode_board
from chesslab.encode.move_vocab import encode_move, VOCAB_SIZE
from chesslab.inference.legal_mask import compute_legal_mask, mask_logits


def test_model_forward_shapes():
    model = ChessResNet(num_blocks=2, channels=32)
    model.eval()

    planes = torch.randn(4, 12, 8, 8)
    rule_state = torch.randn(4, 7)

    with torch.no_grad():
        pol, wdl = model(planes, rule_state)

    assert pol.shape == (4, VOCAB_SIZE)
    assert wdl.shape == (4, 3)


def test_overfit_single_batch():
    """Gate 4: Verifies gradient flow, loss calculation, and optimization on a fixed batch."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = ChessResNet(num_blocks=4, channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Construct a synthetic batch of 16 chess states from legal moves
    boards = []
    board = chess.Board()
    moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+", "Bd2", "Bxd2+", "Nbxd2", "d5"]
    for san in moves:
        boards.append(board.copy())
        board.push_san(san)

    planes_list = []
    rule_list = []
    target_move_list = []
    target_wdl_list = []
    mask_list = []

    for idx, b in enumerate(boards):
        p, r = encode_board(b)
        planes_list.append(torch.from_numpy(p))
        rule_list.append(torch.from_numpy(r))
        mask_list.append(torch.from_numpy(compute_legal_mask(b)))

        # Target is the first legal move
        first_legal = list(b.legal_moves)[0]
        target_move_list.append(encode_move(first_legal, b.turn))
        target_wdl_list.append(idx % 3)  # cycle win, draw, loss

    planes = torch.stack(planes_list).to(device)
    rule_state = torch.stack(rule_list).to(device)
    target_moves = torch.tensor(target_move_list, dtype=torch.long, device=device)
    target_wdl = torch.tensor(target_wdl_list, dtype=torch.long, device=device)
    masks = torch.stack(mask_list).to(device)

    # Train for up to 100 steps
    model.train()
    initial_loss = None
    final_loss = None

    for step in range(120):
        optimizer.zero_grad()
        pol_logits, wdl_logits = model(planes, rule_state)

        # Apply legal mask to logits before cross entropy
        masked_pol = mask_logits(pol_logits, masks)
        loss_pol = F.cross_entropy(masked_pol, target_moves)
        loss_wdl = F.cross_entropy(wdl_logits, target_wdl)
        total_loss = loss_pol + 0.5 * loss_wdl

        if step == 0:
            initial_loss = total_loss.item()

        total_loss.backward()
        optimizer.step()

        final_loss = total_loss.item()
        if final_loss < 0.05:
            break

    # Evaluation
    model.eval()
    with torch.no_grad():
        pol_logits, wdl_logits = model(planes, rule_state)
        masked_pol = mask_logits(pol_logits, masks)
        pred_moves = torch.argmax(masked_pol, dim=-1)
        acc = (pred_moves == target_moves).float().mean().item()

    assert final_loss < initial_loss * 0.1, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"
    assert acc >= 0.95, f"Overfit accuracy {acc} is below 95%"
