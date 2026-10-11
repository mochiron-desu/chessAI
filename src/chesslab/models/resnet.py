"""Compact Residual Convolutional Neural Network for Chess policy and value prediction.

Optimized for fast CPU inference and training with Intel oneDNN / AVX2.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from chesslab.encode.move_vocab import VOCAB_SIZE


class ResBlock(nn.Module):
    """Residual convolutional block with two 3x3 convs and batch normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)


class ChessResNet(nn.Module):
    """Compact Chess ResNet predicting Move Policy and Win/Draw/Loss Value.
    
    Inputs:
        planes: (B, 12, 8, 8) float tensor of spatial board features.
        rule_state: (B, 7) float tensor of castling/ep/clock features.
        
    Outputs:
        policy_logits: (B, VOCAB_SIZE) float tensor of unmasked move logits.
        wdl_logits: (B, 3) float tensor of Win/Draw/Loss outcome logits.
    """

    def __init__(
        self,
        num_blocks: int = 6,
        channels: int = 64,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.channels = channels
        self.vocab_size = vocab_size

        # Input stem
        self.conv_in = nn.Conv2d(12, channels, kernel_size=3, padding=1, bias=False)
        self.rule_proj = nn.Linear(7, channels)
        self.bn_in = nn.BatchNorm2d(channels)

        # Residual tower
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Policy head: 64 channels -> 4 channels (256 dims) -> Linear to vocab
        self.policy_conv = nn.Conv2d(channels, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * 8 * 8, vocab_size)

        # Value head: 64 channels -> 2 channels (128 dims) -> MLP -> 3 (WDL)
        self.value_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 3)

        self._init_weights()

    def _init_weights(self):
        """Initializes weights with Kaiming Normal and zero-residual scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Zero-init the last BN in each residual block for identity at initialization
        for block in self.blocks:
            nn.init.constant_(block.bn2.weight, 0.0)

    def forward(
        self, planes: torch.Tensor, rule_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial conv
        x = self.conv_in(planes)
        # Broadcast rule state across 8x8 spatial plane
        r = self.rule_proj(rule_state).unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.bn_in(x + r), inplace=True)

        # Residual tower
        for block in self.blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v), inplace=True)
        wdl_logits = self.value_fc2(v)

        return policy_logits, wdl_logits
