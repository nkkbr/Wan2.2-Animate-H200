from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class EdgeRefinementNet(nn.Module):
    def __init__(self, in_channels: int = 5, hidden_channels: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


@dataclass
class EdgeRefinementOutputs:
    alpha: torch.Tensor
    boundary: torch.Tensor


def split_outputs(logits: torch.Tensor) -> EdgeRefinementOutputs:
    if logits.shape[1] != 2:
        raise ValueError(f"Expected 2 output channels, got {logits.shape}.")
    alpha = torch.sigmoid(logits[:, 0:1])
    boundary = torch.sigmoid(logits[:, 1:2])
    return EdgeRefinementOutputs(alpha=alpha, boundary=boundary)
