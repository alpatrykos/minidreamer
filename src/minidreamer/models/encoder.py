from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(obs)
        return self.projection(hidden)

