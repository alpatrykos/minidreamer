from __future__ import annotations

from torch import nn


class ConvDecoder(nn.Module):
    def __init__(self, feature_dim: int, out_channels: int = 3) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        hidden = self.projection(features).view(-1, 256, 4, 4)
        return self.decoder(hidden)

