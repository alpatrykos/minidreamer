from __future__ import annotations

from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RewardHead(MLPHead):
    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, hidden_dim=hidden_dim)


class DoneHead(MLPHead):
    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, hidden_dim=hidden_dim)

