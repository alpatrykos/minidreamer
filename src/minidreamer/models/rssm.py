from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def features(self) -> torch.Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            mean=self.mean.detach(),
            std=self.std.detach(),
        )

    def repeat(self, count: int) -> "RSSMState":
        return RSSMState(
            deter=self.deter.repeat(count, 1),
            stoch=self.stoch.repeat(count, 1),
            mean=self.mean.repeat(count, 1),
            std=self.std.repeat(count, 1),
        )


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embedding_dim: int = 256,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        self.input_net = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embedding_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        zeros_deter = torch.zeros(batch_size, self.deter_dim, device=device)
        zeros_stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return RSSMState(
            deter=zeros_deter,
            stoch=zeros_stoch,
            mean=zeros_stoch,
            std=torch.ones_like(zeros_stoch),
        )

    def _stats(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(tensor, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def _action_one_hot(self, action: torch.Tensor) -> torch.Tensor:
        action = action.long().view(-1)
        return F.one_hot(action, num_classes=self.action_dim).float()

    def _next_deter(self, prev_state: RSSMState, action: torch.Tensor) -> torch.Tensor:
        action_one_hot = self._action_one_hot(action)
        gru_input = self.input_net(torch.cat([prev_state.stoch, action_one_hot], dim=-1))
        return self.gru(gru_input, prev_state.deter)

    def prior(self, deter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._stats(self.prior_net(deter))

    def posterior(self, deter: torch.Tensor, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._stats(self.posterior_net(torch.cat([deter, embed], dim=-1)))

    @staticmethod
    def sample(mean: torch.Tensor, std: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            return mean + torch.randn_like(std) * std
        return mean

    def observe(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor | None,
        embed: torch.Tensor,
        sample: bool = True,
    ) -> tuple[RSSMState, tuple[torch.Tensor, torch.Tensor]]:
        if prev_action is None:
            deter = prev_state.deter
        else:
            deter = self._next_deter(prev_state, prev_action)
        prior_mean, prior_std = self.prior(deter)
        post_mean, post_std = self.posterior(deter, embed)
        stoch = self.sample(post_mean, post_std, sample=sample)
        state = RSSMState(deter=deter, stoch=stoch, mean=post_mean, std=post_std)
        return state, (prior_mean, prior_std)

    def imagine(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        sample: bool = False,
    ) -> tuple[RSSMState, tuple[torch.Tensor, torch.Tensor]]:
        deter = self._next_deter(prev_state, action)
        mean, std = self.prior(deter)
        stoch = self.sample(mean, std, sample=sample)
        state = RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)
        return state, (mean, std)

    @staticmethod
    def kl_divergence(
        post_mean: torch.Tensor,
        post_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
    ) -> torch.Tensor:
        log_var_ratio = 2.0 * (torch.log(prior_std) - torch.log(post_std))
        var_ratio = (post_std / prior_std) ** 2
        mean_term = ((post_mean - prior_mean) / prior_std) ** 2
        return 0.5 * torch.sum(var_ratio + mean_term + log_var_ratio - 1.0, dim=-1)

