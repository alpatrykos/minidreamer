from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from minidreamer.models.decoder import ConvDecoder
from minidreamer.models.encoder import ConvEncoder
from minidreamer.models.heads import DoneHead, RewardHead
from minidreamer.models.rssm import RSSM, RSSMState
from minidreamer.utils.common import masked_mean


@dataclass
class WorldModelOutputs:
    states: list[RSSMState]
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    post_mean: torch.Tensor
    post_std: torch.Tensor
    reward_pred: torch.Tensor
    done_logits: torch.Tensor
    reconstructions: torch.Tensor | None


class WorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embedding_dim: int = 256,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        use_decoder: bool = True,
        min_std: float = 0.1,
        obs_channels: int = 3,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.use_decoder = use_decoder
        self.obs_channels = obs_channels

        self.encoder = ConvEncoder(in_channels=obs_channels, embedding_dim=embedding_dim)
        self.rssm = RSSM(
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
            min_std=min_std,
        )
        feature_dim = deter_dim + stoch_dim
        self.reward_head = RewardHead(feature_dim, hidden_dim=hidden_dim)
        self.done_head = DoneHead(feature_dim, hidden_dim=hidden_dim)
        self.decoder = ConvDecoder(feature_dim, out_channels=obs_channels) if use_decoder else None

    @classmethod
    def from_config(cls, config: dict, action_dim: int, obs_channels: int = 3) -> "WorldModel":
        model_cfg = config["model"]
        return cls(
            action_dim=action_dim,
            embedding_dim=model_cfg["embedding_dim"],
            deter_dim=model_cfg["deter_dim"],
            stoch_dim=model_cfg["stoch_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            use_decoder=model_cfg.get("use_decoder", True),
            min_std=model_cfg.get("min_std", 0.1),
            obs_channels=obs_channels,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int) -> RSSMState:
        return self.rssm.initial(batch_size=batch_size, device=self.device)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() < 4:
            raise ValueError(f"Expected at least 4 dims for observations, got {obs.shape}.")
        leading_shape = obs.shape[:-3]
        flat_obs = obs.reshape(-1, *obs.shape[-3:])
        flat_embeddings = self.encoder(flat_obs)
        return flat_embeddings.reshape(*leading_shape, -1)

    def observe_sequence(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        sample: bool = True,
    ) -> WorldModelOutputs:
        if obs.dim() != 5:
            raise ValueError(f"Expected obs shape [B, T+1, C, H, W], got {obs.shape}.")
        if actions.dim() != 2:
            raise ValueError(f"Expected actions shape [B, T], got {actions.shape}.")
        batch_size, time_steps = actions.shape
        embeddings = self.encode(obs)
        state, _ = self.rssm.observe(self.initial_state(batch_size), None, embeddings[:, 0], sample=sample)

        states = [state]
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []
        rewards = []
        done_logits = []
        reconstructions = []

        for t in range(time_steps):
            next_state, (prior_mean, prior_std) = self.rssm.observe(
                state,
                actions[:, t],
                embeddings[:, t + 1],
                sample=sample,
            )
            features = next_state.features()
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(next_state.mean)
            post_stds.append(next_state.std)
            rewards.append(self.reward_head(features).squeeze(-1))
            done_logits.append(self.done_head(features).squeeze(-1))
            if self.decoder is not None:
                reconstructions.append(self.decoder(features))
            states.append(next_state)
            state = next_state

        recon_tensor = torch.stack(reconstructions, dim=1) if reconstructions else None
        return WorldModelOutputs(
            states=states,
            prior_mean=torch.stack(prior_means, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
            post_mean=torch.stack(post_means, dim=1),
            post_std=torch.stack(post_stds, dim=1),
            reward_pred=torch.stack(rewards, dim=1),
            done_logits=torch.stack(done_logits, dim=1),
            reconstructions=recon_tensor,
        )

    def compute_losses(self, batch: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self.observe_sequence(batch["obs"], batch["actions"], sample=True)
        training_cfg = config["training"]
        rewards = batch["rewards"]
        done = batch["done"]
        mask = batch["mask"]

        reward_loss = masked_mean(F.mse_loss(outputs.reward_pred, rewards, reduction="none"), mask)
        done_loss = masked_mean(
            F.binary_cross_entropy_with_logits(outputs.done_logits, done, reduction="none"),
            mask,
        )
        kl_per_step = self.rssm.kl_divergence(
            outputs.post_mean,
            outputs.post_std,
            outputs.prior_mean,
            outputs.prior_std,
        )
        free_nats = torch.full_like(kl_per_step, float(training_cfg.get("free_nats", 1.0)))
        kl_loss = masked_mean(torch.maximum(kl_per_step, free_nats), mask)

        if outputs.reconstructions is not None and training_cfg.get("beta_recon", 0.0) > 0.0:
            recon_target = batch["obs"][:, 1:]
            recon_error = F.mse_loss(outputs.reconstructions, recon_target, reduction="none").mean(dim=(2, 3, 4))
            recon_loss = masked_mean(recon_error, mask)
        else:
            recon_loss = torch.zeros((), device=self.device)

        total_loss = (
            float(training_cfg.get("beta_reward", 1.0)) * reward_loss
            + float(training_cfg.get("beta_done", 1.0)) * done_loss
            + float(training_cfg.get("beta_kl", 1.0)) * kl_loss
            + float(training_cfg.get("beta_recon", 0.0)) * recon_loss
        )
        return {
            "loss": total_loss,
            "reward_loss": reward_loss.detach(),
            "done_loss": done_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "reward_mse": F.mse_loss(outputs.reward_pred, rewards, reduction="none").mul(mask).sum() / mask.sum().clamp_min(1.0),
            "done_bce": F.binary_cross_entropy_with_logits(outputs.done_logits, done, reduction="none").mul(mask).sum() / mask.sum().clamp_min(1.0),
        }

    def posterior_step(
        self,
        prev_state: RSSMState,
        prev_action: int | torch.Tensor | None,
        observation: np.ndarray | torch.Tensor,
        sample: bool = False,
    ) -> RSSMState:
        obs_tensor = self._prepare_single_observation(observation)
        embed = self.encode(obs_tensor)
        if prev_action is None:
            action_tensor = None
        else:
            action_tensor = torch.as_tensor(prev_action, device=self.device).view(1)
        state, _ = self.rssm.observe(prev_state, action_tensor, embed, sample=sample)
        return state

    def imagine_rollout(
        self,
        start_state: RSSMState,
        action_sequences: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        if action_sequences.dim() == 1:
            action_sequences = action_sequences.unsqueeze(0)
        batch_size, horizon = action_sequences.shape
        state = start_state if start_state.deter.shape[0] == batch_size else start_state.repeat(batch_size)
        rewards = []
        done_logits = []
        states = []
        for t in range(horizon):
            state, _ = self.rssm.imagine(state, action_sequences[:, t], sample=sample)
            features = state.features()
            rewards.append(self.reward_head(features).squeeze(-1))
            done_logits.append(self.done_head(features).squeeze(-1))
            states.append(state)
        return {
            "states": states,
            "reward_pred": torch.stack(rewards, dim=1),
            "done_logits": torch.stack(done_logits, dim=1),
        }

    def score_action_sequences(
        self,
        start_state: RSSMState,
        action_sequences: torch.Tensor,
        discount: float = 0.99,
        use_done_mask: bool = True,
    ) -> dict[str, torch.Tensor]:
        rollout = self.imagine_rollout(start_state, action_sequences, sample=False)
        reward_pred = rollout["reward_pred"]
        done_prob = torch.sigmoid(rollout["done_logits"])
        alive = torch.ones(reward_pred.shape[0], device=self.device)
        scores = torch.zeros_like(alive)
        for t in range(reward_pred.shape[1]):
            scores = scores + (discount**t) * alive * reward_pred[:, t]
            if use_done_mask:
                alive = alive * (1.0 - done_prob[:, t])
        rollout["scores"] = scores
        rollout["done_prob"] = done_prob
        return rollout

    def _prepare_single_observation(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(observation):
            obs_tensor = observation.to(self.device).float()
        else:
            obs_tensor = torch.as_tensor(observation, device=self.device).float()
        if obs_tensor.dim() == 3 and obs_tensor.shape[-1] in (1, 3):
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
        elif obs_tensor.dim() == 3 and obs_tensor.shape[0] in (1, 3):
            obs_tensor = obs_tensor.unsqueeze(0)
        elif obs_tensor.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported observation shape {tuple(obs_tensor.shape)}.")
        return obs_tensor

