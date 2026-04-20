from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minidreamer.models.rssm import RSSMState
from minidreamer.models.world_model import WorldModel


@dataclass
class PlannerOutput:
    action: int
    sequence: list[int]
    score: float
    policy: torch.Tensor
    entropy: float


class DiscreteCEMPlanner:
    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int,
        horizon: int = 8,
        candidates: int = 256,
        elites: int = 32,
        iterations: int = 4,
        discount: float = 0.99,
        use_done_mask: bool = True,
        smoothing: float = 1e-3,
    ) -> None:
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.candidates = candidates
        self.elites = min(elites, candidates)
        self.iterations = iterations
        self.discount = discount
        self.use_done_mask = use_done_mask
        self.smoothing = smoothing

    @classmethod
    def from_config(cls, world_model: WorldModel, action_dim: int, config: dict) -> "DiscreteCEMPlanner":
        planner_cfg = config["planner"]
        return cls(
            world_model=world_model,
            action_dim=action_dim,
            horizon=planner_cfg["horizon"],
            candidates=planner_cfg["candidates"],
            elites=planner_cfg["elites"],
            iterations=planner_cfg["iterations"],
            discount=planner_cfg["discount"],
            use_done_mask=planner_cfg.get("use_done_mask", True),
        )

    def _sample_sequences(self, probs: torch.Tensor) -> torch.Tensor:
        flat = probs.unsqueeze(0).expand(self.candidates, -1, -1).reshape(-1, self.action_dim)
        sampled = torch.multinomial(flat, num_samples=1, replacement=True)
        return sampled.view(self.candidates, self.horizon)

    def plan(self, state: RSSMState) -> PlannerOutput:
        device = self.world_model.device
        probs = torch.full(
            (self.horizon, self.action_dim),
            fill_value=1.0 / self.action_dim,
            device=device,
        )
        best_sequence = None
        best_score = torch.tensor(float("-inf"), device=device)

        for _ in range(self.iterations):
            action_sequences = self._sample_sequences(probs)
            scores = self.world_model.score_action_sequences(
                state,
                action_sequences,
                discount=self.discount,
                use_done_mask=self.use_done_mask,
            )["scores"]
            elite_indices = torch.topk(scores, k=self.elites, largest=True).indices
            elites = action_sequences[elite_indices]
            elite_freq = F.one_hot(elites, num_classes=self.action_dim).float().mean(dim=0)
            probs = elite_freq + self.smoothing
            probs = probs / probs.sum(dim=-1, keepdim=True)

            iteration_best_idx = scores.argmax()
            iteration_best_score = scores[iteration_best_idx]
            if iteration_best_score > best_score:
                best_score = iteration_best_score
                best_sequence = action_sequences[iteration_best_idx]

        if best_sequence is None:
            raise RuntimeError("CEM planner failed to sample any action sequence.")
        entropy = float((-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean().detach().cpu())
        return PlannerOutput(
            action=int(best_sequence[0].item()),
            sequence=[int(action.item()) for action in best_sequence],
            score=float(best_score.detach().cpu()),
            policy=probs.detach().cpu(),
            entropy=entropy,
        )

