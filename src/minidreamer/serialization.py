from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from minidreamer.models.world_model import WorldModel


def save_world_model_checkpoint(
    path: str | Path,
    model: WorldModel,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_world_model_checkpoint(
    path: str | Path,
    action_dim: int,
    map_location: str | torch.device | None = None,
) -> tuple[WorldModel, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = payload["config"]
    model = WorldModel.from_config(config, action_dim=action_dim)
    model.load_state_dict(payload["model_state"])
    return model, config, payload.get("metadata", {})

