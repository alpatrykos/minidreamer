from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


def load_config(path: str | Path) -> ConfigDict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return config


def merge_dicts(base: ConfigDict, overrides: ConfigDict) -> ConfigDict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_config(config: ConfigDict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def ensure_run_dirs(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    paths = {
        "base": base,
        "checkpoints": base / "checkpoints",
        "metrics": base / "metrics",
        "plots": base / "plots",
        "replay": base / "replay",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def deep_get(config: ConfigDict, *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

