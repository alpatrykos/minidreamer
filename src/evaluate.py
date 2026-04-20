from __future__ import annotations

import argparse
from pathlib import Path

from minidreamer.config import load_config
from minidreamer.data.replay_buffer import ReplayBuffer
from minidreamer.evaluation import evaluate_random_policy, evaluate_world_model
from minidreamer.envs.make_env import make_env_from_config
from minidreamer.planning.evaluate_planner import evaluate_planner
from minidreamer.serialization import load_world_model_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MiniDreamer components.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser("random", help="Evaluate a random policy.")
    random_parser.add_argument("--config", type=Path, required=True)

    planner_parser = subparsers.add_parser("planner", help="Evaluate a trained planner.")
    planner_parser.add_argument("--config", type=Path, required=True)
    planner_parser.add_argument("--checkpoint", type=Path, required=True)

    world_model_parser = subparsers.add_parser("world-model", help="Evaluate held-out world model metrics.")
    world_model_parser.add_argument("--config", type=Path, required=True)
    world_model_parser.add_argument("--checkpoint", type=Path, required=True)
    world_model_parser.add_argument("--replay-dir", type=Path, required=True)
    world_model_parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "random":
        print(evaluate_random_policy(config))
        return

    env = make_env_from_config(config, seed=config.get("project", {}).get("seed", 0))
    action_dim = env.action_space.n
    env.close()
    model, _, metadata = load_world_model_checkpoint(args.checkpoint, action_dim=action_dim, map_location="cpu")

    if args.command == "planner":
        print({"metadata": metadata, **evaluate_planner(config, model)})
        return

    replay = ReplayBuffer.load(args.replay_dir)
    print({"metadata": metadata, **evaluate_world_model(config, model, replay, split=args.split)})


if __name__ == "__main__":
    main()
