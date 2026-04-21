from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "artifacts" / "world_model" / "metrics"
PLOTS_DIR = ROOT / "plots"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        out.append(running_sum / min(idx + 1, window))
    return out


def generate_learning_curves(train_rows: list[dict]) -> None:
    if not train_rows:
        return
    steps = list(range(1, len(train_rows) + 1))
    window = min(250, len(train_rows))
    loss = rolling_mean([row["loss"] for row in train_rows], window)
    kl = rolling_mean([row["kl_loss"] for row in train_rows], window)
    done = rolling_mean([row["done_loss"] for row in train_rows], window)
    recon = rolling_mean([row["recon_loss"] for row in train_rows], window)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, loss, label="total loss", linewidth=2.0)
    ax.plot(steps, kl, label="kl loss", linewidth=1.5)
    ax.plot(steps, done, label="done loss", linewidth=1.5)
    ax.plot(steps, recon, label="recon loss", linewidth=1.5)
    ax.set_title("World Model Training Curves")
    ax.set_xlabel("Gradient update")
    ax.set_ylabel("Smoothed loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "learning_curves.png", dpi=200)
    plt.close(fig)


def generate_success_plot(eval_rows: list[dict], final_eval: dict | None) -> None:
    steps: list[int] = []
    planner_success: list[float] = []
    random_success: list[float] = []

    for row in eval_rows:
        steps.append(int(row["env_steps"]))
        planner_success.append(float(row["planner/success_rate"]))
        random_success.append(float(row["random/success_rate"]))

    if final_eval is not None:
        steps.append(int(final_eval["metadata"]["env_steps"]))
        planner_success.append(float(final_eval["planner"]["success_rate"]))
        random_success.append(float(final_eval["random"]["success_rate"]))

    if not steps:
        return

    paired = sorted(zip(steps, planner_success, random_success), key=lambda item: item[0])
    steps = [item[0] for item in paired]
    planner_success = [item[1] for item in paired]
    random_success = [item[2] for item in paired]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, planner_success, marker="o", linewidth=2.0, label="planner success rate")
    ax.plot(steps, random_success, marker="o", linewidth=2.0, label="random success rate")
    ax.set_title("Success Rate vs Environment Steps")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "success_rate_vs_env_steps.png", dpi=200)
    plt.close(fig)


def generate_rollout_error_plot(final_eval: dict | None) -> None:
    if final_eval is None:
        return
    horizons = [1, 5, 10]
    reward_errors = [
        float(final_eval["world_model"][f"open_loop_reward_error_h{h}"])
        for h in horizons
    ]
    done_accuracy = [
        float(final_eval["world_model"][f"open_loop_done_accuracy_h{h}"])
        for h in horizons
    ]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar([str(h) for h in horizons], reward_errors, color="#2b6cb0", alpha=0.8)
    ax1.set_xlabel("Open-loop horizon")
    ax1.set_ylabel("Reward MSE", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.set_title("Model Error vs Rollout Horizon")
    ax1.grid(alpha=0.2, axis="y")

    ax2 = ax1.twinx()
    ax2.plot([str(h) for h in horizons], done_accuracy, color="#c05621", marker="o", linewidth=2.0)
    ax2.set_ylabel("Done accuracy", color="#c05621")
    ax2.tick_params(axis="y", labelcolor="#c05621")
    ax2.set_ylim(0.95, 1.001)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_error_vs_rollout_horizon.png", dpi=200)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(METRICS_DIR / "train_metrics.jsonl")
    eval_rows = load_jsonl(METRICS_DIR / "eval_metrics.jsonl")
    final_eval_path = METRICS_DIR / "final_eval_latest.json"
    final_eval = json.loads(final_eval_path.read_text(encoding="utf-8")) if final_eval_path.exists() else None

    generate_learning_curves(train_rows)
    generate_success_plot(eval_rows, final_eval)
    generate_rollout_error_plot(final_eval)


if __name__ == "__main__":
    main()
