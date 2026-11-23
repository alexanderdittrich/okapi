#!/usr/bin/env python3
"""Plot ``docs/benchmark_results.json`` (written by ``run.py``) as learning curves.

    uv run benchmarks/plot.py

One panel per environment: mean evaluation return over seeds (line) and the range
between the worst and best seed (band) against environment steps (log scale, since
the algorithms' budgets differ by two orders of magnitude).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "benchmark_results.json"
OUTPUT = ROOT / "docs" / "_static" / "benchmark_performance.png"

# Fixed categorical order (validated palette); recurrent methods dashed as a second cue.
ALGOS = {
    "ppo": ("PPO", "#2a78d6", "-"),
    "ppo_gru": ("PPO-GRU", "#eb6834", "--"),
    "sac": ("SAC", "#1baf7a", "-"),
    "simbav2": ("SimbaV2", "#eda100", "-"),
    "bro": ("BRO", "#e87ba4", "-"),
    "dreamerv3": ("DreamerV3", "#008300", "--"),
    "tdmpc2": ("TD-MPC2", "#6250d6", "-."),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def curves(runs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Align seeds on the shortest common evaluation grid: ``(steps, rewards[seed, t])``."""
    n = min(len(r["steps"]) for r in runs.values())
    steps = np.array(next(iter(runs.values()))["steps"][:n], float)
    return steps, np.array([r["reward"][:n] for r in runs.values()])


def main() -> None:
    results = json.loads(RESULTS.read_text())
    envs = list(results)
    fig, axes = plt.subplots(1, len(envs), figsize=(6.2 * len(envs), 4.4), squeeze=False)
    for ax, env_id in zip(axes[0], envs):
        for algo, (label, color, style) in ALGOS.items():
            runs = results[env_id].get(algo)
            if not runs:
                continue
            steps, rewards = curves(runs)
            steps = np.maximum(steps, 1e4)
            mean = rewards.mean(0)
            ax.fill_between(steps, rewards.min(0), rewards.max(0), color=color, alpha=0.15, lw=0)
            ax.plot(steps, mean, color=color, ls=style, lw=2, label=f"{label} (n={len(rewards)})")
            ax.annotate(label, (steps[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_title(env_id, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("environment steps", color=MUTED)
        ax.set_ylabel("evaluation return", color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.margins(x=0.12)
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = len(labels) if len(envs) > 1 else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUTPUT, dpi=160, facecolor="#fcfcfb")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
