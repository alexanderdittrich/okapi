#!/usr/bin/env python3
"""Summary benchmark plot: all environments in a single figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "nature"])

DATA_PATH = Path("docs/benchmark_data.json")
OUT_PATH = Path("docs/benchmark_performance.png")

ENVS = [
    "CheetahRun",
    "FishSwim",
    # "HopperHop",
    "WalkerRun",
    "HumanoidRun",
    "CartpoleSwingup",
    "BarkourJoystick",
    "BerkeleyHumanoidJoystickFlatTerrain",
    "Go1JoystickFlatTerrain",
    "G1JoystickFlatTerrain",
    "AlohaHandOver",
    "LeapCubeReorient",
    "PandaPickCubeCartesian",
]
TITLES = {
    "CheetahRun": "Cheetah Run",
    "FishSwim": "Fish Swim",
    # "HopperHop": "Hopper Hop",
    "WalkerRun": "Walker Run",
    "HumanoidRun": "Humanoid Run",
    "BarkourJoystick": "Barkour \nJoystick",
    "CartpoleSwingup": "Cartpole \nSwingup",
    "BerkeleyHumanoidJoystickFlatTerrain": "Berkeley \nJoystick",
    "Go1JoystickFlatTerrain": "Go1 Joystick",
    "G1JoystickFlatTerrain": "G1 Joystick",
    "AlohaHandOver": "Aloha Hand Over",
    "LeapCubeReorient": "Leap Cube \nReorient",
    "PandaPickCubeCartesian": "Panda Pick Cube \nCartesian",
}

BRAX_COLOR = "#f37171ff"
OKAPI_COLOR = "#c4956aff"
OKAPI_WARP_COLOR = "#6a9ec4ff"


def load_curves(seed_list):
    """Return (xs, ys) dropping any seed that contains NaN rewards."""
    valid = []
    for seed in seed_list:
        rewards = [r for _, r in seed]
        if not any(np.isnan(r) for r in rewards):
            valid.append(seed)
    xs = np.array([s for s, _ in valid[0]])
    ys = np.array([[r for _, r in c] for c in valid])
    return xs, ys, len(valid)


def plot_env(ax, env, data):
    r = data[env]

    brax_xs, brax_ys, n_brax = load_curves(r["brax_seeds"])
    okapi_xs, okapi_ys, n_okapi = load_curves(r["okapi_seeds"])

    series = [
        (brax_xs, brax_ys, n_brax, f"Brax MJX (n={n_brax})", BRAX_COLOR),
        (okapi_xs, okapi_ys, n_okapi, f"Okapi MJX (n={n_okapi})", OKAPI_COLOR),
    ]

    if "okapi_warp_seeds" in r:
        warp_xs, warp_ys, n_warp = load_curves(r["okapi_warp_seeds"])
        series.append((warp_xs, warp_ys, n_warp, f"Okapi Warp (n={n_warp})", OKAPI_WARP_COLOR))

    for xs, ys, n, label, color in series:
        mean = ys.mean(0)
        std = ys.std(0)
        ax.plot(xs / 1e6, mean, label=label, color=color, linewidth=1.0, marker="o")
        ax.fill_between(xs / 1e6, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_title(TITLES[env])
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Episode reward")
    ax.legend(fontsize=5, frameon=True)
    ax.grid(alpha=0.3)


def plot_runtime(ax, data):
    envs = list(data.keys())
    titles = [TITLES.get(e, e) for e in envs]
    x = np.arange(len(envs))
    width = 0.25

    brax_means = np.array([np.mean(data[e]["brax_times"]) / 60 for e in envs])
    brax_stds = np.array([np.std(data[e]["brax_times"]) / 60 for e in envs])
    okapi_means = np.array([np.mean(data[e]["okapi_times"]) / 60 for e in envs])
    okapi_stds = np.array([np.std(data[e]["okapi_times"]) / 60 for e in envs])

    has_warp = any("okapi_warp_times" in data[e] for e in envs)
    n_bars = 3 if has_warp else 2
    offsets = np.linspace(-width * (n_bars - 1) / 2, width * (n_bars - 1) / 2, n_bars)

    ax.bar(
        x + offsets[0],
        brax_means,
        width,
        yerr=brax_stds,
        label="Brax MJX",
        color=BRAX_COLOR,
        alpha=0.85,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    ax.bar(
        x + offsets[1],
        okapi_means,
        width,
        yerr=okapi_stds,
        label="Okapi MJX",
        color=OKAPI_COLOR,
        alpha=0.85,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    if has_warp:
        warp_means = np.array([np.mean(data[e]["okapi_warp_times"]) / 60 if "okapi_warp_times" in data[e] else 0.0 for e in envs])
        warp_stds = np.array([np.std(data[e]["okapi_warp_times"]) / 60 if "okapi_warp_times" in data[e] else 0.0 for e in envs])
        ax.bar(
            x + offsets[2],
            warp_means,
            width,
            yerr=warp_stds,
            label="Okapi Warp",
            color=OKAPI_WARP_COLOR,
            alpha=0.85,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=0, ha="center")
    ax.set_ylabel("Wall time (min)")
    # ax.set_title("Training time")
    ax.legend(fontsize=5, frameon=True)
    ax.grid(alpha=0.3, axis="y")


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    envs = [e for e in ENVS if e in data]

    # ── Learning curves ───────────────────────────────────────────────────
    ncols = 4
    nrows = (len(envs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2))
    axes = axes.flatten()

    for i, env in enumerate(envs):
        plot_env(axes[i], env, data)

    for j in range(len(envs), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=500, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")

    # ── Runtime comparison ────────────────────────────────────────────────
    runtime_w = min(len(envs) * 1.2, 10.0)
    fig2, ax2 = plt.subplots(figsize=(runtime_w, 2.5))
    plot_runtime(ax2, {e: data[e] for e in envs})
    fig2.tight_layout()
    runtime_path = OUT_PATH.parent / "benchmark_runtime.png"
    fig2.savefig(runtime_path, dpi=500, bbox_inches="tight")
    print(f"Saved {runtime_path}")


if __name__ == "__main__":
    main()
