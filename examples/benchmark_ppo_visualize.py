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
    "HopperHop",
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
    "HopperHop": "Hopper Hop",
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
    if not seed_list:
        return np.array([]), np.empty((0, 0)), 0

    valid = []
    for seed in seed_list:
        rewards = [r for _, r in seed]
        if not any(np.isnan(r) for r in rewards):
            valid.append(seed)

    if not valid:
        return np.array([]), np.empty((0, 0)), 0

    xs = np.array([s for s, _ in valid[0]])
    ys = np.array([[r for _, r in c] for c in valid])
    return xs, ys, len(valid)


def plot_env(ax, env, data):
    r = data[env]

    series = []

    if "brax_seeds" in r:
        brax_xs, brax_ys, n_brax = load_curves(r["brax_seeds"])
        if n_brax > 0:
            series.append((brax_xs, brax_ys, n_brax, f"Brax MJX (n={n_brax})", BRAX_COLOR))

    if "okapi_seeds" in r:
        okapi_xs, okapi_ys, n_okapi = load_curves(r["okapi_seeds"])
        if n_okapi > 0:
            series.append((okapi_xs, okapi_ys, n_okapi, f"Okapi MJX (n={n_okapi})", OKAPI_COLOR))

    if "okapi_warp_seeds" in r:
        warp_xs, warp_ys, n_warp = load_curves(r["okapi_warp_seeds"])
        if n_warp > 0:
            series.append((warp_xs, warp_ys, n_warp, f"Okapi Warp (n={n_warp})", OKAPI_WARP_COLOR))

    for xs, ys, n, label, color in series:
        mean = ys.mean(0)
        std = ys.std(0)
        ax.plot(xs / 1e6, mean, label=label, color=color, linewidth=1.0, marker="o")
        ax.fill_between(xs / 1e6, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_title(TITLES[env])
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Episode reward")
    if series:
        ax.legend(fontsize=5, frameon=True)
    else:
        ax.text(0.5, 0.5, "No valid curves", transform=ax.transAxes, ha="center", va="center", fontsize=7)
    ax.grid(alpha=0.3)


def plot_runtime(ax, data):
    envs = list(data.keys())
    titles = [TITLES.get(e, e) for e in envs]
    x = np.arange(len(envs))
    width = 0.25

    def _mean_std_minutes(times):
        if not times:
            return np.nan, np.nan
        arr = np.asarray(times, dtype=float)
        return float(np.mean(arr) / 60), float(np.std(arr) / 60)

    brax_stats = [_mean_std_minutes(data[e].get("brax_times")) for e in envs]
    brax_means = np.array([m for m, _ in brax_stats])
    brax_stds = np.array([s for _, s in brax_stats])

    okapi_stats = [_mean_std_minutes(data[e].get("okapi_times")) for e in envs]
    okapi_means = np.array([m for m, _ in okapi_stats])
    okapi_stds = np.array([s for _, s in okapi_stats])

    has_warp = any("okapi_warp_times" in data[e] for e in envs)
    series = []
    if np.any(~np.isnan(brax_means)):
        series.append(("Brax MJX", BRAX_COLOR, brax_means, brax_stds))
    if np.any(~np.isnan(okapi_means)):
        series.append(("Okapi MJX", OKAPI_COLOR, okapi_means, okapi_stds))

    if has_warp:
        warp_stats = [_mean_std_minutes(data[e].get("okapi_warp_times")) for e in envs]
        warp_means = np.array([m for m, _ in warp_stats])
        warp_stds = np.array([s for _, s in warp_stats])
        if np.any(~np.isnan(warp_means)):
            series.append(("Okapi Warp", OKAPI_WARP_COLOR, warp_means, warp_stds))

    n_bars = len(series)
    if n_bars == 0:
        ax.text(0.5, 0.5, "No runtime data", transform=ax.transAxes, ha="center", va="center", fontsize=7)
        return

    offsets = np.linspace(-width * (n_bars - 1) / 2, width * (n_bars - 1) / 2, n_bars)

    for i, (label, color, means, stds) in enumerate(series):
        ax.bar(
            x + offsets[i],
            means,
            width,
            yerr=stds,
            label=label,
            color=color,
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
