#!/usr/bin/env python3
"""Benchmark Okapi PPO vs Brax PPO on MuJoCo Playground environments."""

import functools
import json
import re
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from mujoco_playground import registry, wrapper
from mujoco_playground._src import dm_control_suite as _dmc, manipulation as _manip
from mujoco_playground.config import (
    dm_control_suite_params,
    locomotion_params,
    manipulation_params,
)
from brax.training.agents.ppo import train as brax_ppo
from brax.training.agents.ppo import networks as ppo_networks

from okapi.playground.ppo import PPOConfig, train as okapi_train

import scienceplots

plt.style.use(["science", "nature"])

# ── Settings ──────────────────────────────────────────────────────────────────

WANDB_PROJECT = "okapi-benchmark"
DOCS_DIR = Path("docs")

# ── Benchmark parameters (edit these) ────────────────────────────────────────

ENVS = [
    "CheetahRun",
    "FishSwim",
    "HopperHop",
    "WalkerRun",
    "HumanoidRun",
    "LeapCubeReorient",
    # "Go1JoystickFlatTerrain",
    # "BarkourJoystick",
    # "G1JoystickFlatTerrain",
]
NUM_SEEDS = 10  # seeds used: 0, 1, ..., NUM_SEEDS-1
REPLOT = False  # True = regenerate plots from saved data without retraining

# ── Config helpers ────────────────────────────────────────────────────────────


def get_playground_config(env_name):
    if env_name in _manip.ALL_ENVS:
        return manipulation_params.brax_ppo_config(env_name)
    elif env_name in _dmc.ALL_ENVS:
        return dm_control_suite_params.brax_ppo_config(env_name)
    else:
        return locomotion_params.brax_ppo_config(env_name)


def brax_to_okapi(env_name, bc, seed):
    nf = getattr(bc, "network_factory", None)
    if nf is not None:
        actor_hidden = list(nf.policy_hidden_layer_sizes)
        critic_hidden = list(nf.value_hidden_layer_sizes)
        policy_obs_key = nf.policy_obs_key
        value_obs_key = nf.value_obs_key
    else:
        actor_hidden = [32, 32, 32, 32]
        critic_hidden = [256, 256, 256, 256, 256]
        policy_obs_key = "state"
        value_obs_key = "state"

    # Match Brax's rollout buffer size: batch_size * num_minibatches trajectories
    # each of unroll_length steps. For loco/manip the ratio is 1 (no change).
    segments_per_env = (bc.batch_size * bc.num_minibatches) // bc.num_envs
    num_steps = segments_per_env * bc.unroll_length

    # eval_frequency must be a multiple of log_frequency so both conditions
    # fire together and "training/episode_reward" appears at every eval.
    log_freq = 10
    total_iters = bc.num_timesteps // (bc.num_envs * num_steps)
    eval_freq = max(log_freq, (total_iters // bc.num_evals // log_freq) * log_freq)

    return PPOConfig(
        env_id=env_name,
        num_envs=bc.num_envs,
        total_timesteps=bc.num_timesteps,
        num_steps=num_steps,
        num_minibatches=bc.num_minibatches,
        update_epochs=bc.num_updates_per_batch,
        learning_rate=bc.learning_rate,
        gamma=bc.discounting,
        gae_lambda=0.95,
        clip_coef=getattr(bc, "clipping_epsilon", 0.3),
        ent_coef=bc.entropy_cost,
        max_grad_norm=getattr(bc, "max_grad_norm", None),
        reward_scaling=bc.reward_scaling,
        actor_hidden_sizes=actor_hidden,
        critic_hidden_sizes=critic_hidden,
        activation="swish",
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
        eval_frequency=eval_freq,
        log_frequency=log_freq,
        use_wandb=False,
        use_checkpointing=False,
        verbose=False,
        progress_bar=True,
        warmup_stats=False,
        seed=seed,
    )


# ── Training runners ──────────────────────────────────────────────────────────


def run_brax(env_name, bc, seed):
    env = registry.load(env_name)
    curve = []

    def progress_fn(num_steps, metrics):
        reward = float(metrics.get("eval/episode_reward", 0.0))
        curve.append((int(num_steps), reward))
        wandb.log({"episode_reward": reward}, step=int(num_steps))

    nf = getattr(bc, "network_factory", None)
    if nf is not None:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(nf.policy_hidden_layer_sizes),
            value_hidden_layer_sizes=tuple(nf.value_hidden_layer_sizes),
            policy_obs_key=nf.policy_obs_key,
            value_obs_key=nf.value_obs_key,
        )
    else:
        network_factory = ppo_networks.make_ppo_networks

    training_params = {k: v for k, v in dict(bc).items() if k != "network_factory"}
    num_eval_envs = training_params.pop("num_eval_envs", 128)

    t0 = time.time()
    brax_ppo.train(
        environment=env,
        **training_params,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        seed=seed,
        progress_fn=progress_fn,
    )
    return time.time() - t0, curve


def run_okapi(cfg):
    curve = []

    def log_callback(log, step):
        wandb.log(log, step=step)
        if "training/episode_reward" in log:
            curve.append((step, float(log["training/episode_reward"])))

    t0 = time.time()
    okapi_train(cfg, log_callback=log_callback)
    return time.time() - t0, curve


# ── Plotting ──────────────────────────────────────────────────────────────────


def save_plot(env_name, brax_seeds, okapi_seeds, brax_times, okapi_times):
    DOCS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(brax_seeds)
    for curves, label, color in [
        (brax_seeds, f"Brax  (n={n})", "tab:blue"),
        (
            okapi_seeds,
            f"Okapi (n={n})",
            "tab:orange",
        ),
    ]:
        xs = [s for s, _ in curves[0]]
        ys = np.array([[r for _, r in c] for c in curves])
        ax.plot(xs, ys.mean(axis=0), label=label, color=color)
        if n > 1:
            ax.fill_between(
                xs,
                ys.mean(0) - ys.std(0),
                ys.mean(0) + ys.std(0),
                alpha=0.15,
                color=color,
            )
    ax.set_title(env_name)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = DOCS_DIR / f"benchmark_{env_name.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def replot_from_data(data_path=DOCS_DIR / "benchmark_data.json"):
    with open(data_path) as f:
        data = json.load(f)
    for env_name, r in data.items():
        save_plot(
            env_name,
            [[tuple(p) for p in c] for c in r["brax_seeds"]],
            [[tuple(p) for p in c] for c in r["okapi_seeds"]],
            r["brax_times"],
            r["okapi_times"],
        )
    update_readme(data)


# ── README update ─────────────────────────────────────────────────────────────


def update_readme(results):
    readme = Path("README.md")
    text = readme.read_text()

    perf_imgs = "\n".join(
        f"#### {env}\n![{env}](docs/benchmark_{env.lower()}.png)\n" for env in results
    )
    rows = "\n".join(
        f"| {env} | {np.mean(r['brax_times']) / 3600:.2f} | {np.mean(r['okapi_times']) / 3600:.2f} |"
        for env, r in results.items()
    )
    sections = [
        ("### Performance comparison", f"### Performance comparison\n\n{perf_imgs}"),
        (
            "### Computing time",
            f"### Computing time\n\n| Environment | Brax (h) | Okapi (h) |\n|-------------|----------|----------|\n{rows}\n",
        ),
    ]
    for header, new_section in sections:
        text = re.sub(
            rf"{re.escape(header)}.*?(?=\n###|\n##|\Z)",
            new_section + "\n\n",
            text,
            flags=re.DOTALL,
        )

    readme.write_text(text)
    print("  Updated README.md")


# ── Main loop ─────────────────────────────────────────────────────────────────


def run_group(env_names, seeds):
    data_path = DOCS_DIR / "benchmark_data.json"
    DOCS_DIR.mkdir(exist_ok=True)
    existing = json.loads(data_path.read_text()) if data_path.exists() else {}
    results = {}

    for env_name in env_names:
        print(f"\n{'=' * 60}\n  {env_name}  (seeds={seeds})\n{'=' * 60}")
        bc = get_playground_config(env_name)
        brax_seeds, okapi_seeds, brax_times, okapi_times = [], [], [], []

        for seed in seeds:
            okapi_cfg = brax_to_okapi(env_name, bc, seed)

            print(f"\n[Brax] {env_name}  seed={seed}")
            wandb.init(
                project=WANDB_PROJECT,
                name=f"brax-{env_name}-seed-{seed}",
                entity="tensegrity",
                config={**dict(bc), "seed": seed},
                tags=["brax", env_name],
            )
            brax_time, brax_curve = run_brax(env_name, bc, seed)
            wandb.log({"time/total_hours": brax_time / 3600})
            wandb.finish()
            brax_times.append(brax_time)
            brax_seeds.append(brax_curve)
            print(f"  done in {brax_time / 3600:.2f} h")

            print(f"\n[Okapi] {env_name}  seed={seed}")
            wandb.init(
                project=WANDB_PROJECT,
                name=f"okapi-{env_name}-seed-{seed}",
                entity="tensegrity",
                config=vars(okapi_cfg),
                tags=["okapi", env_name],
            )
            okapi_time, okapi_curve = run_okapi(okapi_cfg)
            wandb.log({"time/total_hours": okapi_time / 3600})
            wandb.finish()
            okapi_times.append(okapi_time)
            okapi_seeds.append(okapi_curve)
            print(f"  done in {okapi_time / 3600:.2f} h")

        results[env_name] = {
            "brax_times": brax_times,
            "okapi_times": okapi_times,
            "brax_seeds": brax_seeds,
            "okapi_seeds": okapi_seeds,
        }
        save_plot(env_name, brax_seeds, okapi_seeds, brax_times, okapi_times)
        data_path.write_text(json.dumps({**existing, **results}, indent=2))

    return results


def main():
    if REPLOT:
        replot_from_data()
        return

    seeds = list(range(NUM_SEEDS))
    print(f"Envs  : {ENVS}")
    print(f"Seeds : {seeds}")
    results = run_group(ENVS, seeds)
    update_readme(results)


if __name__ == "__main__":
    main()
