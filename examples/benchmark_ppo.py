#!/usr/bin/env python3
"""Benchmark Okapi PPO vs Brax PPO on MuJoCo Playground environments."""

import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

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
import jax

if not hasattr(jax, "device_put_replicated"):
    import numpy as _np
    import jax.numpy as _jnp

    def _device_put_replicated(x, devices):
        n = len(devices)
        mesh = jax.sharding.Mesh(_np.array(devices), ("devices",))
        # Partition along first axis so each device holds one slice of shape (1, ...)
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))
        stacked = jax.tree_util.tree_map(lambda v: _jnp.stack([v] * n), x)
        return jax.device_put(stacked, sharding)

    jax.device_put_replicated = _device_put_replicated

from okapi.playground.ppo import PPOConfig, train as okapi_train

# ── Settings ──────────────────────────────────────────────────────────────────

WANDB_PROJECT = "okapi-brax-benchmark"
DOCS_DIR = Path("docs")

# ── Benchmark parameters (edit these) ────────────────────────────────────────

ENVS = [
    # DM Control Suite
    "CheetahRun",
    "FishSwim",
    "HopperHop",
    "WalkerRun",
    "HumanoidRun",
    "CartpoleSwingup",
    # Locomotion
    "BarkourJoystick",
    "BerkeleyHumanoidJoystickFlatTerrain",
    "Go1JoystickFlatTerrain",
    "G1JoystickFlatTerrain",
    # Manipulation
    "AlohaHandOver",
    "LeapCubeReorient",
    "PandaPickCubeCartesian",
]
NUM_SEEDS = 8  # seeds used: 0, 1, ..., NUM_SEEDS-1

# Override total timesteps for envs that need longer training (scales num_evals proportionally)
TIMESTEP_OVERRIDES = {
    "HopperHop": 150_000_000,
    "HumanoidRun": 180_000_000,
}

RUN_OKAPI_MJX = True
RUN_OKAPI_WARP = True
RUN_BRAX_MJX = True
RUN_BRAX_WARP = False


# ── Config helpers ────────────────────────────────────────────────────────────


def get_playground_config(env_name):
    if env_name in _manip.ALL_ENVS:
        return manipulation_params.brax_ppo_config(env_name)
    elif env_name in _dmc.ALL_ENVS:
        return dm_control_suite_params.brax_ppo_config(env_name)
    else:
        return locomotion_params.brax_ppo_config(env_name)


def brax_to_okapi(env_name, bc, seed, impl="jax"):
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

    num_steps = bc.unroll_length
    num_minibatches = bc.num_minibatches

    # Brax collects (batch_size * num_minibatches // num_envs) rollout rounds per
    # training step before running num_updates_per_batch SGD epochs. Okapi collects
    # only 1 round per iteration, so scale down the epochs proportionally to keep
    # the same number of gradient updates per environment sample.
    brax_rollout_rounds = max(1, bc.batch_size * bc.num_minibatches // bc.num_envs)
    update_epochs = max(1, bc.num_updates_per_batch // brax_rollout_rounds)

    log_freq = 10
    total_iters = bc.num_timesteps // (bc.num_envs * num_steps)
    eval_freq = max(log_freq, (total_iters // bc.num_evals // log_freq) * log_freq)

    return PPOConfig(
        env_id=env_name,
        num_envs=bc.num_envs,
        total_timesteps=bc.num_timesteps,
        num_steps=num_steps,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
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
        env_overrides={"impl": impl},
    )


# ── Training runners ──────────────────────────────────────────────────────────


def run_brax(env_name, bc, seed, impl="jax"):
    env = registry.load(env_name, config_overrides={"impl": impl})
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


# ── Main loop ─────────────────────────────────────────────────────────────────


def run_group(env_names, seeds):
    data_path = DOCS_DIR / "benchmark_data.json"
    DOCS_DIR.mkdir(exist_ok=True)

    for env_name in env_names:
        print(f"\n{'=' * 60}\n  {env_name}  (seeds={seeds})\n{'=' * 60}")
        bc = get_playground_config(env_name)
        if env_name in TIMESTEP_OVERRIDES:
            scale = TIMESTEP_OVERRIDES[env_name] / bc.num_timesteps
            bc.num_timesteps = TIMESTEP_OVERRIDES[env_name]
            bc.num_evals = max(1, round(bc.num_evals * scale))

        def load():
            return json.loads(data_path.read_text()) if data_path.exists() else {}

        def save(key_times, key_seeds, times, curves):
            d = load()
            env_entry = d.get(env_name, {})
            env_entry["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            env_entry[key_times] = times
            env_entry[key_seeds] = curves
            d[env_name] = env_entry
            data_path.write_text(json.dumps(d, indent=2))

        if RUN_OKAPI_MJX:
            times, curves = [], []
            for seed in seeds:
                cfg = brax_to_okapi(env_name, bc, seed, impl="jax")
                wandb.init(
                    project=WANDB_PROJECT,
                    name=f"okapi-mjx-{env_name}-seed-{seed}",
                    config=vars(cfg),
                    tags=["okapi", "mjx", env_name],
                )
                t, curve = run_okapi(cfg)
                wandb.log({"time/total_hours": t / 3600})
                wandb.finish()
                times.append(t)
                curves.append(curve)
                print(f"\n[Okapi MJX] {env_name}  seed={seed}  {t / 3600:.2f} h")
            save("okapi_times", "okapi_seeds", times, curves)

        if RUN_OKAPI_WARP:
            times, curves = [], []
            for seed in seeds:
                cfg = brax_to_okapi(env_name, bc, seed, impl="warp")
                wandb.init(
                    project=WANDB_PROJECT,
                    name=f"okapi-warp-{env_name}-seed-{seed}",
                    config=vars(cfg),
                    tags=["okapi", "warp", env_name],
                )
                t, curve = run_okapi(cfg)
                wandb.log({"time/total_hours": t / 3600})
                wandb.finish()
                times.append(t)
                curves.append(curve)
                print(f"\n[Okapi Warp] {env_name}  seed={seed}  {t / 3600:.2f} h")
            save("okapi_warp_times", "okapi_warp_seeds", times, curves)

        if RUN_BRAX_MJX:
            times, curves = [], []
            for seed in seeds:
                wandb.init(
                    project=WANDB_PROJECT,
                    name=f"brax-mjx-{env_name}-seed-{seed}",
                    config={**dict(bc), "seed": seed},
                    tags=["brax", "mjx", env_name],
                )
                t, curve = run_brax(env_name, bc, seed, impl="jax")
                wandb.log({"time/total_hours": t / 3600})
                wandb.finish()
                times.append(t)
                curves.append(curve)
                print(f"\n[Brax MJX] {env_name}  seed={seed}  {t / 3600:.2f} h")
            save("brax_times", "brax_seeds", times, curves)

        if RUN_BRAX_WARP:
            times, curves = [], []
            for seed in seeds:
                wandb.init(
                    project=WANDB_PROJECT,
                    name=f"brax-warp-{env_name}-seed-{seed}",
                    config={**dict(bc), "seed": seed},
                    tags=["brax", "warp", env_name],
                )
                t, curve = run_brax(env_name, bc, seed, impl="warp")
                wandb.log({"time/total_hours": t / 3600})
                wandb.finish()
                times.append(t)
                curves.append(curve)
                print(f"\n[Brax Warp] {env_name}  seed={seed}  {t / 3600:.2f} h")
            save("brax_warp_times", "brax_warp_seeds", times, curves)


def main():
    seeds = list(range(NUM_SEEDS))
    print(f"Envs        : {ENVS}")
    print(f"Seeds       : {seeds}")
    print(f"Brax MJX    : {RUN_BRAX_MJX}")
    print(f"Brax Warp   : {RUN_BRAX_WARP}")
    print(f"Okapi MJX   : {RUN_OKAPI_MJX}")
    print(f"Okapi Warp  : {RUN_OKAPI_WARP}")
    run_group(ENVS, seeds)


if __name__ == "__main__":
    main()
