"""Penalty shoot-out curriculum: train a striker, then a keeper against it, then render.

uv run examples/penalties/train.py                       # both stages + GIF
uv run examples/penalties/train.py --stage keeper --striker runs/penalties-striker-...
uv run examples/penalties/train.py --stage render --striker runs/... --keeper runs/...
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import PenaltyKick  # noqa: E402

import okapi  # noqa: E402
from okapi.ppo import PPOConfig, train  # noqa: E402

BASE = PPOConfig(
    num_envs=4096,
    full_reset=True,  # the env keeps the shot target and flags in info
    episode_length=80,
    num_steps=20,
    num_minibatches=16,
    update_epochs=4,
    learning_rate=1e-4,
    gamma=0.99,
    ent_coef=1e-2,
    clip_coef=0.2,
    max_grad_norm=0.5,
    reward_scaling=0.1,
    actor_hidden_sizes=[256, 256, 256],
    critic_hidden_sizes=[512, 256, 256],
    num_evals=20,
    checkpoint=True,
)
STRIKER = dataclasses.replace(
    BASE,
    env_id="Penalties-Striker",
    run_name="penalties-striker",
    total_timesteps=20_000_000,
    seed=0,
)
KEEPER = dataclasses.replace(
    BASE, env_id="Penalties-Keeper", run_name="penalties-keeper", total_timesteps=45_000_000, seed=1
)


def frozen(run_dir: str):
    """A striker policy as a pure function of the striker observation."""
    policy = okapi.load_policy(run_dir)
    return lambda obs: policy(obs)[0]


def render(
    striker_dir: str, keeper_dir: str, output: str, shots: int = 4, keeper_step: int | None = None
) -> None:
    import imageio.v2 as imageio

    env = PenaltyKick("keeper", striker_policy=frozen(striker_dir))
    keeper = okapi.load_policy(keeper_dir, keeper_step)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    act = jax.jit(lambda obs: keeper(obs["keeper"])[0])
    frames, goals, saves = [], 0, 0
    for i in range(shots):
        state = reset(jax.random.key(10 + i))
        states = [state]
        for _ in range(env._config.episode_length):
            state = step(state, act(state.obs))
            states.append(state)
            goals += int(state.metrics["goal"])
            saves += int(state.metrics["save"])
        frames += env.render(states[::2], camera="broadcast", height=360, width=640)
        frames += [frames[-1]] * 5  # short pause between shots
    print(f"{shots} shots: {goals} goals, {saves} saves")
    imageio.mimsave(output, [np.asarray(f) for f in frames], fps=round(1 / env.dt / 2), loop=0)
    print(f"Saved {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "striker", "keeper", "render"], default="all")
    ap.add_argument("--striker", default=str(Path(STRIKER.log_dir) / STRIKER.run_name))
    ap.add_argument("--keeper", default=str(Path(KEEPER.log_dir) / KEEPER.run_name))
    ap.add_argument(
        "--keeper_step", type=int, default=None, help="Keeper checkpoint (default: latest)"
    )
    ap.add_argument("--output", default="docs/_static/penalties.gif")
    args = ap.parse_args()
    if args.stage in ("all", "striker"):
        train(STRIKER, env_factory=lambda: PenaltyKick("striker"))
    if args.stage in ("all", "keeper"):
        striker = frozen(args.striker)
        train(KEEPER, env_factory=lambda: PenaltyKick("keeper", striker_policy=striker))
    if args.stage in ("all", "render"):
        render(args.striker, args.keeper, args.output, keeper_step=args.keeper_step)


if __name__ == "__main__":
    main()
