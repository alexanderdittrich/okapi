#!/usr/bin/env python3
"""
Browse the checkpoints of a run in the MuJoCo passive viewer (any okapi algorithm).

Parts of this code are adapted from https://github.com/Andrew-Luo1/rscope
(MIT License, Copyright (c) 2025 Jing Yuan Luo).

Usage:
    uv run scripts/view.py runs/<run_dir>
    uv run scripts/view.py runs/<run_dir> --evals 5 --stochastic --last
    uv run scripts/view.py runs/<run_dir> --env examples/okapi_locomotion/env.py:OkapiJoystick

Rollouts are computed on the CPU with the MJX backend (``--gpu`` to use the GPU),
so the viewer can run next to an ongoing training run.

Controls:
    ← / →  : previous / next evaluation rollout
    ↓ / ↑  : earlier / later checkpoint
    space  : pause / resume
"""

from __future__ import annotations

import argparse
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MUJOCO_GL"] = "glfw"

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import okapi
from okapi.common import checkpoint, envs
from okapi.common.config import TrainConfig


def load_factory(spec: str):
    """``path/to/file.py:Name`` → the callable ``Name`` defined in that file."""
    path, name = spec.rsplit(":", 1)
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(Path(path).parent.resolve()))
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Run directory (contains metadata.yaml and checkpoints/)")
    ap.add_argument("--evals", type=int, default=5, help="Rollouts per checkpoint")
    ap.add_argument("--stochastic", action="store_true", help="Sample actions")
    ap.add_argument("--last", action="store_true", help="Only load the latest checkpoint")
    ap.add_argument("--env", default=None, help="Custom env factory 'file.py:Name'")
    ap.add_argument("--gpu", action="store_true", help="Compute rollouts on the GPU")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    meta = checkpoint.load_metadata(run_dir)
    algo = okapi.get_algorithm(meta["algorithm"])
    cfg = TrainConfig.from_dict({k: v for k, v in meta["config"].items()
                                 if k in TrainConfig.__dataclass_fields__})
    if not args.gpu and args.env is None:
        cfg.impl = "jax"  # MuJoCo Warp needs a GPU
    factory = load_factory(args.env) if args.env else None
    env = envs.make_env(cfg, envs.load_env(cfg, factory), args.evals, jax.random.key(0))
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    print(f"{meta['algorithm']} on {cfg.env_id}  |  run: {run_dir.name}")

    steps = checkpoint.steps(run_dir)
    if not steps:
        sys.exit("No checkpoints found (train with checkpoint=true).")
    steps = steps[-1:] if args.last else steps
    print(f"Found {len(steps)} checkpoints: {steps}")

    n_steps = cfg.episode_length // cfg.action_repeat
    policy0 = algo.load_policy(run_dir, steps[0])

    @jax.jit
    def rollout(params, key):
        reset_key, key = jax.random.split(key)
        state = envs.reset(env, args.evals, reset_key)

        def step(carry, t):
            state, policy_carry, key = carry
            key, act_key = jax.random.split(key)
            first = jnp.full(args.evals, t == 0, jnp.float32)
            action, policy_carry = policy0.apply(params, policy_carry, state.obs, first, act_key,
                                                 not args.stochastic)
            out = (state.data.qpos, state.data.qvel, state.done)
            return (env.step(state, action), policy_carry, key), out

        _, (qpos, qvel, done) = jax.lax.scan(
            step, (state, policy0.init_carry(args.evals), key), jnp.arange(n_steps))
        return qpos, qvel, done

    rollouts: list[tuple[int, list]] = []
    key = jax.random.key(0)
    for i, step in enumerate(steps):
        policy = algo.load_policy(run_dir, step)
        key, rollout_key = jax.random.split(key)
        qpos, qvel, done = map(np.asarray, rollout(policy.params, rollout_key))
        evals = []
        for e in range(args.evals):
            ends = np.where(done[:, e] > 0)[0]
            end = int(ends[0]) + 1 if len(ends) else len(qpos)
            evals.append((qpos[:end, e], qvel[:end, e]))
        rollouts.append((step, evals))
        print(f"  [{i + 1}/{len(steps)}] step {step:,}: lengths {[len(q) for q, _ in evals]}")

    ci, ei, fi, paused = [len(steps) - 1], [0], [0], [False]

    def key_callback(keycode: int) -> None:
        right, left, down, up, space = 262, 263, 264, 265, 32
        if keycode in (right, left):
            ei[0] = (ei[0] + (1 if keycode == right else -1)) % args.evals
            fi[0] = 0
        elif keycode in (up, down):
            ci[0] = min(max(ci[0] + (1 if keycode == up else -1), 0), len(steps) - 1)
            ei[0], fi[0] = 0, 0
        elif keycode == space:
            paused[0] = not paused[0]

    print("\nControls: ← → eval  ↓ ↑ checkpoint  space pause\n")
    dt = float(getattr(env, "dt", mj_model.opt.timestep))
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            gstep, evals = rollouts[ci[0]]
            qpos, qvel = evals[ei[0]]
            if not paused[0]:
                t = fi[0] % len(qpos)
                mj_data.qpos[:], mj_data.qvel[:] = qpos[t], qvel[t]
                mujoco.mj_forward(mj_model, mj_data)
                fi[0] = (t + 1) % len(qpos)
            viewer.sync()
            left = "Checkpoint\nEval\nFrame" + ("\n[PAUSED]" if paused[0] else "")
            right = (f"{ci[0] + 1}/{len(steps)}  (step {gstep:,})\n"
                     f"{ei[0] + 1}/{args.evals}\n{fi[0]}/{len(qpos)}")
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
            time.sleep(dt)


if __name__ == "__main__":
    main()
