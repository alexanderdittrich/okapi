#!/usr/bin/env python3
"""Render the policy of a run (any okapi algorithm) to a video or GIF.

Usage:
    uv run scripts/render.py runs/<run_dir> --output rollout.mp4
    uv run scripts/render.py runs/<run_dir> --output rollout.gif --steps 500 --camera side
    uv run scripts/render.py runs/<run_dir> --env examples/okapi_locomotion/env.py:OkapiJoystick
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir")
    ap.add_argument("--output", default="rollout.mp4", help=".mp4 or .gif")
    ap.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    ap.add_argument("--steps", type=int, default=1000, help="Environment steps to render")
    ap.add_argument("--render_every", type=int, default=2, help="Render every n-th step")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--camera", default=None, help="Camera name (default: env default)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", default=None, help="Custom env factory 'file.py:Name'")
    args = ap.parse_args()

    meta = checkpoint.load_metadata(args.run_dir)
    cfg = TrainConfig.from_dict(
        {k: v for k, v in meta["config"].items() if k in TrainConfig.__dataclass_fields__}
    )
    env = envs.load_env(cfg, load_factory(args.env) if args.env else None)
    policy = okapi.load_policy(args.run_dir, args.step)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    batch = lambda x: jax.tree.map(lambda v: v[None], x)

    @jax.jit
    def act(params, carry, obs, first, key):
        action, carry = policy.apply(
            params, carry, batch(obs), first[None], key, not args.stochastic
        )
        return action[0], carry

    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)
    state = reset(reset_key)
    carry, first = policy.init_carry(1), jnp.ones(())
    states, ret = [state], 0.0
    for _ in range(args.steps):
        key, act_key = jax.random.split(key)
        action, carry = act(policy.params, carry, state.obs, first, act_key)
        state = step(state, action)
        states.append(state)
        ret += float(state.reward)
        first = jnp.zeros(())
        if float(state.done) > 0:
            break
    print(f"Episode: {len(states) - 1} steps, return {ret:.1f}")

    kwargs = {"width": args.width, "height": args.height}
    if args.camera is not None:
        kwargs["camera"] = args.camera
    frames = [np.asarray(f) for f in env.render(states[:: args.render_every], **kwargs)]
    fps = round(1.0 / env.dt / args.render_every)
    if args.output.endswith(".gif"):
        import imageio.v2 as imageio

        imageio.mimsave(args.output, frames, fps=fps, loop=0)
    else:
        mediapy.write_video(args.output, frames, fps=fps)
    print(f"Saved {args.output} ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
