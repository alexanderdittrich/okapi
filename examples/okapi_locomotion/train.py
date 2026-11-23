"""Train the okapi to walk with PPO and render a GIF.

    uv run examples/okapi_locomotion/train.py                  # train + render
    uv run examples/okapi_locomotion/train.py --render runs/okapi-ppo-...   # render only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import OkapiJoystick  # noqa: E402

import okapi  # noqa: E402
from okapi.ppo import PPOConfig, train  # noqa: E402

CONFIG = PPOConfig(
    env_id="OkapiJoystick",
    num_envs=4096,
    total_timesteps=75_000_000,
    num_steps=20,
    num_minibatches=32,
    update_epochs=4,
    learning_rate=3e-4,
    gamma=0.97,
    ent_coef=5e-3,
    clip_coef=0.2,
    max_grad_norm=1.0,
    actor_hidden_sizes=[512, 256, 128],
    critic_hidden_sizes=[512, 256, 128],
    value_obs_key="privileged_state",
    num_evals=20,
    checkpoint=True,
    seed=1,
)

# (seconds, [vx, vy, yaw rate]): walk, speed up, turn left, turn right
SCRIPT = [(2.5, [0.6, 0.0, 0.0]), (2.5, [1.2, 0.0, 0.0]), (2.5, [0.8, 0.0, 0.6]),
          (2.5, [0.8, 0.0, -0.6])]


def render(run_dir: str, output: str) -> None:
    import imageio.v2 as imageio
    import mujoco

    env = OkapiJoystick()
    policy = okapi.load_policy(run_dir)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    act = jax.jit(lambda obs: policy(jax.tree.map(lambda x: x[None], obs))[0][0])
    state = reset(jax.random.key(3))
    # start at the origin facing +x, so the camera sees the okapi from the side
    data = state.data
    state = state.replace(data=data.replace(qpos=data.qpos.at[:7].set(env._init_q[:7]),
                                            qvel=jnp.zeros_like(data.qvel)))
    states = []
    for seconds, command in SCRIPT:
        for _ in range(int(seconds / env.dt)):
            state.info["command"] = jnp.array(command)
            state.info["steps_until_next_cmd"] = jnp.array(10_000, jnp.int32)
            state = step(state, act(state.obs))
            states.append(state)
    option = mujoco.MjvOption()
    every = 3  # ~17 fps keeps the GIF small
    frames = env.render(states[::every], camera="cinematic", height=270, width=480,
                        scene_option=option)
    imageio.mimsave(output, [np.asarray(f) for f in frames], fps=round(1 / env.dt / every),
                    loop=0)
    print(f"Saved {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=None, help="Run directory to render (skips training)")
    ap.add_argument("--output", default="docs/_static/okapi_locomotion.gif")
    args = ap.parse_args()
    run_dir = args.render
    if run_dir is None:
        train(CONFIG, env_factory=lambda: OkapiJoystick(scenery=False))  # decoration only slows training
        run_dir = str(sorted(Path(CONFIG.log_dir).glob("okapijoystick-ppo-*"))[-1])
    render(run_dir, args.output)


if __name__ == "__main__":
    main()
