# Custom environments

Any `mujoco_playground` `MjxEnv` works. Pass a factory instead of `env_id`:

```python
from okapi.ppo import PPOConfig, train
from my_env import MyEnv          # subclass of mujoco_playground._src.mjx_env.MjxEnv

cfg = PPOConfig(env_id="MyEnv", total_timesteps=50_000_000)   # env_id only names the run
policy = train(cfg, env_factory=MyEnv)
```

An environment needs `reset(rng) -> State`, `step(state, action) -> State`,
`xml_path`, `action_size`, `mj_model` and `mjx_model` (see the Playground docs or
`examples/` in this repository). Some tips:

- Return a dict observation such as `{"state": ..., "privileged_state": ...}` for
  asymmetric actor-critic and set `value_obs_key="privileged_state"`.
- Put reward terms into `state.metrics` (e.g. `reward/tracking`). Evaluation logs
  their per-episode sums as `eval/reward/tracking`.
- Signal failures with `done=1`. Time limits are added by okapi (`episode_length`), so
  the environment does not need to truncate itself.

Policies can be frozen into other environments, e.g. as an opponent:

```python
striker = okapi.load_policy("runs/penalties-striker-...")
action, _ = striker(obs)        # jit/vmap-compatible
```

Use `--env path/to/file.py:Factory` with `scripts/view.py` and `scripts/render.py`
for custom environments.
