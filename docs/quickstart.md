# Quickstart

## Train from the command line

`scripts/train.py` trains any algorithm on any MuJoCo Playground environment.
Options are [Hydra](https://hydra.cc) overrides:

```bash
uv run scripts/train.py algo=sac env_id=CheetahRun
uv run scripts/train.py algo=ppo env_id=Go1JoystickFlatTerrain value_obs_key=privileged_state
uv run scripts/train.py algo=simbav2 env_id=CheetahRun total_timesteps=2_000_000 checkpoint=true
```

Each run writes a directory `runs/<env>-<algo>-<time>-<id>/` containing

- `metadata.yaml`: algorithm, full config, environment config, package versions, git commit,
- `metrics.jsonl`: one JSON line per log (`eval/*`, `loss/*`, `train/*`, `time/*`),
- `checkpoints/`: Orbax checkpoints at every evaluation (with `checkpoint=true`).

Add `use_wandb=true` to log to Weights & Biases.

## Train from Python

```python
from okapi.sac import SACConfig, train

cfg = SACConfig(env_id="CheetahRun", total_timesteps=2_000_000)
policy = train(cfg)
```

`train` accepts an `env_factory` (a callable returning an `MjxEnv`) for custom
environments and a `log_callback(metrics, step)`, e.g. for live plots in a notebook.

## Use a policy

All algorithms return a {class}`~okapi.common.policy.Policy` that acts on raw
environment observations. Recurrent policies (PPO-GRU, DreamerV3) carry a state;
for feed-forward policies the carry is `()`:

```python
import jax
from mujoco_playground import registry

env = registry.load("CheetahRun")
state = jax.jit(env.reset)(jax.random.key(0))
step, act = jax.jit(env.step), jax.jit(policy.__call__)

carry, first = policy.init_carry(1), True
for _ in range(1000):
    obs = jax.tree.map(lambda x: x[None], state.obs)        # add a batch axis
    action, carry = act(obs, carry, jax.numpy.array([float(first)]))
    state = step(state, action[0])
    first = False
```

Load the policy of a finished run with

```python
import okapi
policy = okapi.load_policy("runs/cheetahrun-sac-...")   # latest checkpoint
```

## Look at a policy

```bash
# interactive viewer: ←/→ rollouts, ↓/↑ checkpoints, space pauses (runs on the CPU)
uv run scripts/view.py runs/<run_dir>

# video or GIF
uv run scripts/render.py runs/<run_dir> --output rollout.mp4
```
