# Configuration

Every algorithm config extends {class}`~okapi.common.config.TrainConfig`, so the
options below mean the same for all algorithms. Algorithm hyperparameters are
documented in the [API reference](api.md) and in `configs/algo/<name>.yaml`.

| Option | Default | Meaning |
|---|---|---|
| `env_id` | `CheetahRun` | MuJoCo Playground environment |
| `impl` | `null` | physics backend: `jax` (MJX), `warp`, or the environment default |
| `num_envs` | per algorithm | parallel training environments |
| `episode_length` | `1000` | steps before truncation |
| `action_repeat` | `1` | physics steps per agent step |
| `policy_obs_key` | `state` | observation entry of the policy |
| `value_obs_key` | `state` | observation entry of value functions (e.g. `privileged_state`) |
| `domain_randomization` | `false` | use the registry's domain randomizer |
| `full_reset` | `false` | auto-reset calls `env.reset` (for envs with per-episode state in `info`) |
| `env_overrides` | `{}` | extra `config_overrides` for `registry.load` |
| `total_timesteps` | per algorithm | training budget in environment steps |
| `seed` | `0` | random seed |
| `num_evals` | `20` | evaluations during training (plus one at step 0) |
| `num_eval_envs` | `128` | episodes per evaluation (deterministic policy) |
| `num_logs` | `100` | training-metric log points |
| `log_dir`, `run_name` | `runs`, auto | where the run directory is created |
| `use_wandb`, `wandb_project` | `false`, `okapi` | Weights & Biases logging |
| `checkpoint` | `false` | save a checkpoint at every evaluation |
| `verbose` | `true` | print one line per log |

## Hydra

`configs/train.yaml` holds the shared options and selects an algorithm file from
`configs/algo/`. The algorithm file overrides the shared options, and the command line
overrides both:

```bash
uv run scripts/train.py algo=bro env_id=CheetahRun learning_rate=1e-4 seed=3
```

Sweeps use Hydra's multirun:

```bash
uv run scripts/train.py -m algo=sac,simbav2,bro seed=0,1,2
```

## Observations

Dict observations (e.g. `{"state": ..., "privileged_state": ...}`) let the policy and
the critic see different inputs (asymmetric actor-critic). Flat array observations are
treated as `{"state": obs}`, so the keys can stay at their defaults. Observations are
normalized with running statistics; DreamerV3 uses `symlog` instead.

## Episode ends

On-policy algorithms (PPO, PPO-GRU) use Brax's auto-reset and handle time limits
with Brax's truncation-aware GAE. Replay-based algorithms (SAC, SimbaV2, BRO,
DreamerV3) store the final observation of every episode before it is reset (see
{func}`~okapi.common.envs.step_or_reset`), so time-limit truncations bootstrap from
the true final state and terminations do not.
