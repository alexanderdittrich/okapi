# Benchmarks

Four seeds per algorithm and task on one RTX 4080 Super. Evaluation returns use the
deterministic policy on 128 episodes, 20 times per run. Lines show the mean over seeds,
bands the range from the worst to the best seed. The x-axis is logarithmic because the
budgets differ by two orders of magnitude.

```{image} _static/benchmark_performance.png
:width: 100%
```

Final evaluation return, mean ± standard deviation over 4 seeds (environment steps in parentheses):

| Algorithm | CheetahRun | Go1JoystickFlatTerrain |
|---|---|---|
| PPO | 763.2 ± 86.9 (61M) | 30.7 ± 0.1 (200M) |
| PPO-GRU | 701.2 ± 143.8 (61M) | 27.2 ± 3.3 (200M) |
| SAC | 836.3 ± 19.6 (5M) | 13.5 ± 0.6 (10M) |
| SimbaV2 | 873.8 ± 59.0 (5M) | 30.4 ± 0.2 (10M) |
| BRO | 930.3 ± 1.8 (5M) | 7.3 ± 0.8 (10M) |
| DreamerV3 | 537.1 ± 56.5 (1M) | 2.6 ± 1.5 (1M) |

- **CheetahRun**: the off-policy methods beat PPO with 12x fewer steps; BRO is best and almost
  seed-independent. DreamerV3 is still improving at its 1M-step budget.
- **Go1**: SimbaV2 matches PPO with 20x fewer steps. SAC and BRO first learn to stand
  (return ~18), then their policies collapse after 2-7M steps and only partly recover;
  DreamerV3 does not learn the task within 1M steps. These runs use one shared setting
  per algorithm without per-task tuning.

## Settings

Reproduce with `uv run benchmarks/run.py` (and `benchmarks/plot.py`,
`benchmarks/throughput.py`). The exact settings are in `benchmarks/run.py`. Anything not
listed there uses the algorithm's defaults.

| | CheetahRun | Go1JoystickFlatTerrain |
|---|---|---|
| PPO | Playground's Brax PPO settings, 60M steps | Playground's Brax PPO settings, 200M steps |
| PPO-GRU | as PPO, 8 epochs, lr 3e-4, GRU 128 | as PPO, GRU 128 |
| SAC, SimbaV2, BRO | defaults (128 envs, batch 512, 8 updates/step), 5M steps | 1024 envs, batch 8192, 4 updates/step, γ = 0.97, 3-step returns, 10M steps |
| DreamerV3 | defaults (32 envs), 1M steps | defaults, γ = 0.97, 1M steps |
| TD-MPC2 | defaults (64 envs), 1M steps | defaults, γ = 0.97, 1M steps |

Go1 uses the privileged state for all critics (`value_obs_key=privileged_state`);
DreamerV3 and TD-MPC2 see only the policy observation.

## Throughput

Environment steps per second of training (collection + updates), measured one job at a time with `benchmarks/throughput.py` (the benchmark runs above shared the GPU, so their wall-clock times are not comparable):

| Algorithm | CheetahRun | Go1JoystickFlatTerrain |
|---|---|---|
| PPO | 662k | 702k |
| PPO-GRU | 298k | 247k |
| SAC | 53k | 133k |
| SimbaV2 | 6.1k | 8.7k |
| BRO | 11k | 16k |
| DreamerV3 | 0.6k | 0.6k |
| TD-MPC2 | 0.5k | 0.5k |

Multiply by the budget to get the training time, e.g. SAC on CheetahRun: 5M / 53k ≈ 1.6 min; SimbaV2: 5M / 6.1k ≈ 14 min; PPO: 60M / 662k ≈ 1.5 min.
