# okapi

```{image} _static/okapi.png
:width: 160px
:align: right
```

**okapi** is a small collection of deep reinforcement learning algorithms written in
JAX and [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html) for
[MuJoCo Playground](https://playground.mujoco.org/) environments.

- **Single-file algorithms.** Everything specific to an algorithm (config, networks,
  losses, update and training loop) lives in one readable file.
- **Fully on the GPU.** Physics (MJX or MuJoCo Warp), inference, replay buffers and
  gradient updates run in `jit`-compiled JAX.
- **One interface.** Every algorithm has a `Config`, a `train` function that
  returns a {class}`~okapi.common.policy.Policy`, and a `load_policy` function.
  Options are named the same across algorithms.
- **Batteries included.** Hydra configs, W&B logging, Orbax checkpoints, a checkpoint
  viewer and a video renderer.

| Algorithm | Module | Type |
|---|---|---|
| PPO | {mod}`okapi.ppo` | on-policy |
| PPO-GRU | {mod}`okapi.ppo_gru` | on-policy, recurrent |
| SAC | {mod}`okapi.sac` | off-policy |
| SimbaV2 | {mod}`okapi.simbav2` | off-policy |
| BRO | {mod}`okapi.bro` | off-policy |
| DreamerV3 | {mod}`okapi.dreamerv3` | model-based, recurrent |
| TD-MPC2 | {mod}`okapi.tdmpc2` | model-based, planning |

```{toctree}
:maxdepth: 2
:caption: Guide

installation
quickstart
configuration
algorithms
custom_environments
```

```{toctree}
:maxdepth: 2
:caption: Results

benchmarks
showcases
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
