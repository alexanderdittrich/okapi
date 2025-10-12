# rlx
<a><img src="docs/rlx.png" width="240" align="right"/></a>
Baseline RL algorithms implemented in JAX Flax NNX. While JAX can provide a significant speed up to computation, it is often not intuitive to use. This is in particular the case for Flax Linen API. This small framework provides a collection of reimplemented algorithms in the new API - Flax NNX. The single-file orientation of this repo is heavily inspired by the design philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl).

**Features:**
- (Almost) single-file implementation of DRL baselines with Flax NNX.
- Checkpointing with [orbax-checkpoint](https://orbax.readthedocs.io/)
- Config management with [hydra](https://hydra.cc)
- Logging with [wandb](https://wandb.ai/)
- Dependency management with [uv](https://docs.astral.sh/uv/) 

**Disclaimer:** This repository is not actively developed and will not provide any further support or documentation. It is intended for hobbyists and as a look-up for the usage of Flax NNX in DRL. For serious research, we recommend more mature frameworks e.g. [stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [BRAX](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl).


## Getting started
Setup training environment with `uv`.
```bash
git clone git@github.com:alexanderdittrich/rlx.git
cd rlx 
uv sync
```

Run training:
```bash
uv run src/rlx/ppo.py env_id=CartPole-v1 num_train_steps=500000
```

## Benchmarks
- [ ] Discrete environments: `Acrobot-v1`, `CartPole-v1`, `MountainCar-v0`, `LunarLander-v3`
- [ ] Continuous environments: `Pendulum-v1`,`BipedalWalker-v3`,`HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`
- [ ] Vision: `CarRacing-v3`, `ALE/SpaceInvaders-v5`, `ALE/Breakout-v5`


## Roadmap:
- [ ] Playground MJX API. 
- [ ] `nnx.scan`-integration.
- [ ] Extensive benchmarking.
- [ ] External learn-API.
- [ ] Checkpointing and replay.
- [ ] Integration of further algorithms -> Rainbow DQN, SAC, DreamerV3.
- [ ] Add some custom performance showcase -> OP3 football / humanoid football.