# Algorithms

All algorithms share the same structure. Each file contains, in order:

1. **Config**: a dataclass extending {class}`~okapi.common.config.TrainConfig`.
2. **Networks**: Flax NNX modules and `make_policy_apply`.
3. **Losses**: pure loss functions and the update step.
4. **Training**: `train(cfg, env_factory=None, log_callback=None, resume_from=None) -> Policy`.
5. **Loading**: `load_policy(run_dir, step=None) -> Policy`.

The shared, non-algorithmic parts live in `okapi.common`: environment creation,
the GPU replay buffer, running statistics, the evaluation/logging/checkpointing loop
and the {class}`~okapi.common.policy.Policy` interface.

## Which one should I use?

- **PPO** for cheap simulation and large budgets (hundreds of millions of steps in
  minutes). It is the most robust choice for locomotion with domain randomization.
- **SAC / BRO / SimbaV2** when samples matter: they need 10–50x fewer environment
  steps. SimbaV2 and BRO are slower per step but learn the most per step.
- **PPO-GRU / DreamerV3** when the task needs memory (partial observability).
  DreamerV3 is the most sample-efficient on some tasks but also the slowest per step.
- **TD-MPC2** for maximal sample efficiency on continuous control. It plans at every
  step, so acting is expensive.

## PPO

{mod}`okapi.ppo`, Schulman et al., 2017,
[arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

A close port of Brax PPO: tanh-squashed Gaussian policy, separate policy and value
MLPs, observation normalization, GAE that masks time-limit truncations, and values/GAE
recomputed with the current critic in every minibatch. `num_rollout_rounds` rollouts are
collected before each pass of `update_epochs` epochs. The defaults are MuJoCo
Playground's locomotion settings.

Key options: `num_envs`, `num_steps`, `num_rollout_rounds`, `num_minibatches`,
`update_epochs`, `learning_rate`, `gamma`, `ent_coef`, `reward_scaling`.

## PPO-GRU

{mod}`okapi.ppo_gru`

PPO with a GRU after the first hidden layer of actor and critic. Hidden states are reset
at episode starts. Updates replay each collected sequence from its initial hidden state
(truncated backpropagation through time over `num_steps`), and minibatches consist of
whole environment sequences.

Key options: as PPO, plus `gru_size`.

## SAC

{mod}`okapi.sac`, Haarnoja et al., 2018,
[arXiv:1812.05905](https://arxiv.org/abs/1812.05905).

Soft actor-critic in the massively parallel regime: every iteration, all environments
take one step into a GPU replay buffer, followed by `updates_per_step` gradient updates
on `batch_size` transitions. The algorithm uses a tanh-Gaussian policy, twin Q-critics with
Polyak-averaged targets, clipped double-Q n-step targets and a learned temperature.
The defaults follow MuJoCo Playground's Brax SAC settings.

Key options: `num_envs`, `batch_size`, `updates_per_step`, `learning_rate`, `gamma`,
`n_step`, `buffer_size`, `learning_starts`.

## SimbaV2

{mod}`okapi.simbav2`, Lee et al., 2025,
[arXiv:2502.15280](https://arxiv.org/abs/2502.15280).

SAC with *hyperspherical normalization*. Inputs and hidden activations live on the unit
sphere, residual blocks interpolate towards their output, and weights are projected back
onto the sphere after every update. The critic is categorical (C51, clipped double-Q),
and rewards are scaled by the running standard deviation of the discounted return.

Key options: as SAC, plus `critic_width`, `critic_blocks`, `actor_width`,
`actor_blocks`, `num_atoms`, `v_max`, `normalize_rewards`.

## BRO

{mod}`okapi.bro`, Nauman et al., 2024,
[arXiv:2405.16158](https://arxiv.org/abs/2405.16158).

*Bigger, Regularized, Optimistic*: SAC with BroNet actor and critics (LayerNorm residual
MLPs), AdamW weight decay, quantile critics with reduced pessimism (mean instead of
minimum of both critics), and optimistic exploration. A separate exploration actor
collects the data by maximizing `Q + optimism * |Q1 - Q2| / 2`, and a KL penalty keeps it
close to the conservative actor. BRO's periodic network resets are omitted.

Key options: as SAC, plus `weight_decay`, `critic_width`, `critic_blocks`,
`num_quantiles`, `optimism`, `kl_coef`.

## DreamerV3

{mod}`okapi.dreamerv3`, Hafner et al., 2023,
[arXiv:2301.04104](https://arxiv.org/abs/2301.04104).

A recurrent world model (RSSM with categorical latents) trained on sequences from the
replay buffer to reconstruct symlog observations and predict two-hot rewards and the
continue flag. An actor-critic is trained purely in imagination on `horizon`-step
latent rollouts, using percentile-normalized lambda-returns and a two-hot critic
regularized towards its slow copy. For continuous actions, returns are backpropagated
through the learned dynamics.

Key options: `num_envs`, `batch_size`, `seq_len`, `updates_per_step`, `deter`,
`stoch`, `classes`, `hidden`, `horizon`, `gamma`.

## TD-MPC2

{mod}`okapi.tdmpc2`, Hansen et al., 2024,
[arXiv:2310.16828](https://arxiv.org/abs/2310.16828).

A decoder-free world model: an encoder maps observations to SimNorm latents, and a
latent dynamics model, a two-hot reward head and an ensemble of two-hot Q-functions
are trained on short sequences with a latent consistency loss. Actions come from
MPPI planning over `horizon` steps, warm-started from the previous plan and seeded
with proposals from a policy prior. Each trajectory is scored by predicted rewards
plus the Q-value at the horizon.

Key options: `num_envs`, `batch_size`, `horizon`, `updates_per_step`, `latent_dim`,
`num_q`, `num_samples`, `iterations`, `mpc`.

```{note}
The ports adapt the published methods to many parallel environments and fixed
hyperparameters. The simplifications are listed in each module docstring. They aim
for faithful, readable implementations, not exact reproductions of the papers'
numbers.
```
