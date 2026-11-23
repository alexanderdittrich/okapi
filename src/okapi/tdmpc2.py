"""TD-MPC2 (Hansen et al., 2024): model-predictive control in a learned latent space.

An implicit (decoder-free) world model and a planner:

* **World model**: an encoder maps observations to latents normalized with
  SimNorm (softmax over groups of ``simnorm_dim``); a latent dynamics model, a
  two-hot reward head and an ensemble of ``num_q`` two-hot Q-functions are
  trained jointly on ``horizon``-step sequences with a latent consistency loss
  (predicted vs. encoded next latent), reward and TD losses, weighted by
  ``rho ** t``. TD targets use the minimum of two random target Q-functions.
* **Policy prior**: a tanh-Gaussian maximizing the (scale-normalized) Q-value
  of the latents plus a small entropy bonus.
* **Planning**: MPPI over ``horizon`` steps with ``num_samples`` action
  sequences (``num_pi_trajs`` of them from the policy prior), ``iterations``
  refinements of a Gaussian towards the top ``num_elites``, warm-started from
  the previous step's plan. Trajectories are scored by predicted rewards plus
  the discounted Q-value at the horizon.

Every iteration, all environments plan and take one step into a sequence
replay buffer, followed by ``updates_per_step`` updates. The returned policy
plans at every step (``mpc=False`` acts with the policy prior instead).

Simplifications: no Q-function dropout, no termination head (terminations are
read from the replay buffer instead), per-module gradient clipping, and sizes
chosen for parallel environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from okapi.common import checkpoint, envs, replay, running_statistics
from okapi.common.config import TrainConfig
from okapi.common.logger import LogCallback
from okapi.common.networks import MLP
from okapi.common.policy import Policy
from okapi.common.training import setup, train_loop

NAME = "tdmpc2"
sg = jax.lax.stop_gradient


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class TDMPC2Config(TrainConfig):
    """TD-MPC2 hyperparameters.

    Attributes:
        buffer_size: Replay capacity in steps (split evenly over environments).
        batch_size: Sequences per update.
        horizon: Model rollout / planning horizon.
        updates_per_step: Updates per environment step (of all environments).
        learning_starts: Random-action steps collected before learning.
        learning_rate: Adam learning rate of world model and policy.
        encoder_lr_scale: Learning-rate multiplier of the encoder.
        grad_clip: Gradient-norm clipping of each module.
        gamma: Discount factor.
        tau: Polyak rate of the target Q-functions.
        rho: Temporal weight decay of the losses over the horizon.
        consistency_coef: Weight of the latent consistency loss.
        reward_coef: Weight of the reward loss.
        value_coef: Weight of the TD loss.
        entropy_coef: Entropy bonus of the policy prior.
        scale_rate: EMA rate of the Q-value scale used by the policy loss.
        latent_dim: Size of the latent state.
        hidden: Width of the dynamics, reward, Q and policy MLPs.
        encoder_dim: Width of the encoder MLP.
        num_q: Q-functions in the ensemble.
        simnorm_dim: Group size of SimNorm.
        num_bins: Bins of the two-hot reward and value distributions.
        v_min: Lower end of the two-hot support (symlog space).
        v_max: Upper end of the two-hot support (symlog space).
        log_std_min: Lower bound of the policy log-std.
        log_std_max: Upper bound of the policy log-std.
        mpc: Act by planning (``False``: with the policy prior).
        iterations: MPPI refinement iterations.
        num_samples: Action sequences evaluated per iteration.
        num_elites: Top sequences used to refit the sampling distribution.
        num_pi_trajs: Sequences proposed by the policy prior.
        min_std: Lower bound of the planner's std.
        max_std: Initial (and maximal) std of the planner.
        temperature: MPPI temperature.
    """

    num_envs: int = 64
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    horizon: int = 3
    updates_per_step: int = 1
    learning_starts: int = 8192
    learning_rate: float = 3e-4
    encoder_lr_scale: float = 0.3
    grad_clip: float = 20.0
    gamma: float = 0.99
    tau: float = 0.01
    rho: float = 0.5
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 1e-4
    scale_rate: float = 0.01
    latent_dim: int = 512
    hidden: int = 512
    encoder_dim: int = 256
    num_q: int = 5
    simnorm_dim: int = 8
    num_bins: int = 101
    v_min: float = -10.0
    v_max: float = 10.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    mpc: bool = True
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def twohot(y: jax.Array, bins: jax.Array) -> jax.Array:
    """Two-hot encoding of ``y`` (already symlog-transformed) over ``bins``."""
    y = jnp.clip(y, bins[0], bins[-1])
    i = jnp.clip(jnp.searchsorted(bins, y) - 1, 0, bins.shape[0] - 2)
    w_hi = (y - bins[i]) / (bins[i + 1] - bins[i])
    n = bins.shape[0]
    return jax.nn.one_hot(i, n) * (1 - w_hi)[..., None] + jax.nn.one_hot(i + 1, n) * w_hi[..., None]


def twohot_mean(logits: jax.Array, bins: jax.Array) -> jax.Array:
    """Expected value (in the original space) of a two-hot distribution."""
    return symexp((jax.nn.softmax(logits) * bins).sum(-1))


def cross_entropy(logits: jax.Array, target: jax.Array) -> jax.Array:
    return -(target * jax.nn.log_softmax(logits)).sum(-1)


def simnorm(x: jax.Array, dim: int) -> jax.Array:
    """Simplicial normalization: softmax over consecutive groups of ``dim`` features."""
    groups = x.reshape(*x.shape[:-1], -1, dim)
    return jax.nn.softmax(groups, -1).reshape(x.shape)


# ── Networks ─────────────────────────────────────────────────────────────────


def mlp(cfg: TDMPC2Config, in_size: int, hidden: list[int], out_size: int, rngs: nnx.Rngs,
        out_scale: float | None = None) -> MLP:
    """TD-MPC2 MLP: ``Linear → LayerNorm → Mish`` hidden layers and a linear output."""
    return MLP(in_size, hidden, out_size, "mish", layer_norm=True, out_scale=out_scale, rngs=rngs)


class Encoder(nnx.Module):
    """Observation → SimNorm latent."""

    def __init__(self, cfg: TDMPC2Config, obs_size: int, *, rngs: nnx.Rngs):
        self.net = mlp(cfg, obs_size, [cfg.encoder_dim], cfg.latent_dim, rngs)
        self.simnorm_dim = cfg.simnorm_dim

    def __call__(self, obs: jax.Array) -> jax.Array:
        return simnorm(self.net(obs), self.simnorm_dim)


class QEnsemble(nnx.Module):
    """``num_q`` two-hot Q-functions; ``qs(z, a)`` returns logits ``(num_q, ..., num_bins)``."""

    def __init__(self, cfg: TDMPC2Config, action_size: int, *, rngs: nnx.Rngs):
        self.qs = nnx.List([
            mlp(cfg, cfg.latent_dim + action_size, [cfg.hidden] * 2, cfg.num_bins, rngs,
                out_scale=0.0)
            for _ in range(cfg.num_q)
        ])

    def __call__(self, z: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([z, action], -1)
        return jnp.stack([q(x) for q in self.qs])

    def values(self, z: jax.Array, action: jax.Array, bins: jax.Array) -> jax.Array:
        return twohot_mean(self(z, action), bins)


class Model(nnx.Module):
    """Latent dynamics, reward head and Q ensemble (trained with one optimizer)."""

    def __init__(self, cfg: TDMPC2Config, action_size: int, *, rngs: nnx.Rngs):
        in_size = cfg.latent_dim + action_size
        self.dynamics = mlp(cfg, in_size, [cfg.hidden] * 2, cfg.latent_dim, rngs)
        self.reward_net = mlp(cfg, in_size, [cfg.hidden] * 2, cfg.num_bins, rngs, out_scale=0.0)
        self.qs = QEnsemble(cfg, action_size, rngs=rngs)
        self.simnorm_dim = cfg.simnorm_dim

    def next(self, z: jax.Array, action: jax.Array) -> jax.Array:
        return simnorm(self.dynamics(jnp.concatenate([z, action], -1)), self.simnorm_dim)

    def reward(self, z: jax.Array, action: jax.Array) -> jax.Array:
        """Reward logits ``(..., num_bins)``."""
        return self.reward_net(jnp.concatenate([z, action], -1))


class Actor(nnx.Module):
    """Tanh-Gaussian policy prior on latents."""

    def __init__(self, cfg: TDMPC2Config, action_size: int, *, rngs: nnx.Rngs):
        self.net = mlp(cfg, cfg.latent_dim, [cfg.hidden] * 2, 2 * action_size, rngs)
        self.log_std_min, self.log_std_max = cfg.log_std_min, cfg.log_std_max

    def __call__(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean, log_std = jnp.split(self.net(z), 2, axis=-1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            jnp.tanh(log_std) + 1.0)
        return mean, log_std

    def sample(self, z: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample ``tanh(N(mean, std))`` and its log-density (summed over action dims)."""
        mean, log_std = self(z)
        std = jnp.exp(log_std)
        u = mean + std * jax.random.normal(key, mean.shape)
        log_prob = -0.5 * jnp.square((u - mean) / std) - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
        log_prob = log_prob - 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        return jnp.tanh(u), log_prob.sum(-1)

    def mode(self, z: jax.Array) -> jax.Array:
        return jnp.tanh(self(z)[0])


class Networks(nnx.Module):
    """All learned modules and their optimizers, split / merged as one state."""

    def __init__(self, cfg: TDMPC2Config, obs_size: int, action_size: int, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.encoder = Encoder(cfg, obs_size, rngs=rngs)
        self.model = Model(cfg, action_size, rngs=rngs)
        self.target_qs = nnx.clone(self.model.qs)
        self.actor = Actor(cfg, action_size, rngs=rngs)
        tx = lambda lr: optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(lr))
        self.encoder_opt = nnx.Optimizer(self.encoder, tx(cfg.learning_rate * cfg.encoder_lr_scale),
                                         wrt=nnx.Param)
        self.model_opt = nnx.Optimizer(self.model, tx(cfg.learning_rate), wrt=nnx.Param)
        self.actor_opt = nnx.Optimizer(self.actor, tx(cfg.learning_rate), wrt=nnx.Param)


class Agent(nnx.Module):
    """The modules needed to act: encoder, model (for planning) and policy prior."""

    def __init__(self, encoder: Encoder, model: Model, actor: Actor):
        self.encoder = encoder
        self.model = model
        self.actor = actor


# ── Planning ─────────────────────────────────────────────────────────────────


def estimate_value(agent: Agent, z: jax.Array, actions: jax.Array, bins: jax.Array,
                   cfg: TDMPC2Config, key: jax.Array) -> jax.Array:
    """Discounted predicted rewards of ``actions (..., H, A)`` plus the Q-value at the horizon."""
    value, discount = jnp.zeros(z.shape[:-1]), 1.0
    for t in range(cfg.horizon):
        a = actions[..., t, :]
        value = value + discount * twohot_mean(agent.model.reward(z, a), bins)
        z = agent.model.next(z, a)
        discount = discount * cfg.gamma
    action_key, q_key = jax.random.split(key)
    a, _ = agent.actor.sample(z, action_key)
    q = agent.model.qs.values(z, a, bins)
    pair = jax.random.choice(q_key, cfg.num_q, (2,), replace=False)
    return value + discount * q[pair].mean(0)


def plan(agent: Agent, z: jax.Array, mean: jax.Array, bins: jax.Array, cfg: TDMPC2Config,
         key: jax.Array, deterministic: bool) -> tuple[jax.Array, jax.Array]:
    """MPPI from latents ``z (B, L)``, warm-started at ``mean (B, H, A)``.

    Returns the action ``(B, A)`` and the refined mean to warm-start the next step.
    """
    B, H, A = mean.shape
    keys = iter(jax.random.split(key, 2 * cfg.iterations + cfg.horizon + 2))
    # proposals of the policy prior
    zp = jnp.repeat(z[:, None], cfg.num_pi_trajs, 1)
    pi_actions = []
    for _ in range(H):
        a, _ = agent.actor.sample(zp, next(keys))
        pi_actions.append(a)
        zp = agent.model.next(zp, a)
    pi_actions = jnp.stack(pi_actions, 2)  # (B, num_pi, H, A)

    z_rep = jnp.repeat(z[:, None], cfg.num_samples, 1)
    std = jnp.full((B, H, A), cfg.max_std)
    for _ in range(cfg.iterations):
        eps = jax.random.normal(next(keys), (B, cfg.num_samples - cfg.num_pi_trajs, H, A))
        samples = jnp.clip(mean[:, None] + std[:, None] * eps, -1.0, 1.0)
        actions = jnp.concatenate([pi_actions, samples], 1)  # (B, N, H, A)
        value = jnp.nan_to_num(estimate_value(agent, z_rep, actions, bins, cfg, next(keys)))
        elite_value, elite_idx = jax.lax.top_k(value, cfg.num_elites)
        elites = jnp.take_along_axis(actions, elite_idx[..., None, None], 1)  # (B, K, H, A)
        score = jax.nn.softmax(cfg.temperature * elite_value, axis=1)[..., None, None]
        mean = (score * elites).sum(1)
        std = jnp.sqrt((score * jnp.square(elites - mean[:, None])).sum(1))
        std = jnp.clip(std, cfg.min_std, cfg.max_std)
    choice = jax.random.categorical(next(keys), jnp.log(score[..., 0, 0]), axis=1)
    action = jnp.take_along_axis(elites, choice[:, None, None, None], 1)[:, 0, 0]
    if not deterministic:
        noise = jax.random.normal(next(keys), action.shape)
        action = jnp.clip(action + std[:, 0] * noise, -1.0, 1.0)
    return action, mean


def act(agent: Agent, obs: jax.Array, mean: jax.Array, first: jax.Array, bins: jax.Array,
        cfg: TDMPC2Config, key: jax.Array, deterministic: bool) -> tuple[jax.Array, jax.Array]:
    """Encode (normalized) ``obs`` and plan, or sample the policy prior if ``cfg.mpc`` is off."""
    z = agent.encoder(obs)
    if not cfg.mpc:
        action = agent.actor.mode(z) if deterministic else agent.actor.sample(z, key)[0]
        return action, mean
    shifted = jnp.concatenate([mean[:, 1:], jnp.zeros_like(mean[:, :1])], 1)
    shifted = shifted * (1.0 - first)[..., None, None]  # no warm start at episode starts
    return plan(agent, z, shifted, bins, cfg, key, deterministic)


def make_policy_apply(graphdef: nnx.GraphDef, cfg: TDMPC2Config, policy_key: str):
    """``apply(params, carry, obs, first, key, deterministic)``; ``carry`` is the planner mean."""
    bins = jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_bins)

    def apply(params, carry, obs, first, key, deterministic):
        agent = nnx.merge(graphdef, params["agent"])
        x = envs.select_obs(obs, (policy_key,))
        x = envs.normalize_obs(x, {policy_key: params["obs_stats"][policy_key]})[policy_key]
        return act(agent, x, carry, first, bins, cfg, key, deterministic)

    return apply


# ── Losses ───────────────────────────────────────────────────────────────────


def world_model_loss(encoder: Encoder, model: Model, target_qs: QEnsemble, actor: Actor,
                     seq: dict[str, jax.Array], bins: jax.Array, cfg: TDMPC2Config,
                     key: jax.Array):
    """Consistency, reward and TD losses on ``[H+1, B]`` sequences (normalized obs)."""
    H = cfg.horizon
    action, reward = seq["action"][:H], seq["reward"][1:]
    cont = 1.0 - seq["terminal"][1:]
    valid = jnp.cumprod(1.0 - seq["last"][:H], 0)  # transitions inside one episode
    weight = cfg.rho ** jnp.arange(H)

    next_z = sg(encoder(seq["obs"][1:]))  # (H, B, L)
    action_key, q_key = jax.random.split(key)
    next_action, _ = actor.sample(next_z, action_key)
    pair = jax.random.choice(q_key, cfg.num_q, (2,), replace=False)
    next_q = target_qs.values(next_z, next_action, bins)[pair].min(0)
    td_target = sg(reward + cfg.gamma * cont * next_q)

    z = encoder(seq["obs"][0])
    zs, consistency = [z], 0.0
    for t in range(H):
        z = model.next(z, action[t])
        consistency += weight[t] * (valid[t] * jnp.square(z - next_z[t]).mean(-1)).mean()
        zs.append(z)
    rollout = jnp.stack(zs[:-1])  # (H, B, L)
    reward_ce = cross_entropy(model.reward(rollout, action), twohot(symlog(reward), bins))
    q_ce = cross_entropy(model.qs(rollout, action), twohot(symlog(td_target), bins)[None])
    reward_loss = (weight[:, None] * valid * reward_ce).mean(-1).sum() / H
    value_loss = (weight[None, :, None] * valid[None] * q_ce).mean(-1).sum() / (H * cfg.num_q)
    consistency = consistency / H
    loss = (cfg.consistency_coef * consistency + cfg.reward_coef * reward_loss
            + cfg.value_coef * value_loss)
    metrics = {"loss/world_model": loss, "loss/consistency": consistency,
               "loss/reward": reward_loss, "loss/value": value_loss}
    return loss, (sg(jnp.stack(zs)), metrics)


def actor_loss(actor: Actor, model: Model, zs: jax.Array, scale: jax.Array, bins: jax.Array,
               cfg: TDMPC2Config, key: jax.Array):
    """Maximize scale-normalized Q (mean of two random critics) plus entropy over the rollout."""
    action_key, q_key = jax.random.split(key)
    action, log_prob = actor.sample(zs, action_key)
    pair = jax.random.choice(q_key, cfg.num_q, (2,), replace=False)
    q = model.qs.values(zs, action, bins)[pair].mean(0)  # (H+1, B)
    q_range = jnp.percentile(sg(q), 95) - jnp.percentile(sg(q), 5)
    weight = cfg.rho ** jnp.arange(zs.shape[0])
    per_step = (cfg.entropy_coef * log_prob - q / jnp.maximum(scale, 1.0)).mean(-1)
    return (weight * per_step).mean(), (q_range, -log_prob.mean(), q.mean())


def update(nets: Networks, seq: dict[str, jax.Array], scale: jax.Array, bins: jax.Array,
           cfg: TDMPC2Config, key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One update of encoder, model and policy prior (in place); returns the new Q scale."""
    model_key, actor_key = jax.random.split(key)
    grad_fn = nnx.value_and_grad(world_model_loss, argnums=(0, 1), has_aux=True)
    (_, (zs, metrics)), (encoder_grads, model_grads) = grad_fn(
        nets.encoder, nets.model, nets.target_qs, nets.actor, seq, bins, cfg, model_key)
    nets.encoder_opt.update(nets.encoder, encoder_grads)
    nets.model_opt.update(nets.model, model_grads)

    (a_loss, (q_range, entropy, q_mean)), grads = nnx.value_and_grad(actor_loss, has_aux=True)(
        nets.actor, nets.model, zs, scale, bins, cfg, actor_key)
    nets.actor_opt.update(nets.actor, grads)

    target = jax.tree.map(lambda t, s: (1.0 - cfg.tau) * t + cfg.tau * s,
                          nnx.state(nets.target_qs), nnx.state(nets.model.qs))
    nnx.update(nets.target_qs, target)
    scale = (1.0 - cfg.scale_rate) * scale + cfg.scale_rate * q_range
    metrics.update({"loss/actor": a_loss, "train/entropy": entropy, "train/q_mean": q_mean,
                    "train/q_scale": scale})
    return scale, metrics


# ── Training ─────────────────────────────────────────────────────────────────


class TDMPC2State(NamedTuple):
    nets: nnx.State
    obs_stats: envs.ObsStats
    scale: jax.Array  # EMA of the 5th-95th percentile range of Q-values
    buffer: replay.ReplayBuffer
    env_state: Any
    plan_mean: jax.Array  # (num_envs, H, A) warm start of the planner
    first: jax.Array
    key: jax.Array


def train(
    cfg: TDMPC2Config,
    env_factory: envs.EnvFactory | None = None,
    log_callback: LogCallback | None = None,
    resume_from: str | Path | None = None,
) -> Policy:
    """Train TD-MPC2 and return the final (planning) policy.

    Args:
        cfg: Hyperparameters.
        env_factory: Optional callable returning a raw ``MjxEnv`` (replaces ``cfg.env_id``).
        log_callback: Optional ``callback(metrics, step)`` called on every log.
        resume_from: Run directory whose latest checkpoint initializes the networks
            (the replay buffer starts empty).
    """
    s = setup(cfg, NAME, env_factory, log_callback, autoreset=False)
    env, pk, n, A, H = s.env, s.policy_key, cfg.num_envs, s.action_size, cfg.horizon
    key, net_key = jax.random.split(s.key)
    nets = Networks(cfg, s.obs_shapes[pk][-1], A, net_key)
    graphdef, nets_state = nnx.split(nets)
    agent_graphdef = nnx.split(Agent(nets.encoder, nets.model, nets.actor))[0]
    bins = jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_bins)
    obs = envs.select_obs(s.env_state.obs, (pk,))
    obs_stats = envs.init_obs_stats(obs)
    scale = jnp.ones(())
    if resume_from is not None:
        _, restored = checkpoint.load(
            resume_from, {"nets": nets_state, "obs_stats": obs_stats, "scale": scale})
        nets_state, obs_stats, scale = restored["nets"], restored["obs_stats"], restored["scale"]

    row = {"obs": obs, "action": jnp.zeros((n, A)), "reward": jnp.zeros(n),
           "terminal": jnp.zeros(n), "last": jnp.zeros(n), "first": jnp.zeros(n)}
    buffer = replay.init(row, cfg.buffer_size // n)

    def collect(state: TDMPC2State, nets: Networks, random: bool) -> TDMPC2State:
        """Plan in all environments (or act randomly), store the step and act."""
        key, act_key, reset_key = jax.random.split(state.key, 3)
        env_state = state.env_state
        obs = envs.select_obs(env_state.obs, (pk,))
        if random:
            action = jax.random.uniform(act_key, (n, A), minval=-1.0, maxval=1.0)
            plan_mean = state.plan_mean
        else:
            x = envs.normalize_obs(obs, state.obs_stats)[pk]
            agent = Agent(nets.encoder, nets.model, nets.actor)
            action, plan_mean = act(agent, x, state.plan_mean, state.first, bins, cfg, act_key,
                                    deterministic=False)
        buffer = replay.add(state.buffer, {
            "obs": obs, "action": action, "reward": env_state.reward,
            "terminal": envs.termination(env_state), "last": env_state.done,
            "first": state.first})
        next_state = envs.step_or_reset(env, env_state, action, reset_key)
        return state._replace(obs_stats=envs.update_obs_stats(state.obs_stats, obs),
                              buffer=buffer, env_state=next_state, plan_mean=plan_mean,
                              first=env_state.done, key=key)

    def learn(state: TDMPC2State, nets: Networks):
        key, scale, metrics = state.key, state.scale, {}
        for _ in range(cfg.updates_per_step):
            key, sample_key, update_key = jax.random.split(key, 3)
            seq = replay.sample_sequences(state.buffer, sample_key, cfg.batch_size, H + 1)
            seq["obs"] = envs.normalize_obs(seq["obs"], state.obs_stats)[pk]
            scale, metrics = update(nets, seq, scale, bins, cfg, update_key)
        return state._replace(scale=scale, key=key), metrics

    def make_train_fn(num_iterations: int, random: bool):
        def iteration(state: TDMPC2State, _):
            nets = nnx.merge(graphdef, state.nets)
            state = collect(state, nets, random)
            metrics = {}
            if not random:
                state, metrics = learn(state, nets)
            return state._replace(nets=nnx.split(nets)[1]), metrics

        @jax.jit(donate_argnums=0)  # update the replay buffer in place
        def run(buffer: replay.ReplayBuffer, state: TDMPC2State):
            state, metrics = jax.lax.scan(iteration, state._replace(buffer=buffer), None,
                                          num_iterations)
            return state.buffer, state._replace(buffer=None), jax.tree.map(lambda x: x[-1], metrics)

        def train_fn(state: TDMPC2State):
            buffer, state, metrics = run(state.buffer, state._replace(buffer=None))
            return state._replace(buffer=buffer), metrics

        return train_fn

    state = TDMPC2State(nets_state, obs_stats, scale, buffer, s.env_state,
                        jnp.zeros((n, H, A)), jnp.ones(n), key)
    warmup_iterations = max(math.ceil(cfg.learning_starts / n), H + 2)
    state, _ = make_train_fn(warmup_iterations, random=True)(state)
    iterations = max(1, min(cfg.total_timesteps // (cfg.num_logs * n), 1000))

    def policy_params(state: TDMPC2State) -> dict[str, Any]:
        nets = nnx.merge(graphdef, state.nets)
        return {"agent": nnx.state(Agent(nets.encoder, nets.model, nets.actor)),
                "obs_stats": state.obs_stats}

    init_carry = lambda batch_size: jnp.zeros((batch_size, H, A))
    apply = make_policy_apply(agent_graphdef, cfg, pk)
    evaluate = envs.make_evaluator(cfg, s.eval_env, apply, init_carry)
    state, _ = train_loop(
        cfg, s.logger, state, make_train_fn(iterations, random=False), iterations * n,
        evaluate, policy_params,
        lambda st: {"nets": st.nets, "obs_stats": st.obs_stats, "scale": st.scale},
        jax.random.key(cfg.seed + 1), warmup_iterations * n)
    s.logger.close()
    return Policy(NAME, policy_params(state), apply, init_carry)


def load_policy(run_dir: str | Path, step: int | None = None) -> Policy:
    """Rebuild the policy saved in ``run_dir`` (latest checkpoint unless ``step`` is given)."""
    meta = checkpoint.load_metadata(run_dir)
    cfg = TDMPC2Config.from_dict(meta["config"])
    spec = meta["spec"]
    pk, A = spec["policy_obs_key"], spec["action_size"]
    shapes = {k: tuple(v) for k, v in spec["obs_shapes"].items()}
    nets = Networks(cfg, shapes[pk][-1], A, jax.random.key(0))
    graphdef, nets_state = nnx.split(nets)
    obs_stats = {pk: running_statistics.init_state(shapes[pk][-1:])}
    target = {"nets": nets_state, "obs_stats": obs_stats, "scale": jnp.ones(())}
    _, restored = checkpoint.load(run_dir, target, step)
    nets = nnx.merge(graphdef, restored["nets"])
    agent_graphdef, agent_state = nnx.split(Agent(nets.encoder, nets.model, nets.actor))
    params = {"agent": agent_state, "obs_stats": restored["obs_stats"]}
    return Policy(NAME, params, make_policy_apply(agent_graphdef, cfg, pk),
                  lambda batch_size: jnp.zeros((batch_size, cfg.horizon, A)))


Config = TDMPC2Config
