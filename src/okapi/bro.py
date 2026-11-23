"""BRO (Nauman et al., 2024): Bigger, Regularized, Optimistic soft actor-critic.

Soft actor-critic in the massively parallel regime of :mod:`okapi.sac`, with
the BRO recipe:

* **BroNet** actor and critics: a ``Dense → LayerNorm → ReLU`` stem followed by
  residual blocks ``[Dense → LN → ReLU → Dense → LN] + skip``.
* **Regularization**: AdamW weight decay.
* **Distributional critic** with quantile regression and reduced pessimism (the
  target and the actor use the mean of both critics instead of their minimum).
* **Optimistic exploration**: a second actor collects the data, maximizing
  ``Q + optimism * |Q1 - Q2| / 2`` while a KL penalty keeps it close to the
  conservative actor (which is the one evaluated and returned).

Simplifications: no periodic network resets and batch size / updates per step
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
from okapi.common.policy import Policy
from okapi.common.training import setup, train_loop

NAME = "bro"
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class BROConfig(TrainConfig):
    """BRO hyperparameters (SAC options as in :class:`okapi.sac.SACConfig`).

    Attributes:
        weight_decay: AdamW weight decay of actors and critics.
        critic_width: Hidden width of each critic.
        critic_blocks: Residual blocks per critic.
        actor_width: Hidden width of the actors.
        actor_blocks: Residual blocks of the actors.
        num_quantiles: Quantiles per critic.
        optimism: Weight of the critic disagreement for the exploration actor (0 disables it).
        kl_coef: KL penalty between exploration and conservative actor.
    """

    num_envs: int = 128
    total_timesteps: int = 10_000_000
    buffer_size: int = 4_194_304
    batch_size: int = 512
    updates_per_step: int = 8
    learning_starts: int = 8192
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float | None = None
    gamma: float = 0.99
    n_step: int = 1
    tau: float = 0.005
    init_alpha: float = 0.1
    target_entropy: float | None = None
    reward_scaling: float = 1.0
    critic_width: int = 512
    critic_blocks: int = 2
    actor_width: int = 256
    actor_blocks: int = 1
    num_quantiles: int = 100
    optimism: float = 1.0
    kl_coef: float = 1.0


# ── Networks ─────────────────────────────────────────────────────────────────


class BroNet(nnx.Module):
    """``Dense → LN → ReLU`` stem, ``num_blocks`` LayerNorm residual blocks, linear output."""

    def __init__(self, in_size: int, width: int, num_blocks: int, out_size: int,
                 *, rngs: nnx.Rngs):
        init = nnx.initializers.orthogonal(math.sqrt(2.0))
        dense = lambda a, b: nnx.Linear(a, b, kernel_init=init, rngs=rngs)
        self.stem = dense(in_size, width)
        self.stem_norm = nnx.LayerNorm(width, rngs=rngs)
        self.dense1 = nnx.List([dense(width, width) for _ in range(num_blocks)])
        self.norm1 = nnx.List([nnx.LayerNorm(width, rngs=rngs) for _ in range(num_blocks)])
        self.dense2 = nnx.List([dense(width, width) for _ in range(num_blocks)])
        self.norm2 = nnx.List([nnx.LayerNorm(width, rngs=rngs) for _ in range(num_blocks)])
        self.out = nnx.Linear(width, out_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jax.nn.relu(self.stem_norm(self.stem(x)))
        for d1, n1, d2, n2 in zip(self.dense1, self.norm1, self.dense2, self.norm2):
            h = h + n2(d2(jax.nn.relu(n1(d1(h)))))
        return self.out(h)


def sample_tanh_normal(mean: jax.Array, log_std: jax.Array,
                       key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sample ``tanh(N(mean, std))`` and its log-density (summed over action dims)."""
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    log_prob = -0.5 * jnp.square((u - mean) / std) - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    log_prob = log_prob - 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    return jnp.tanh(u), log_prob.sum(-1)


class Actor(nnx.Module):
    """Tanh-Gaussian policy on a BroNet."""

    def __init__(self, obs_size: int, action_size: int, cfg: BROConfig, *, rngs: nnx.Rngs):
        self.net = BroNet(obs_size, cfg.actor_width, cfg.actor_blocks, 2 * action_size,
                          rngs=rngs)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean, log_std = jnp.split(self.net(obs), 2, axis=-1)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (jnp.tanh(log_std) + 1.0)
        return mean, log_std

    def sample(self, obs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        return sample_tanh_normal(*self(obs), key)

    def mode(self, obs: jax.Array) -> jax.Array:
        return jnp.tanh(self(obs)[0])


class Critic(nnx.Module):
    """Twin quantile critics; ``critic(obs, action)`` returns ``(2, batch, num_quantiles)``."""

    def __init__(self, obs_size: int, action_size: int, cfg: BROConfig, *, rngs: nnx.Rngs):
        self.qs = nnx.List([
            BroNet(obs_size + action_size, cfg.critic_width, cfg.critic_blocks,
                   cfg.num_quantiles, rngs=rngs)
            for _ in range(2)
        ])

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], -1)
        return jnp.stack([q(x) for q in self.qs])


class Temperature(nnx.Module):
    def __init__(self, init_alpha: float):
        self.log_alpha = nnx.Param(jnp.log(jnp.asarray(init_alpha, jnp.float32)))

    def __call__(self) -> jax.Array:
        return jnp.exp(self.log_alpha[...])


class Networks(nnx.Module):
    """All learned modules and their optimizers, split / merged as one state."""

    def __init__(self, cfg: BROConfig, policy_obs_size: int, value_obs_size: int,
                 action_size: int, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.actor = Actor(policy_obs_size, action_size, cfg, rngs=rngs)
        self.explorer = Actor(policy_obs_size, action_size, cfg, rngs=rngs)
        self.critic = Critic(value_obs_size, action_size, cfg, rngs=rngs)
        self.target_critic = nnx.clone(self.critic)
        self.temperature = Temperature(cfg.init_alpha)

        def tx():
            opt = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)
            if cfg.max_grad_norm is None:
                return opt
            return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), opt)

        self.actor_opt = nnx.Optimizer(self.actor, tx(), wrt=nnx.Param)
        self.explorer_opt = nnx.Optimizer(self.explorer, tx(), wrt=nnx.Param)
        self.critic_opt = nnx.Optimizer(self.critic, tx(), wrt=nnx.Param)
        self.temperature_opt = nnx.Optimizer(self.temperature, optax.adam(cfg.learning_rate),
                                             wrt=nnx.Param)


def make_policy_apply(graphdef: nnx.GraphDef, policy_key: str):
    """``apply(params, carry, obs, first, key, deterministic)`` for :class:`Policy`."""

    def apply(params, carry, obs, first, key, deterministic):
        actor = nnx.merge(graphdef, params["actor"])
        x = envs.select_obs(obs, (policy_key,))
        x = envs.normalize_obs(x, {policy_key: params["obs_stats"][policy_key]})[policy_key]
        action = actor.mode(x) if deterministic else actor.sample(x, key)[0]
        return action, carry

    return apply


# ── Losses ───────────────────────────────────────────────────────────────────


def quantile_huber_loss(pred: jax.Array, target: jax.Array, kappa: float = 1.0) -> jax.Array:
    """Quantile Huber loss of ``pred`` (B, N) quantiles against ``target`` (B, M) samples."""
    n = pred.shape[-1]
    tau = (jnp.arange(n) + 0.5) / n
    u = target[:, None, :] - pred[:, :, None]  # (B, N, M)
    huber = jnp.where(jnp.abs(u) <= kappa, 0.5 * u**2, kappa * (jnp.abs(u) - 0.5 * kappa))
    return (jnp.abs(tau[None, :, None] - (u < 0)) * huber / kappa).sum(1).mean()


def critic_loss(critic: Critic, actor: Actor, target_critic: Critic, alpha: jax.Array,
                batch: dict[str, Any], pk: str, vk: str,
                key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Quantile regression towards the soft n-step target of the mean target critic."""
    next_action, next_log_prob = actor.sample(batch["next_obs"][pk], key)
    next_quantiles = target_critic(batch["next_obs"][vk], next_action).mean(0)  # (B, N)
    target = batch["reward"][:, None] + batch["discount"][:, None] * (
        next_quantiles - alpha * next_log_prob[:, None])
    target = jax.lax.stop_gradient(target)
    quantiles = critic(batch["obs"][vk], batch["action"])
    loss = quantile_huber_loss(quantiles[0], target) + quantile_huber_loss(quantiles[1], target)
    return loss, {"loss/critic": loss, "train/q_mean": quantiles.mean()}


def actor_loss(actor: Actor, critic: Critic, alpha: jax.Array, batch: dict[str, Any],
               pk: str, vk: str,
               key: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Maximize the soft value under the mean of both critics (reduced pessimism)."""
    mean, log_std = actor(batch["obs"][pk])
    action, log_prob = sample_tanh_normal(mean, log_std, key)
    q = critic(batch["obs"][vk], action).mean((0, -1))
    return (alpha * log_prob - q).mean(), (log_prob, mean, log_std)


def explorer_loss(explorer: Actor, critic: Critic, batch: dict[str, Any], mean: jax.Array,
                  log_std: jax.Array, cfg: BROConfig, pk: str, vk: str,
                  key: jax.Array) -> jax.Array:
    """Maximize ``Q + optimism * |Q1 - Q2| / 2`` with a KL penalty to the conservative actor."""
    e_mean, e_log_std = explorer(batch["obs"][pk])
    action, _ = sample_tanh_normal(e_mean, e_log_std, key)
    q = critic(batch["obs"][vk], action).mean(-1)  # (2, B)
    optimistic_q = q.mean(0) + cfg.optimism * jnp.abs(q[0] - q[1]) / 2.0
    kl = (log_std - e_log_std
          + (jnp.exp(2 * e_log_std) + jnp.square(e_mean - mean)) / (2 * jnp.exp(2 * log_std))
          - 0.5).sum(-1)
    return (-optimistic_q + cfg.kl_coef * kl).mean()


def temperature_loss(temperature: Temperature, log_prob: jax.Array,
                     target_entropy: float) -> jax.Array:
    return -(temperature.log_alpha[...] * jax.lax.stop_gradient(log_prob + target_entropy)).mean()


def update(nets: Networks, batch: dict[str, Any], cfg: BROConfig, pk: str, vk: str,
           target_entropy: float, key: jax.Array) -> dict[str, jax.Array]:
    """One gradient step of critic, actor, explorer and temperature, then the target update."""
    critic_key, actor_key, explorer_key = jax.random.split(key, 3)
    alpha = nets.temperature()
    (_, metrics), grads = nnx.value_and_grad(critic_loss, has_aux=True)(
        nets.critic, nets.actor, nets.target_critic, alpha, batch, pk, vk, critic_key)
    nets.critic_opt.update(nets.critic, grads)

    (loss, (log_prob, mean, log_std)), grads = nnx.value_and_grad(actor_loss, has_aux=True)(
        nets.actor, nets.critic, alpha, batch, pk, vk, actor_key)
    nets.actor_opt.update(nets.actor, grads)

    alpha_loss, grads = nnx.value_and_grad(temperature_loss)(nets.temperature, log_prob,
                                                            target_entropy)
    nets.temperature_opt.update(nets.temperature, grads)

    if cfg.optimism > 0:
        mean, log_std = jax.lax.stop_gradient((mean, log_std))
        e_loss, grads = nnx.value_and_grad(explorer_loss)(
            nets.explorer, nets.critic, batch, mean, log_std, cfg, pk, vk, explorer_key)
        nets.explorer_opt.update(nets.explorer, grads)
        metrics["loss/explorer"] = e_loss

    target = jax.tree.map(lambda t, s: (1.0 - cfg.tau) * t + cfg.tau * s,
                          nnx.state(nets.target_critic), nnx.state(nets.critic))
    nnx.update(nets.target_critic, target)
    return {**metrics, "loss/actor": loss, "loss/alpha": alpha_loss,
            "train/alpha": nets.temperature(), "train/entropy": -log_prob.mean()}


# ── Training ─────────────────────────────────────────────────────────────────


class BROState(NamedTuple):
    nets: nnx.State
    obs_stats: envs.ObsStats
    buffer: replay.ReplayBuffer
    env_state: Any
    first: jax.Array
    key: jax.Array


def train(
    cfg: BROConfig,
    env_factory: envs.EnvFactory | None = None,
    log_callback: LogCallback | None = None,
    resume_from: str | Path | None = None,
) -> Policy:
    """Train BRO and return the final (conservative) policy (see :func:`okapi.sac.train`)."""
    s = setup(cfg, NAME, env_factory, log_callback, autoreset=False)
    env, pk, vk, n = s.env, s.policy_key, s.value_key, cfg.num_envs
    keys = (pk, vk)
    key, net_key = jax.random.split(s.key)
    nets = Networks(cfg, s.obs_shapes[pk][-1], s.obs_shapes[vk][-1], s.action_size, net_key)
    graphdef, nets_state = nnx.split(nets)
    actor_graphdef = nnx.split(nets.actor)[0]
    obs = envs.select_obs(s.env_state.obs, keys)
    obs_stats = envs.init_obs_stats(obs)
    target_entropy = (-0.5 * s.action_size if cfg.target_entropy is None
                      else cfg.target_entropy)
    if resume_from is not None:
        _, restored = checkpoint.load(resume_from, {"nets": nets_state, "obs_stats": obs_stats})
        nets_state, obs_stats = restored["nets"], restored["obs_stats"]

    row = {"obs": obs, "action": jnp.zeros((n, s.action_size)), "reward": jnp.zeros(n),
           "terminal": jnp.zeros(n), "last": jnp.zeros(n), "first": jnp.zeros(n)}
    buffer = replay.init(row, cfg.buffer_size // n)

    def collect(state: BROState, nets: Networks, random: bool) -> BROState:
        """Store the current step and act with the exploration actor in all environments."""
        key, action_key, reset_key = jax.random.split(state.key, 3)
        env_state = state.env_state
        obs = envs.select_obs(env_state.obs, keys)
        if random:
            action = jax.random.uniform(action_key, (n, s.action_size), minval=-1.0, maxval=1.0)
        else:
            x = envs.normalize_obs(obs, state.obs_stats)[pk]
            behavior = nets.explorer if cfg.optimism > 0 else nets.actor
            action, _ = behavior.sample(x, action_key)
        buffer = replay.add(state.buffer, {
            "obs": obs, "action": action, "reward": env_state.reward * cfg.reward_scaling,
            "terminal": envs.termination(env_state), "last": env_state.done,
            "first": state.first})
        next_state = envs.step_or_reset(env, env_state, action, reset_key)
        return state._replace(obs_stats=envs.update_obs_stats(state.obs_stats, obs),
                              buffer=buffer, env_state=next_state, first=env_state.done, key=key)

    def learn(state: BROState, nets: Networks) -> tuple[BROState, dict[str, jax.Array]]:
        metrics = {}
        key = state.key
        for _ in range(cfg.updates_per_step):
            key, sample_key, update_key = jax.random.split(key, 3)
            batch = replay.sample_transitions(state.buffer, sample_key, cfg.batch_size,
                                              cfg.n_step, cfg.gamma)
            batch["obs"] = envs.normalize_obs(batch["obs"], state.obs_stats)
            batch["next_obs"] = envs.normalize_obs(batch["next_obs"], state.obs_stats)
            metrics = update(nets, batch, cfg, pk, vk, target_entropy, update_key)
        return state._replace(key=key), metrics

    def make_train_fn(num_iterations: int, random: bool):
        def iteration(state: BROState, _):
            nets = nnx.merge(graphdef, state.nets)
            state = collect(state, nets, random)
            metrics = {}
            if not random:
                state, metrics = learn(state, nets)
            return state._replace(nets=nnx.split(nets)[1]), metrics

        @jax.jit(donate_argnums=0)  # update the replay buffer in place
        def run(buffer: replay.ReplayBuffer, state: BROState):
            state, metrics = jax.lax.scan(iteration, state._replace(buffer=buffer), None,
                                          num_iterations)
            return state.buffer, state._replace(buffer=None), jax.tree.map(lambda x: x[-1], metrics)

        def train_fn(state: BROState) -> tuple[BROState, dict[str, jax.Array]]:
            buffer, state, metrics = run(state.buffer, state._replace(buffer=None))
            return state._replace(buffer=buffer), metrics

        return train_fn

    state = BROState(nets_state, obs_stats, buffer, s.env_state, jnp.ones(n), key)
    warmup_iterations = max(math.ceil(cfg.learning_starts / n), cfg.n_step + 2)
    state, _ = make_train_fn(warmup_iterations, random=True)(state)
    iterations = max(1, min(cfg.total_timesteps // (cfg.num_logs * n), 1000))

    def policy_params(state: BROState) -> dict[str, Any]:
        return {"actor": nnx.state(nnx.merge(graphdef, state.nets).actor),
                "obs_stats": state.obs_stats}

    apply = make_policy_apply(actor_graphdef, pk)
    evaluate = envs.make_evaluator(cfg, s.eval_env, apply, lambda n: ())
    state, _ = train_loop(
        cfg, s.logger, state, make_train_fn(iterations, random=False), iterations * n,
        evaluate, policy_params, lambda st: {"nets": st.nets, "obs_stats": st.obs_stats},
        jax.random.key(cfg.seed + 1), warmup_iterations * n)
    s.logger.close()
    return Policy(NAME, policy_params(state), apply)


def load_policy(run_dir: str | Path, step: int | None = None) -> Policy:
    """Rebuild the policy saved in ``run_dir`` (latest checkpoint unless ``step`` is given)."""
    meta = checkpoint.load_metadata(run_dir)
    cfg = BROConfig.from_dict(meta["config"])
    spec = meta["spec"]
    pk, vk = spec["policy_obs_key"], spec["value_obs_key"]
    shapes = {k: tuple(v) for k, v in spec["obs_shapes"].items()}
    nets = Networks(cfg, shapes[pk][-1], shapes[vk][-1], spec["action_size"], jax.random.key(0))
    graphdef, nets_state = nnx.split(nets)
    obs_stats = {k: running_statistics.init_state(v[-1:]) for k, v in shapes.items()}
    _, restored = checkpoint.load(run_dir, {"nets": nets_state, "obs_stats": obs_stats}, step)
    actor_graphdef, actor_state = nnx.split(nnx.merge(graphdef, restored["nets"]).actor)
    params = {"actor": actor_state, "obs_stats": restored["obs_stats"]}
    return Policy(NAME, params, make_policy_apply(actor_graphdef, pk))


Config = BROConfig
