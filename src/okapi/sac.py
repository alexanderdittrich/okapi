"""Soft Actor-Critic (Haarnoja et al., 2018) for massively parallel environments.

Every iteration, all environments take one step into a GPU replay buffer,
followed by ``updates_per_step`` gradient updates (FastSAC / Brax SAC style;
collection and learning are compiled together). Tanh-Gaussian policy, twin Q
critics with Polyak-averaged targets trained towards clipped double-Q
n-step targets, and a learned temperature. Time-limit truncations bootstrap
from the true final observation (see :mod:`okapi.common.replay`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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

NAME = "sac"
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class SACConfig(TrainConfig):
    """SAC hyperparameters (defaults: MuJoCo Playground's Brax SAC settings).

    Attributes:
        buffer_size: Replay capacity in transitions (split evenly over environments).
        batch_size: Transitions per gradient update.
        updates_per_step: Gradient updates per environment step (of all environments).
        learning_starts: Random-action transitions collected before learning.
        learning_rate: Adam learning rate of actor, critic and temperature.
        max_grad_norm: Global gradient-norm clipping (``None`` disables).
        gamma: Discount factor.
        n_step: Return horizon of the critic targets.
        tau: Polyak averaging rate of the target critics.
        init_alpha: Initial temperature.
        target_entropy: Entropy target; ``None`` means ``-action_size / 2``.
        reward_scaling: Constant reward multiplier.
        actor_hidden_sizes: Actor MLP widths.
        critic_hidden_sizes: Critic MLP widths.
        activation: Hidden activation.
        critic_layer_norm: LayerNorm in the critic MLPs.
    """

    num_envs: int = 128
    total_timesteps: int = 10_000_000
    buffer_size: int = 4_194_304
    batch_size: int = 512
    updates_per_step: int = 8
    learning_starts: int = 8192
    learning_rate: float = 1e-3
    max_grad_norm: float | None = None
    gamma: float = 0.99
    n_step: int = 1
    tau: float = 0.005
    init_alpha: float = 1.0
    target_entropy: float | None = None
    reward_scaling: float = 1.0
    actor_hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    critic_layer_norm: bool = True


# ── Networks ─────────────────────────────────────────────────────────────────


def sample_tanh_normal(mean: jax.Array, log_std: jax.Array,
                       key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sample ``tanh(N(mean, std))`` and its log-density (summed over action dims)."""
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    log_prob = -0.5 * jnp.square((u - mean) / std) - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    log_prob = log_prob - 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    return jnp.tanh(u), log_prob.sum(-1)


class Actor(nnx.Module):
    """Tanh-Gaussian policy with state-dependent, bounded log-std."""

    def __init__(self, obs_size: int, action_size: int, cfg: SACConfig, *, rngs: nnx.Rngs):
        self.net = MLP(obs_size, cfg.actor_hidden_sizes, 2 * action_size, cfg.activation,
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
    """Twin Q-networks; ``critic(obs, action)`` returns ``(2, batch)``."""

    def __init__(self, obs_size: int, action_size: int, cfg: SACConfig, *, rngs: nnx.Rngs):
        self.qs = nnx.List([
            MLP(obs_size + action_size, cfg.critic_hidden_sizes, 1, cfg.activation,
                layer_norm=cfg.critic_layer_norm, rngs=rngs)
            for _ in range(2)
        ])

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], -1)
        return jnp.stack([q(x).squeeze(-1) for q in self.qs])


class Temperature(nnx.Module):
    def __init__(self, init_alpha: float):
        self.log_alpha = nnx.Param(jnp.log(jnp.asarray(init_alpha, jnp.float32)))

    def __call__(self) -> jax.Array:
        return jnp.exp(self.log_alpha[...])


class Networks(nnx.Module):
    """All learned modules and their optimizers, split / merged as one state."""

    def __init__(self, cfg: SACConfig, policy_obs_size: int, value_obs_size: int,
                 action_size: int, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.actor = Actor(policy_obs_size, action_size, cfg, rngs=rngs)
        self.critic = Critic(value_obs_size, action_size, cfg, rngs=rngs)
        self.target_critic = nnx.clone(self.critic)
        self.temperature = Temperature(cfg.init_alpha)

        def tx():
            opt = optax.adam(cfg.learning_rate)
            if cfg.max_grad_norm is None:
                return opt
            return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), opt)

        self.actor_opt = nnx.Optimizer(self.actor, tx(), wrt=nnx.Param)
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


def critic_loss(critic: Critic, actor: Actor, target_critic: Critic, alpha: jax.Array,
                batch: dict[str, Any], pk: str, vk: str,
                key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Squared error towards the clipped double-Q soft n-step target."""
    next_action, next_log_prob = actor.sample(batch["next_obs"][pk], key)
    next_q = target_critic(batch["next_obs"][vk], next_action).min(0)
    target = batch["reward"] + batch["discount"] * (next_q - alpha * next_log_prob)
    q = critic(batch["obs"][vk], batch["action"])
    loss = 0.5 * jnp.square(q - jax.lax.stop_gradient(target)[None]).mean(-1).sum()
    return loss, {"loss/critic": loss, "train/q_mean": q.mean()}


def actor_loss(actor: Actor, critic: Critic, alpha: jax.Array, batch: dict[str, Any],
               pk: str, vk: str, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Maximize the soft value under the minimum of both critics."""
    action, log_prob = actor.sample(batch["obs"][pk], key)
    q = critic(batch["obs"][vk], action).min(0)
    return (alpha * log_prob - q).mean(), log_prob


def temperature_loss(temperature: Temperature, log_prob: jax.Array,
                     target_entropy: float) -> jax.Array:
    return -(temperature.log_alpha[...] * jax.lax.stop_gradient(log_prob + target_entropy)).mean()


def update(nets: Networks, batch: dict[str, Any], cfg: SACConfig, pk: str, vk: str,
           target_entropy: float, key: jax.Array) -> dict[str, jax.Array]:
    """One gradient step of critic, actor and temperature, then the target update (in place)."""
    critic_key, actor_key = jax.random.split(key)
    alpha = nets.temperature()
    (_, metrics), grads = nnx.value_and_grad(critic_loss, has_aux=True)(
        nets.critic, nets.actor, nets.target_critic, alpha, batch, pk, vk, critic_key)
    nets.critic_opt.update(nets.critic, grads)

    (loss, log_prob), grads = nnx.value_and_grad(actor_loss, has_aux=True)(
        nets.actor, nets.critic, alpha, batch, pk, vk, actor_key)
    nets.actor_opt.update(nets.actor, grads)

    alpha_loss, grads = nnx.value_and_grad(temperature_loss)(nets.temperature, log_prob,
                                                            target_entropy)
    nets.temperature_opt.update(nets.temperature, grads)

    target = jax.tree.map(lambda t, s: (1.0 - cfg.tau) * t + cfg.tau * s,
                          nnx.state(nets.target_critic), nnx.state(nets.critic))
    nnx.update(nets.target_critic, target)
    return {**metrics, "loss/actor": loss, "loss/alpha": alpha_loss,
            "train/alpha": nets.temperature(), "train/entropy": -log_prob.mean()}


# ── Training ─────────────────────────────────────────────────────────────────


class SACState(NamedTuple):
    nets: nnx.State
    obs_stats: envs.ObsStats
    buffer: replay.ReplayBuffer
    env_state: Any
    first: jax.Array  # current obs starts an episode
    key: jax.Array


def train(
    cfg: SACConfig,
    env_factory: envs.EnvFactory | None = None,
    log_callback: LogCallback | None = None,
    resume_from: str | Path | None = None,
) -> Policy:
    """Train SAC and return the final policy.

    Args:
        cfg: Hyperparameters.
        env_factory: Optional callable returning a raw ``MjxEnv`` (replaces ``cfg.env_id``).
        log_callback: Optional ``callback(metrics, step)`` called on every log.
        resume_from: Run directory whose latest checkpoint initializes the networks
            (the replay buffer starts empty).
    """
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

    def collect(state: SACState, nets: Networks, random: bool) -> SACState:
        """Store the current step and act in all environments."""
        key, action_key, reset_key = jax.random.split(state.key, 3)
        env_state = state.env_state
        obs = envs.select_obs(env_state.obs, keys)
        if random:
            action = jax.random.uniform(action_key, (n, s.action_size), minval=-1.0, maxval=1.0)
        else:
            x = envs.normalize_obs(obs, state.obs_stats)[pk]
            action, _ = nets.actor.sample(x, action_key)
        buffer = replay.add(state.buffer, {
            "obs": obs, "action": action, "reward": env_state.reward * cfg.reward_scaling,
            "terminal": envs.termination(env_state), "last": env_state.done,
            "first": state.first})
        next_state = envs.step_or_reset(env, env_state, action, reset_key)
        return state._replace(obs_stats=envs.update_obs_stats(state.obs_stats, obs),
                              buffer=buffer, env_state=next_state, first=env_state.done, key=key)

    def learn(state: SACState, nets: Networks) -> tuple[SACState, dict[str, jax.Array]]:
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
        def iteration(state: SACState, _):
            nets = nnx.merge(graphdef, state.nets)
            state = collect(state, nets, random)
            metrics = {}
            if not random:
                state, metrics = learn(state, nets)
            return state._replace(nets=nnx.split(nets)[1]), metrics

        @jax.jit(donate_argnums=0)  # update the replay buffer in place
        def run(buffer: replay.ReplayBuffer, state: SACState):
            state, metrics = jax.lax.scan(iteration, state._replace(buffer=buffer), None,
                                          num_iterations)
            return state.buffer, state._replace(buffer=None), jax.tree.map(lambda x: x[-1], metrics)

        def train_fn(state: SACState) -> tuple[SACState, dict[str, jax.Array]]:
            buffer, state, metrics = run(state.buffer, state._replace(buffer=None))
            return state._replace(buffer=buffer), metrics

        return train_fn

    state = SACState(nets_state, obs_stats, buffer, s.env_state, jnp.ones(n), key)
    warmup_iterations = max(math.ceil(cfg.learning_starts / n), cfg.n_step + 2)
    state, _ = make_train_fn(warmup_iterations, random=True)(state)
    iterations = max(1, min(cfg.total_timesteps // (cfg.num_logs * n), 1000))

    def policy_params(state: SACState) -> dict[str, Any]:
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
    cfg = SACConfig.from_dict(meta["config"])
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


Config = SACConfig
