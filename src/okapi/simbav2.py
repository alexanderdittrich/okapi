"""SimbaV2 (Lee et al., 2025): SAC with hyperspherical networks and a distributional critic.

Soft actor-critic in the massively parallel regime of :mod:`okapi.sac`, with
the SimbaV2 recipe:

* **Hyperspherical networks.** The (running-normalized) input gets a constant
  shift feature and is projected onto the unit sphere; every hidden vector stays
  on the sphere, residual blocks interpolate (LERP) towards the block output,
  learnable scalers set magnitudes, and weight matrices are projected back onto
  the sphere after every update.
* **Categorical critic** (C51) on ``[-v_max, v_max]``, clipped double-Q by
  taking the target distribution with the lower mean.
* **Reward scaling** by the running standard deviation of the discounted return.

Simplifications: constant learning rate (the paper decays it), plain linear
output layers, and batch size / updates per step chosen for parallel environments.
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

NAME = "simbav2"
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
SHIFT = 3.0  # constant input feature before the first projection onto the sphere


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class SimbaV2Config(TrainConfig):
    """SimbaV2 hyperparameters (SAC options as in :class:`okapi.sac.SACConfig`).

    Attributes:
        critic_width: Hidden width of each critic.
        critic_blocks: Residual LERP blocks per critic.
        actor_width: Hidden width of the actor.
        actor_blocks: Residual LERP blocks of the actor.
        num_atoms: Atoms of the categorical critic.
        v_max: Support of the categorical critic is ``[-v_max, v_max]`` (scaled returns).
        normalize_rewards: Divide rewards by the running std of the discounted return.
    """

    num_envs: int = 128
    total_timesteps: int = 10_000_000
    buffer_size: int = 4_194_304
    batch_size: int = 512
    updates_per_step: int = 8
    learning_starts: int = 8192
    learning_rate: float = 1e-4
    max_grad_norm: float | None = None
    gamma: float = 0.99
    n_step: int = 1
    tau: float = 0.005
    init_alpha: float = 0.01
    target_entropy: float | None = None
    reward_scaling: float = 1.0
    critic_width: int = 512
    critic_blocks: int = 2
    actor_width: int = 128
    actor_blocks: int = 1
    num_atoms: int = 101
    v_max: float = 5.0
    normalize_rewards: bool = True


# ── Networks ─────────────────────────────────────────────────────────────────


def l2_normalize(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    """Project onto the unit sphere (finite gradient at 0, unlike ``x / norm(x)``)."""
    return x * jax.lax.rsqrt(jnp.maximum(jnp.sum(x * x, axis, keepdims=True), eps**2))


class HyperLinear(nnx.Module):
    """Bias-free linear map whose weight columns have unit norm."""

    def __init__(self, in_size: int, out_size: int, *, rngs: nnx.Rngs):
        w = nnx.initializers.orthogonal()(rngs.params(), (in_size, out_size))
        self.kernel = nnx.Param(l2_normalize(w, axis=0))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.kernel[...]

    def project(self) -> None:
        self.kernel[...] = l2_normalize(self.kernel[...], axis=0)


class SimbaV2Net(nnx.Module):
    """Hyperspherical embedding, ``num_blocks`` LERP residual blocks and a linear output."""

    def __init__(self, in_size: int, width: int, num_blocks: int, out_size: int,
                 *, rngs: nnx.Rngs):
        self.embed = HyperLinear(in_size + 1, width, rngs=rngs)
        self.embed_scale = nnx.Param(jnp.ones(width))
        self.up = nnx.List([HyperLinear(width, 4 * width, rngs=rngs) for _ in range(num_blocks)])
        self.up_scale = nnx.List([nnx.Param(jnp.ones(4 * width)) for _ in range(num_blocks)])
        self.down = nnx.List([HyperLinear(4 * width, width, rngs=rngs) for _ in range(num_blocks)])
        self.alpha = nnx.List([nnx.Param(jnp.full(width, 1.0 / (num_blocks + 1)))
                               for _ in range(num_blocks)])
        self.head = HyperLinear(width, width, rngs=rngs)
        self.head_scale = nnx.Param(jnp.ones(width))
        self.out = nnx.Linear(width, out_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = l2_normalize(jnp.concatenate([x, jnp.full(x.shape[:-1] + (1,), SHIFT)], -1))
        h = l2_normalize(self.embed_scale[...] * self.embed(x))
        for up, up_scale, down, alpha in zip(self.up, self.up_scale, self.down, self.alpha):
            u = jax.nn.relu(up_scale[...] * up(h))
            h = l2_normalize(h + alpha[...] * (l2_normalize(down(u)) - h))
        h = l2_normalize(self.head_scale[...] * self.head(h))
        return self.out(h)


def project_weights(module: nnx.Module) -> None:
    """Put every hyperspherical weight matrix of ``module`` back onto the sphere."""
    for _, m in nnx.iter_modules(module):
        if isinstance(m, HyperLinear):
            m.project()


def sample_tanh_normal(mean: jax.Array, log_std: jax.Array,
                       key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sample ``tanh(N(mean, std))`` and its log-density (summed over action dims)."""
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    log_prob = -0.5 * jnp.square((u - mean) / std) - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    log_prob = log_prob - 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    return jnp.tanh(u), log_prob.sum(-1)


class Actor(nnx.Module):
    """Tanh-Gaussian policy on a SimbaV2 network."""

    def __init__(self, obs_size: int, action_size: int, cfg: SimbaV2Config, *, rngs: nnx.Rngs):
        self.net = SimbaV2Net(obs_size, cfg.actor_width, cfg.actor_blocks, 2 * action_size,
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
    """Twin categorical critics; ``critic(obs, action)`` returns logits ``(2, batch, num_atoms)``.

    The action is scaled by ``sqrt(obs_size / action_size)`` so that it weighs as much
    as the observation after the input is projected onto the sphere.
    """

    def __init__(self, obs_size: int, action_size: int, cfg: SimbaV2Config, *, rngs: nnx.Rngs):
        self.qs = nnx.List([
            SimbaV2Net(obs_size + action_size, cfg.critic_width, cfg.critic_blocks,
                       cfg.num_atoms, rngs=rngs)
            for _ in range(2)
        ])
        self.action_scale = math.sqrt(obs_size / action_size)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, self.action_scale * action], -1)
        return jnp.stack([q(x) for q in self.qs])


class Temperature(nnx.Module):
    def __init__(self, init_alpha: float):
        self.log_alpha = nnx.Param(jnp.log(jnp.asarray(init_alpha, jnp.float32)))

    def __call__(self) -> jax.Array:
        return jnp.exp(self.log_alpha[...])


class Networks(nnx.Module):
    """All learned modules and their optimizers, split / merged as one state."""

    def __init__(self, cfg: SimbaV2Config, policy_obs_size: int, value_obs_size: int,
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


def project_categorical(atoms: jax.Array, probs: jax.Array, support: jax.Array) -> jax.Array:
    """C51 projection of a distribution (``probs`` on ``atoms``) onto the fixed ``support``."""
    v_min, v_max, n = support[0], support[-1], support.shape[0]
    b = (jnp.clip(atoms, v_min, v_max) - v_min) / ((v_max - v_min) / (n - 1))
    lo, hi = jnp.floor(b), jnp.ceil(b)
    w_lo = jnp.where(hi == lo, 1.0, hi - b)
    w_hi = b - lo
    one_hot = lambda i: jax.nn.one_hot(i.astype(jnp.int32), n)
    return (probs[..., None] * (w_lo[..., None] * one_hot(lo) + w_hi[..., None] * one_hot(hi))).sum(-2)


def critic_loss(critic: Critic, actor: Actor, target_critic: Critic, alpha: jax.Array,
                batch: dict[str, Any], support: jax.Array, pk: str, vk: str,
                key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Cross-entropy towards the projected soft n-step target distribution."""
    next_action, next_log_prob = actor.sample(batch["next_obs"][pk], key)
    next_logits = target_critic(batch["next_obs"][vk], next_action)  # (2, B, N)
    next_q = (jax.nn.softmax(next_logits) * support).sum(-1)
    lower = jnp.argmin(next_q, 0)  # clipped double-Q: distribution with the lower mean
    next_probs = jax.nn.softmax(jnp.take_along_axis(next_logits, lower[None, :, None], 0)[0])
    atoms = batch["reward"][:, None] + batch["discount"][:, None] * (
        support[None] - alpha * next_log_prob[:, None])
    target = jax.lax.stop_gradient(project_categorical(atoms, next_probs, support))
    logits = critic(batch["obs"][vk], batch["action"])
    loss = -(target[None] * jax.nn.log_softmax(logits)).sum(-1).mean(-1).sum()
    q = (jax.nn.softmax(logits) * support).sum(-1)
    return loss, {"loss/critic": loss, "train/q_mean": q.mean()}


def actor_loss(actor: Actor, critic: Critic, alpha: jax.Array, batch: dict[str, Any],
               support: jax.Array, pk: str, vk: str,
               key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Maximize the soft value under the lower of both critics' means."""
    action, log_prob = actor.sample(batch["obs"][pk], key)
    q = (jax.nn.softmax(critic(batch["obs"][vk], action)) * support).sum(-1).min(0)
    return (alpha * log_prob - q).mean(), log_prob


def temperature_loss(temperature: Temperature, log_prob: jax.Array,
                     target_entropy: float) -> jax.Array:
    return -(temperature.log_alpha[...] * jax.lax.stop_gradient(log_prob + target_entropy)).mean()


def update(nets: Networks, batch: dict[str, Any], cfg: SimbaV2Config, support: jax.Array,
           pk: str, vk: str, target_entropy: float, key: jax.Array) -> dict[str, jax.Array]:
    """One gradient step of critic, actor and temperature, weight projection, target update."""
    critic_key, actor_key = jax.random.split(key)
    alpha = nets.temperature()
    (_, metrics), grads = nnx.value_and_grad(critic_loss, has_aux=True)(
        nets.critic, nets.actor, nets.target_critic, alpha, batch, support, pk, vk, critic_key)
    nets.critic_opt.update(nets.critic, grads)
    project_weights(nets.critic)

    (loss, log_prob), grads = nnx.value_and_grad(actor_loss, has_aux=True)(
        nets.actor, nets.critic, alpha, batch, support, pk, vk, actor_key)
    nets.actor_opt.update(nets.actor, grads)
    project_weights(nets.actor)

    alpha_loss, grads = nnx.value_and_grad(temperature_loss)(nets.temperature, log_prob,
                                                            target_entropy)
    nets.temperature_opt.update(nets.temperature, grads)

    target = jax.tree.map(lambda t, s: (1.0 - cfg.tau) * t + cfg.tau * s,
                          nnx.state(nets.target_critic), nnx.state(nets.critic))
    nnx.update(nets.target_critic, target)
    project_weights(nets.target_critic)
    return {**metrics, "loss/actor": loss, "loss/alpha": alpha_loss,
            "train/alpha": nets.temperature(), "train/entropy": -log_prob.mean()}


# ── Training ─────────────────────────────────────────────────────────────────


class SimbaV2State(NamedTuple):
    nets: nnx.State
    obs_stats: envs.ObsStats
    return_stats: running_statistics.RunningStatisticsState  # of the discounted return
    discounted_return: jax.Array  # (num_envs,) running discounted return
    buffer: replay.ReplayBuffer
    env_state: Any
    first: jax.Array
    key: jax.Array


def train(
    cfg: SimbaV2Config,
    env_factory: envs.EnvFactory | None = None,
    log_callback: LogCallback | None = None,
    resume_from: str | Path | None = None,
) -> Policy:
    """Train SimbaV2 and return the final policy (see :func:`okapi.sac.train` for arguments)."""
    s = setup(cfg, NAME, env_factory, log_callback, autoreset=False)
    env, pk, vk, n = s.env, s.policy_key, s.value_key, cfg.num_envs
    keys = (pk, vk)
    key, net_key = jax.random.split(s.key)
    nets = Networks(cfg, s.obs_shapes[pk][-1], s.obs_shapes[vk][-1], s.action_size, net_key)
    graphdef, nets_state = nnx.split(nets)
    actor_graphdef = nnx.split(nets.actor)[0]
    obs = envs.select_obs(s.env_state.obs, keys)
    obs_stats = envs.init_obs_stats(obs)
    return_stats = running_statistics.init_state(())
    support = jnp.linspace(-cfg.v_max, cfg.v_max, cfg.num_atoms)
    target_entropy = (-0.5 * s.action_size if cfg.target_entropy is None
                      else cfg.target_entropy)
    if resume_from is not None:
        _, restored = checkpoint.load(resume_from, {"nets": nets_state, "obs_stats": obs_stats,
                                                    "return_stats": return_stats})
        nets_state, obs_stats = restored["nets"], restored["obs_stats"]
        return_stats = restored["return_stats"]

    row = {"obs": obs, "action": jnp.zeros((n, s.action_size)), "reward": jnp.zeros(n),
           "terminal": jnp.zeros(n), "last": jnp.zeros(n), "first": jnp.zeros(n)}
    buffer = replay.init(row, cfg.buffer_size // n)

    def collect(state: SimbaV2State, nets: Networks, random: bool) -> SimbaV2State:
        """Store the current step, track the discounted return and act in all environments."""
        key, action_key, reset_key = jax.random.split(state.key, 3)
        env_state = state.env_state
        obs = envs.select_obs(env_state.obs, keys)
        if random:
            action = jax.random.uniform(action_key, (n, s.action_size), minval=-1.0, maxval=1.0)
        else:
            x = envs.normalize_obs(obs, state.obs_stats)[pk]
            action, _ = nets.actor.sample(x, action_key)
        reward = env_state.reward * cfg.reward_scaling
        buffer = replay.add(state.buffer, {
            "obs": obs, "action": action, "reward": reward,
            "terminal": envs.termination(env_state), "last": env_state.done,
            "first": state.first})
        ret = state.discounted_return * cfg.gamma * (1.0 - state.first) + reward
        next_state = envs.step_or_reset(env, env_state, action, reset_key)
        return state._replace(
            obs_stats=envs.update_obs_stats(state.obs_stats, obs),
            return_stats=running_statistics.update(state.return_stats, ret),
            discounted_return=ret, buffer=buffer, env_state=next_state,
            first=env_state.done, key=key)

    def learn(state: SimbaV2State, nets: Networks) -> tuple[SimbaV2State, dict[str, jax.Array]]:
        scale = state.return_stats.std if cfg.normalize_rewards else 1.0
        metrics = {}
        key = state.key
        for _ in range(cfg.updates_per_step):
            key, sample_key, update_key = jax.random.split(key, 3)
            batch = replay.sample_transitions(state.buffer, sample_key, cfg.batch_size,
                                              cfg.n_step, cfg.gamma)
            batch["reward"] = batch["reward"] / scale
            batch["obs"] = envs.normalize_obs(batch["obs"], state.obs_stats)
            batch["next_obs"] = envs.normalize_obs(batch["next_obs"], state.obs_stats)
            metrics = update(nets, batch, cfg, support, pk, vk, target_entropy, update_key)
        metrics["train/reward_scale"] = jnp.asarray(scale)
        return state._replace(key=key), metrics

    def make_train_fn(num_iterations: int, random: bool):
        def iteration(state: SimbaV2State, _):
            nets = nnx.merge(graphdef, state.nets)
            state = collect(state, nets, random)
            metrics = {}
            if not random:
                state, metrics = learn(state, nets)
            return state._replace(nets=nnx.split(nets)[1]), metrics

        @jax.jit(donate_argnums=0)  # update the replay buffer in place
        def run(buffer: replay.ReplayBuffer, state: SimbaV2State):
            state, metrics = jax.lax.scan(iteration, state._replace(buffer=buffer), None,
                                          num_iterations)
            return state.buffer, state._replace(buffer=None), jax.tree.map(lambda x: x[-1], metrics)

        def train_fn(state: SimbaV2State) -> tuple[SimbaV2State, dict[str, jax.Array]]:
            buffer, state, metrics = run(state.buffer, state._replace(buffer=None))
            return state._replace(buffer=buffer), metrics

        return train_fn

    state = SimbaV2State(nets_state, obs_stats, return_stats, jnp.zeros(n), buffer,
                         s.env_state, jnp.ones(n), key)
    warmup_iterations = max(math.ceil(cfg.learning_starts / n), cfg.n_step + 2)
    state, _ = make_train_fn(warmup_iterations, random=True)(state)
    iterations = max(1, min(cfg.total_timesteps // (cfg.num_logs * n), 1000))

    def policy_params(state: SimbaV2State) -> dict[str, Any]:
        return {"actor": nnx.state(nnx.merge(graphdef, state.nets).actor),
                "obs_stats": state.obs_stats}

    apply = make_policy_apply(actor_graphdef, pk)
    evaluate = envs.make_evaluator(cfg, s.eval_env, apply, lambda n: ())
    state, _ = train_loop(
        cfg, s.logger, state, make_train_fn(iterations, random=False), iterations * n,
        evaluate, policy_params,
        lambda st: {"nets": st.nets, "obs_stats": st.obs_stats, "return_stats": st.return_stats},
        jax.random.key(cfg.seed + 1), warmup_iterations * n)
    s.logger.close()
    return Policy(NAME, policy_params(state), apply)


def load_policy(run_dir: str | Path, step: int | None = None) -> Policy:
    """Rebuild the policy saved in ``run_dir`` (latest checkpoint unless ``step`` is given)."""
    meta = checkpoint.load_metadata(run_dir)
    cfg = SimbaV2Config.from_dict(meta["config"])
    spec = meta["spec"]
    pk, vk = spec["policy_obs_key"], spec["value_obs_key"]
    shapes = {k: tuple(v) for k, v in spec["obs_shapes"].items()}
    nets = Networks(cfg, shapes[pk][-1], shapes[vk][-1], spec["action_size"], jax.random.key(0))
    graphdef, nets_state = nnx.split(nets)
    obs_stats = {k: running_statistics.init_state(v[-1:]) for k, v in shapes.items()}
    target = {"nets": nets_state, "obs_stats": obs_stats,
              "return_stats": running_statistics.init_state(())}
    _, restored = checkpoint.load(run_dir, target, step)
    actor_graphdef, actor_state = nnx.split(nnx.merge(graphdef, restored["nets"]).actor)
    params = {"actor": actor_state, "obs_stats": restored["obs_stats"]}
    return Policy(NAME, params, make_policy_apply(actor_graphdef, pk))


Config = SimbaV2Config
