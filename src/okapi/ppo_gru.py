"""PPO with recurrent (GRU) actor and critic.

Identical to :mod:`okapi.ppo` (Brax-style PPO) except that actor and critic
each insert a GRU after their first hidden layer. Hidden states are reset
where an episode starts. Each update replays the collected sequences from the
hidden states stored at the start of the rollout (truncated BPTT over
``num_steps``), and minibatches are formed from whole environment sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from okapi.common import checkpoint, envs, running_statistics
from okapi.common.config import TrainConfig
from okapi.common.logger import LogCallback
from okapi.common.networks import ACTIVATIONS, MLP
from okapi.common.policy import Policy
from okapi.common.training import setup, train_loop

NAME = "ppo_gru"


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class PPOGRUConfig(TrainConfig):
    """PPO-GRU hyperparameters (PPO options as in :class:`okapi.ppo.PPOConfig`).

    Attributes:
        actor_hidden_sizes: Policy widths; the GRU follows the first layer.
        critic_hidden_sizes: Value widths; the GRU follows the first layer.
        gru_size: Hidden units of each GRU.
    """

    num_envs: int = 8192
    total_timesteps: int = 200_000_000
    num_steps: int = 20
    num_rollout_rounds: int = 1
    num_minibatches: int = 32
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float | None = 1.0
    reward_scaling: float = 1.0
    norm_adv: bool = True
    actor_hidden_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "swish"
    gru_size: int = 256
    warmup_stats: bool = True

    @property
    def steps_per_iteration(self) -> int:
        return self.num_envs * self.num_steps * self.num_rollout_rounds


# ── Networks ─────────────────────────────────────────────────────────────────


class NormalTanhDistribution:
    """Gaussian with softplus scale, squashed by tanh (Brax ``NormalTanhDistribution``)."""

    MIN_STD: float = 0.001

    def __init__(self, params: jax.Array):
        self.loc, raw_scale = jnp.split(params, 2, axis=-1)
        self.scale = jax.nn.softplus(raw_scale) + self.MIN_STD

    @staticmethod
    def _tanh_log_jacobian(x: jax.Array) -> jax.Array:
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def sample_raw(self, key: jax.Array) -> jax.Array:
        return self.loc + self.scale * jax.random.normal(key, self.loc.shape)

    @staticmethod
    def postprocess(raw_action: jax.Array) -> jax.Array:
        return jnp.tanh(raw_action)

    def log_prob(self, raw_action: jax.Array) -> jax.Array:
        gaussian = (-0.5 * jnp.square((raw_action - self.loc) / self.scale)
                    - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(self.scale))
        return jnp.sum(gaussian - self._tanh_log_jacobian(raw_action), axis=-1)

    def entropy(self, key: jax.Array) -> jax.Array:
        gaussian = 0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return jnp.sum(gaussian + self._tanh_log_jacobian(self.sample_raw(key)), axis=-1)

    @property
    def mode(self) -> jax.Array:
        return jnp.tanh(self.loc)


def reset_hidden(h: jax.Array, first: jax.Array) -> jax.Array:
    """Zero the hidden state of environments whose episode starts now."""
    return jnp.where(first[..., None] > 0, 0.0, h)


class RecurrentMLP(nnx.Module):
    """``Linear → activation → GRU → MLP``: an MLP with a GRU after its first layer."""

    def __init__(self, in_size: int, hidden_sizes: list[int], gru_size: int, out_size: int,
                 activation: str, *, rngs: nnx.Rngs):
        self.encoder = nnx.Linear(in_size, hidden_sizes[0],
                                  kernel_init=nnx.initializers.lecun_uniform(), rngs=rngs)
        self.gru = nnx.GRUCell(hidden_sizes[0], gru_size, rngs=rngs)
        self.head = MLP(gru_size, hidden_sizes[1:], out_size, activation, rngs=rngs)
        self.activation = ACTIVATIONS[activation]

    def __call__(self, h: jax.Array, x: jax.Array, first: jax.Array):
        """One step: returns ``(new_hidden, output)``."""
        h, y = self.gru(reset_hidden(h, first), self.activation(self.encoder(x)))
        return h, self.head(y)


class Hidden(NamedTuple):
    actor: jax.Array  # (..., gru_size)
    critic: jax.Array  # (..., gru_size)


class RecurrentActorCritic(nnx.Module):
    def __init__(self, policy_obs_size: int, value_obs_size: int, action_size: int,
                 cfg: PPOGRUConfig, *, rngs: nnx.Rngs):
        self.policy_net = RecurrentMLP(policy_obs_size, cfg.actor_hidden_sizes, cfg.gru_size,
                                       2 * action_size, cfg.activation, rngs=rngs)
        self.value_net = RecurrentMLP(value_obs_size, cfg.critic_hidden_sizes, cfg.gru_size, 1,
                                      cfg.activation, rngs=rngs)

    def policy(self, h: jax.Array, obs: jax.Array, first: jax.Array):
        h, out = self.policy_net(h, obs, first)
        return h, NormalTanhDistribution(out)

    def value(self, h: jax.Array, obs: jax.Array, first: jax.Array):
        h, out = self.value_net(h, obs, first)
        return h, out.squeeze(-1)


def make_model(cfg: PPOGRUConfig, obs_shapes: dict[str, tuple[int, ...]], policy_key: str,
               value_key: str, action_size: int, key: jax.Array):
    model = RecurrentActorCritic(obs_shapes[policy_key][-1], obs_shapes[value_key][-1],
                                 action_size, cfg, rngs=nnx.Rngs(key))
    tx = optax.adam(cfg.learning_rate, eps=1e-5)
    if cfg.max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), tx)
    return model, nnx.Optimizer(model, tx, wrt=nnx.Param)


def make_policy_apply(graphdef: nnx.GraphDef, policy_key: str):
    """``apply(params, carry, obs, first, key, deterministic)``; ``carry`` is the actor hidden state."""

    def apply(params, carry, obs, first, key, deterministic):
        model = nnx.merge(graphdef, params["model"])
        x = envs.select_obs(obs, (policy_key,))
        x = envs.normalize_obs(x, {policy_key: params["obs_stats"][policy_key]})[policy_key]
        h, dist = model.policy(carry, x, first)
        action = dist.mode if deterministic else dist.postprocess(dist.sample_raw(key))
        return action, h

    return apply


# ── Losses ───────────────────────────────────────────────────────────────────


class Transition(NamedTuple):
    obs: envs.Obs
    first: jax.Array  # obs[t] starts an episode
    raw_action: jax.Array
    logprob: jax.Array
    reward: jax.Array
    done: jax.Array
    truncation: jax.Array


def compute_gae(
    rewards: jax.Array, values: jax.Array, dones: jax.Array, truncations: jax.Array,
    bootstrap_value: jax.Array, gamma: float, gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Brax GAE (see :func:`okapi.ppo.compute_gae`)."""
    trunc_mask = 1.0 - truncations
    termination = dones * trunc_mask
    values_next = jnp.concatenate([values[1:], bootstrap_value[None]], axis=0)
    deltas = (rewards + gamma * (1.0 - termination) * values_next - values) * trunc_mask

    def step(acc, x):
        delta, term, tmask = x
        acc = delta + gamma * (1.0 - term) * tmask * gae_lambda * acc
        return acc, acc

    _, vs_minus_v = jax.lax.scan(step, jnp.zeros_like(bootstrap_value),
                                 (deltas, termination, trunc_mask), reverse=True)
    vs = vs_minus_v + values
    vs_next = jnp.concatenate([vs[1:], bootstrap_value[None]], axis=0)
    advantages = (rewards + gamma * (1.0 - termination) * vs_next - values) * trunc_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def ppo_loss(model: RecurrentActorCritic, traj: Transition, h0: Hidden,
             bootstrap_obs: jax.Array, bootstrap_first: jax.Array, obs_stats: envs.ObsStats,
             policy_key: str, value_key: str, cfg: PPOGRUConfig,
             key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    """PPO loss on ``[T, B]`` sequences replayed from the hidden states ``h0``."""
    T, B = traj.reward.shape
    obs = envs.normalize_obs(traj.obs, obs_stats)

    def step(h: Hidden, x):
        p_obs, v_obs, first = x
        ha, dist_params = model.policy_net(h.actor, p_obs, first)
        hc, value = model.value(h.critic, v_obs, first)
        return Hidden(ha, hc), (dist_params, value)

    h, (dist_params, values) = jax.lax.scan(
        step, h0, (obs[policy_key], obs[value_key], traj.first))
    boot_obs = envs.normalize_obs({value_key: bootstrap_obs},
                                  {value_key: obs_stats[value_key]})[value_key]
    _, bootstrap_value = model.value(h.critic, boot_obs, bootstrap_first)
    targets, advantages = compute_gae(traj.reward, values, traj.done, traj.truncation,
                                      bootstrap_value, cfg.gamma, cfg.gae_lambda)
    if cfg.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dist = NormalTanhDistribution(dist_params)
    logprob = dist.log_prob(traj.raw_action)
    ratio = jnp.exp(logprob - traj.logprob)
    clipped = jnp.clip(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
    policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()
    value_loss = 0.5 * cfg.vf_coef * jnp.mean(jnp.square(values - targets))
    entropy = dist.entropy(key).mean()
    total = policy_loss + value_loss - cfg.ent_coef * entropy
    return total, {
        "loss/policy": policy_loss,
        "loss/value": value_loss,
        "loss/entropy": entropy,
        "loss/total": total,
        "train/clip_frac": (jnp.abs(ratio - 1.0) > cfg.clip_coef).mean(),
        "train/approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
    }


# ── Training ─────────────────────────────────────────────────────────────────


class PPOGRUState(NamedTuple):
    model_opt: nnx.State
    obs_stats: envs.ObsStats
    env_state: Any
    hidden: Hidden  # hidden states before the current obs
    first: jax.Array  # current obs starts an episode
    key: jax.Array


def train(
    cfg: PPOGRUConfig,
    env_factory: envs.EnvFactory | None = None,
    log_callback: LogCallback | None = None,
    resume_from: str | Path | None = None,
) -> Policy:
    """Train PPO-GRU and return the final policy (see :func:`okapi.ppo.train` for arguments)."""
    s = setup(cfg, NAME, env_factory, log_callback, autoreset=True)
    env, pk, vk = s.env, s.policy_key, s.value_key
    keys = (pk, vk)
    key, model_key = jax.random.split(s.key)
    model, optimizer = make_model(cfg, s.obs_shapes, pk, vk, s.action_size, model_key)
    graphdef, model_opt = nnx.split((model, optimizer))
    model_graphdef = nnx.split(model)[0]
    obs_stats = envs.init_obs_stats(envs.select_obs(s.env_state.obs, keys))

    start_step = 0
    if resume_from is not None:
        start_step, restored = checkpoint.load(
            resume_from, {"model_opt": model_opt, "obs_stats": obs_stats})
        model_opt, obs_stats = restored["model_opt"], restored["obs_stats"]

    def zeros_hidden(n: int) -> Hidden:
        return Hidden(jnp.zeros((n, cfg.gru_size)), jnp.zeros((n, cfg.gru_size)))

    def collect(model: RecurrentActorCritic, obs_stats, env_state, hidden, first, key):
        """One rollout of ``num_steps`` carrying actor and critic hidden states."""

        def step(carry, _):
            env_state, hidden, first, key = carry
            key, action_key = jax.random.split(key)
            obs = envs.select_obs(env_state.obs, keys)
            norm = envs.normalize_obs(obs, obs_stats)
            ha, dist = model.policy(hidden.actor, norm[pk], first)
            hc, _ = model.value(hidden.critic, norm[vk], first)
            raw_action = dist.sample_raw(action_key)
            next_state = env.step(env_state, dist.postprocess(raw_action))
            truncation = next_state.info.get("truncation", jnp.zeros_like(next_state.done))
            transition = Transition(obs, first, raw_action, dist.log_prob(raw_action),
                                    next_state.reward * cfg.reward_scaling, next_state.done,
                                    truncation)
            return (next_state, Hidden(ha, hc), next_state.done, key), transition

        (env_state, hidden, first, _), traj = jax.lax.scan(
            step, (env_state, hidden, first, key), None, cfg.num_steps)
        return env_state, hidden, first, traj

    @jax.jit
    def warmup(model_opt, obs_stats, env_state, key):
        model, _ = nnx.merge(graphdef, model_opt)
        n = cfg.num_envs
        *_, traj = collect(model, obs_stats, env_state, zeros_hidden(n), jnp.ones(n), key)
        return envs.update_obs_stats(obs_stats, traj.obs)

    if resume_from is None and cfg.warmup_stats:
        key, warmup_key = jax.random.split(key)
        obs_stats = warmup(model_opt, obs_stats, s.env_state, warmup_key)

    total_envs = cfg.num_envs * cfg.num_rollout_rounds
    mb_size = total_envs // cfg.num_minibatches

    @jax.jit
    def train_iteration(state: PPOGRUState) -> tuple[PPOGRUState, dict[str, jax.Array]]:
        model, optimizer = nnx.merge(graphdef, state.model_opt)
        key, rollout_key, update_key = jax.random.split(state.key, 3)

        def collect_round(carry, round_key):
            env_state, hidden, first = carry
            h0 = hidden
            env_state, hidden, first, traj = collect(model, state.obs_stats, env_state,
                                                     hidden, first, round_key)
            bootstrap = envs.select_obs(env_state.obs, (vk,))[vk]
            return (env_state, hidden, first), (traj, h0, bootstrap, first)

        (env_state, hidden, first), (rounds, h0, bootstrap, boot_first) = jax.lax.scan(
            collect_round, (state.env_state, state.hidden, state.first),
            jax.random.split(rollout_key, cfg.num_rollout_rounds))
        flat = lambda x: x.reshape(total_envs, *x.shape[2:])  # [R, N, ...] → [R*N, ...]
        batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1).reshape(
            cfg.num_steps, total_envs, *x.shape[3:]), rounds)
        h0, bootstrap, boot_first = jax.tree.map(flat, (h0, bootstrap, boot_first))
        obs_stats = envs.update_obs_stats(state.obs_stats, batch.obs)

        grad_fn = nnx.value_and_grad(ppo_loss, has_aux=True)

        def epoch(model_opt, epoch_key):
            perm_key, loss_key = jax.random.split(epoch_key)
            perm = jax.random.permutation(perm_key, total_envs).reshape(cfg.num_minibatches, mb_size)

            def minibatch(model_opt, x):
                idx, mb_key = x
                model, optimizer = nnx.merge(graphdef, model_opt)
                traj = jax.tree.map(lambda v: v[:, idx], batch)
                mb_h0 = jax.tree.map(lambda v: v[idx], h0)
                (_, metrics), grads = grad_fn(model, traj, mb_h0, bootstrap[idx],
                                              boot_first[idx], obs_stats, pk, vk, cfg, mb_key)
                optimizer.update(model, grads)
                return nnx.split((model, optimizer))[1], metrics

            return jax.lax.scan(minibatch, model_opt,
                                (perm, jax.random.split(loss_key, cfg.num_minibatches)))

        model_opt, metrics = jax.lax.scan(
            epoch, nnx.split((model, optimizer))[1],
            jax.random.split(update_key, cfg.update_epochs))
        metrics = jax.tree.map(jnp.mean, metrics)
        metrics["train/reward_per_step"] = batch.reward.mean() / cfg.reward_scaling
        return PPOGRUState(model_opt, obs_stats, env_state, hidden, first, key), metrics

    def policy_params(state: PPOGRUState) -> dict[str, Any]:
        model, _ = nnx.merge(graphdef, state.model_opt)
        return {"model": nnx.split(model)[1], "obs_stats": state.obs_stats}

    init_carry = lambda n: jnp.zeros((n, cfg.gru_size))
    apply = make_policy_apply(model_graphdef, pk)
    evaluate = envs.make_evaluator(cfg, s.eval_env, apply, init_carry)
    state = PPOGRUState(model_opt, obs_stats, s.env_state, zeros_hidden(cfg.num_envs),
                        jnp.ones(cfg.num_envs), key)
    state, _ = train_loop(
        cfg, s.logger, state, train_iteration, cfg.steps_per_iteration, evaluate,
        policy_params, lambda st: {"model_opt": st.model_opt, "obs_stats": st.obs_stats},
        jax.random.key(cfg.seed + 1), start_step)
    s.logger.close()
    return Policy(NAME, policy_params(state), apply, init_carry)


def load_policy(run_dir: str | Path, step: int | None = None) -> Policy:
    """Rebuild the policy saved in ``run_dir`` (latest checkpoint unless ``step`` is given)."""
    meta = checkpoint.load_metadata(run_dir)
    cfg = PPOGRUConfig.from_dict(meta["config"])
    spec = meta["spec"]
    pk, vk = spec["policy_obs_key"], spec["value_obs_key"]
    shapes = {k: tuple(v) for k, v in spec["obs_shapes"].items()}
    model, optimizer = make_model(cfg, shapes, pk, vk, spec["action_size"], jax.random.key(0))
    graphdef, model_opt = nnx.split((model, optimizer))
    obs_stats = {k: running_statistics.init_state(v[-1:]) for k, v in shapes.items()}
    _, restored = checkpoint.load(run_dir, {"model_opt": model_opt, "obs_stats": obs_stats}, step)
    model, _ = nnx.merge(graphdef, restored["model_opt"])
    model_graphdef, model_state = nnx.split(model)
    params = {"model": model_state, "obs_stats": restored["obs_stats"]}
    return Policy(NAME, params, make_policy_apply(model_graphdef, pk),
                  lambda n: jnp.zeros((n, cfg.gru_size)))


Config = PPOGRUConfig
