"""DreamerV3 (Hafner et al., 2023) for parallel environments and proprioceptive observations.

A world model and an actor-critic trained purely in imagination:

* **World model** (RSSM): a LayerNorm GRU deterministic state plus categorical
  stochastic latents (``stoch`` groups of ``classes``) with 1 % uniform mixing,
  trained to reconstruct symlog observations, predict symlog two-hot rewards and
  the continue flag, with KL balancing and free bits.
* **Actor-critic** on ``horizon``-step latent rollouts: lambda-returns, returns
  normalized by their 5th-95th percentile range (EMA), a two-hot critic
  regularized towards its slow (EMA) copy, and a Normal policy with bounded std
  trained by backpropagating returns through the learned dynamics.

Every iteration, all environments take one step into a sequence replay buffer,
followed by ``updates_per_step`` updates on ``batch_size`` sequences of
``seq_len`` steps. Terminal observations are stored (continue = 0) before the
next episode starts.

Simplifications: a dense LayerNorm GRU instead of the block-diagonal one, zero
initial latent states, fixed learning rates and sizes chosen for parallel envs.
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

from okapi.common import checkpoint, envs, replay
from okapi.common.config import TrainConfig
from okapi.common.logger import LogCallback
from okapi.common.networks import MLP
from okapi.common.policy import Policy
from okapi.common.training import setup, train_loop

NAME = "dreamerv3"
sg = jax.lax.stop_gradient


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class DreamerV3Config(TrainConfig):
    """DreamerV3 hyperparameters.

    Attributes:
        buffer_size: Replay capacity in steps (split evenly over environments).
        batch_size: Sequences per update.
        seq_len: Steps per sequence.
        updates_per_step: Updates per environment step (of all environments).
        learning_starts: Random-action steps collected before learning.
        deter: Size of the deterministic (GRU) state.
        stoch: Number of categorical latents.
        classes: Classes per categorical latent.
        hidden: Width of all MLPs.
        num_layers: Hidden layers of all MLPs.
        unimix: Uniform mixing of the latent and policy distributions.
        free_nats: Free bits of the KL terms.
        dyn_scale: Weight of the dynamics KL (trains the prior).
        rep_scale: Weight of the representation KL (trains the posterior).
        model_lr: World model learning rate.
        actor_lr: Actor learning rate.
        critic_lr: Critic learning rate.
        model_grad_clip: Gradient-norm clipping of the world model.
        ac_grad_clip: Gradient-norm clipping of actor and critic.
        horizon: Imagination horizon.
        gamma: Discount factor.
        gae_lambda: Lambda of the lambda-returns.
        ent_coef: Entropy bonus weight (returns are normalized to ~[0, 1]).
        num_bins: Bins of the two-hot reward and value distributions.
        slow_critic_rate: EMA rate of the slow critic.
        slow_critic_reg: Weight of the regularization towards the slow critic.
        return_norm_decay: EMA decay of the return scale.
        min_std: Lower bound of the policy std.
        max_std: Upper bound of the policy std.
    """

    num_envs: int = 32
    total_timesteps: int = 2_000_000
    buffer_size: int = 1_000_000
    batch_size: int = 16
    seq_len: int = 64
    updates_per_step: int = 1
    learning_starts: int = 8192
    deter: int = 512
    stoch: int = 32
    classes: int = 32
    hidden: int = 512
    num_layers: int = 2
    unimix: float = 0.01
    free_nats: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    model_grad_clip: float = 1000.0
    ac_grad_clip: float = 100.0
    horizon: int = 15
    gamma: float = 0.997
    gae_lambda: float = 0.95
    ent_coef: float = 3e-4
    num_bins: int = 255
    slow_critic_rate: float = 0.02
    slow_critic_reg: float = 1.0
    return_norm_decay: float = 0.99
    min_std: float = 0.1
    max_std: float = 1.0


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


class Latent(NamedTuple):
    h: jax.Array  # (..., deter) deterministic state
    z: jax.Array  # (..., stoch * classes) one-hot stochastic state


# ── Networks ─────────────────────────────────────────────────────────────────


def mlp(
    cfg: DreamerV3Config,
    in_size: int,
    out_size: int,
    rngs: nnx.Rngs,
    layers: int | None = None,
    out_scale: float | None = None,
) -> MLP:
    """DreamerV3 MLP: ``Dense → LayerNorm → SiLU`` hidden layers and a linear output."""
    layers = cfg.num_layers if layers is None else layers
    return MLP(
        in_size,
        [cfg.hidden] * layers,
        out_size,
        "silu",
        layer_norm=True,
        out_scale=out_scale,
        rngs=rngs,
    )


class LayerNormGRU(nnx.Module):
    """GRU cell with LayerNorm on the gate pre-activations (DreamerV3)."""

    def __init__(self, in_size: int, size: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            in_size + size, 3 * size, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs
        )
        self.norm = nnx.LayerNorm(3 * size, rngs=rngs)

    def __call__(self, x: jax.Array, h: jax.Array) -> jax.Array:
        reset, cand, update = jnp.split(self.norm(self.linear(jnp.concatenate([x, h], -1))), 3, -1)
        cand = jnp.tanh(jax.nn.sigmoid(reset) * cand)
        update = jax.nn.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * h


class WorldModel(nnx.Module):
    """RSSM with observation encoder / decoder and reward / continue heads."""

    def __init__(self, cfg: DreamerV3Config, obs_size: int, action_size: int, *, rngs: nnx.Rngs):
        self.cfg = cfg
        S = cfg.stoch * cfg.classes
        F = cfg.deter + S
        self.encoder = mlp(cfg, obs_size, cfg.hidden, rngs)
        self.img_in = mlp(cfg, S + action_size, cfg.hidden, rngs, layers=1)
        self.gru = LayerNormGRU(cfg.hidden, cfg.deter, rngs=rngs)
        self.prior = mlp(cfg, cfg.deter, S, rngs, layers=1)
        self.post = mlp(cfg, cfg.deter + cfg.hidden, S, rngs, layers=1)
        self.decoder = mlp(cfg, F, obs_size, rngs)
        self.reward = mlp(cfg, F, cfg.num_bins, rngs, out_scale=0.0)
        self.cont = mlp(cfg, F, 1, rngs)

    def initial(self, batch_shape: tuple[int, ...]) -> Latent:
        S = self.cfg.stoch * self.cfg.classes
        return Latent(jnp.zeros((*batch_shape, self.cfg.deter)), jnp.zeros((*batch_shape, S)))

    @staticmethod
    def feat(s: Latent) -> jax.Array:
        return jnp.concatenate([s.h, s.z], -1)

    def _dist(self, logits: jax.Array) -> jax.Array:
        """Categorical probabilities ``(..., stoch, classes)`` with uniform mixing."""
        c = self.cfg
        probs = jax.nn.softmax(logits.reshape(*logits.shape[:-1], c.stoch, c.classes))
        return (1 - c.unimix) * probs + c.unimix / c.classes

    def _draw(self, probs: jax.Array, key: jax.Array) -> jax.Array:
        """One-hot sample with straight-through gradients, flattened to ``(..., stoch*classes)``."""
        one_hot = jax.nn.one_hot(jax.random.categorical(key, jnp.log(probs)), self.cfg.classes)
        one_hot = one_hot + probs - sg(probs)
        return one_hot.reshape(*one_hot.shape[:-2], -1)

    def _deter(self, s: Latent, action: jax.Array) -> jax.Array:
        return self.gru(self.img_in(jnp.concatenate([s.z, action], -1)), s.h)

    def img_step(self, s: Latent, action: jax.Array, key: jax.Array) -> Latent:
        """Prior transition: imagine the next latent without an observation."""
        h = self._deter(s, action)
        return Latent(h, self._draw(self._dist(self.prior(h)), key))

    def obs_step(
        self, s: Latent, prev_action: jax.Array, embed: jax.Array, first: jax.Array, key: jax.Array
    ) -> tuple[Latent, jax.Array, jax.Array]:
        """Posterior transition (reset where ``first``). Returns ``(latent, post, prior)``."""
        keep = (1.0 - first)[..., None]
        s, prev_action = Latent(s.h * keep, s.z * keep), prev_action * keep
        h = self._deter(s, prev_action)
        prior = self._dist(self.prior(h))
        post = self._dist(self.post(jnp.concatenate([h, embed], -1)))
        return Latent(h, self._draw(post, key)), post, prior

    def encode(self, obs: jax.Array) -> jax.Array:
        return self.encoder(symlog(obs))


class Actor(nnx.Module):
    """Normal policy on latent features: ``tanh`` mean and std in ``[min_std, max_std]``."""

    def __init__(self, cfg: DreamerV3Config, action_size: int, *, rngs: nnx.Rngs):
        self.net = mlp(
            cfg, cfg.deter + cfg.stoch * cfg.classes, 2 * action_size, rngs, out_scale=0.01
        )
        self.min_std, self.max_std = cfg.min_std, cfg.max_std

    def __call__(self, feat: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean, std = jnp.split(self.net(feat), 2, axis=-1)
        std = (self.max_std - self.min_std) * jax.nn.sigmoid(std + 2.0) + self.min_std
        return jnp.tanh(mean), std

    def sample(self, feat: jax.Array, key: jax.Array) -> jax.Array:
        mean, std = self(feat)
        return jnp.clip(mean + std * jax.random.normal(key, mean.shape), -1.0, 1.0)


class Critic(nnx.Module):
    """Two-hot value distribution over symlog-spaced bins."""

    def __init__(self, cfg: DreamerV3Config, *, rngs: nnx.Rngs):
        self.net = mlp(cfg, cfg.deter + cfg.stoch * cfg.classes, cfg.num_bins, rngs, out_scale=0.0)

    def __call__(self, feat: jax.Array) -> jax.Array:
        return self.net(feat)


class Networks(nnx.Module):
    """All learned modules and their optimizers, split / merged as one state."""

    def __init__(self, cfg: DreamerV3Config, obs_size: int, action_size: int, key: jax.Array):
        rngs = nnx.Rngs(key)
        self.world_model = WorldModel(cfg, obs_size, action_size, rngs=rngs)
        self.actor = Actor(cfg, action_size, rngs=rngs)
        self.critic = Critic(cfg, rngs=rngs)
        self.slow_critic = nnx.clone(self.critic)
        tx = lambda lr, clip: optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr, eps=1e-8))
        self.model_opt = nnx.Optimizer(
            self.world_model, tx(cfg.model_lr, cfg.model_grad_clip), wrt=nnx.Param
        )
        self.actor_opt = nnx.Optimizer(
            self.actor, tx(cfg.actor_lr, cfg.ac_grad_clip), wrt=nnx.Param
        )
        self.critic_opt = nnx.Optimizer(
            self.critic, tx(cfg.critic_lr, cfg.ac_grad_clip), wrt=nnx.Param
        )


class Agent(nnx.Module):
    """The modules needed to act: world model (for filtering) and actor."""

    def __init__(self, world_model: WorldModel, actor: Actor):
        self.world_model = world_model
        self.actor = actor


def agent_step(
    agent: Agent,
    latent: Latent,
    prev_action: jax.Array,
    obs: jax.Array,
    first: jax.Array,
    key: jax.Array,
    deterministic: bool,
) -> tuple[jax.Array, Latent]:
    """Filter the observation into the latent state and pick an action."""
    latent_key, action_key = jax.random.split(key)
    wm = agent.world_model
    latent, _, _ = wm.obs_step(latent, prev_action, wm.encode(obs), first, latent_key)
    feat = wm.feat(latent)
    action = agent.actor(feat)[0] if deterministic else agent.actor.sample(feat, action_key)
    return action, latent


def make_policy_apply(graphdef: nnx.GraphDef, policy_key: str):
    """``apply(params, carry, obs, first, key, deterministic)``; carry is ``(latent, prev_action)``."""

    def apply(params, carry, obs, first, key, deterministic):
        agent = nnx.merge(graphdef, params["agent"])
        latent, prev_action = carry
        x = envs.select_obs(obs, (policy_key,))[policy_key]
        action, latent = agent_step(agent, latent, prev_action, x, first, key, deterministic)
        return action, (latent, action)

    return apply


# ── Losses ───────────────────────────────────────────────────────────────────


def kl_categorical(p: jax.Array, q: jax.Array) -> jax.Array:
    """KL(p || q) summed over the latent groups."""
    return (p * (jnp.log(p) - jnp.log(q))).sum((-2, -1))


def world_model_loss(
    wm: WorldModel, seq: dict[str, jax.Array], bins: jax.Array, cfg: DreamerV3Config, key: jax.Array
):
    """Reconstruction, reward, continue and balanced KL losses on ``[L, B]`` sequences."""
    obs = seq["obs"]
    embed = wm.encode(obs)
    L, B = seq["first"].shape
    first = seq["first"].at[0].set(1.0)  # windows start from a reset state
    prev_action = jnp.concatenate([jnp.zeros_like(seq["action"][:1]), seq["action"][:-1]], 0)

    def step(s, x):
        e, a, f, k = x
        s, post, prior = wm.obs_step(s, a, e, f, k)
        return s, (s, post, prior)

    _, (latents, post, prior) = jax.lax.scan(
        step, wm.initial((B,)), (embed, prev_action, first, jax.random.split(key, L))
    )
    feat = wm.feat(latents)
    recon = 0.5 * jnp.square(wm.decoder(feat) - symlog(obs)).sum(-1)
    reward = -(twohot(symlog(seq["reward"]), bins) * jax.nn.log_softmax(wm.reward(feat))).sum(-1)
    cont = optax.sigmoid_binary_cross_entropy(wm.cont(feat)[..., 0], 1.0 - seq["terminal"])
    dyn = jnp.maximum(cfg.free_nats, kl_categorical(sg(post), prior))
    rep = jnp.maximum(cfg.free_nats, kl_categorical(post, sg(prior)))
    loss = (recon + reward + cont + cfg.dyn_scale * dyn + cfg.rep_scale * rep).mean()
    metrics = {
        "loss/world_model": loss,
        "loss/reconstruction": recon.mean(),
        "loss/reward": reward.mean(),
        "loss/continue": cont.mean(),
        "train/kl": kl_categorical(post, prior).mean(),
    }
    return loss, (latents, metrics)


def lambda_returns(
    reward: jax.Array, cont: jax.Array, value: jax.Array, gamma: float, lam: float
) -> jax.Array:
    """``reward, cont``: ``(H, N)`` on arriving at states 1..H; ``value``: ``(H+1, N)``."""

    def step(nxt, x):
        r, c, v = x
        ret = r + gamma * c * ((1 - lam) * v + lam * nxt)
        return ret, ret

    _, rets = jax.lax.scan(step, value[-1], (reward, cont, value[1:]), reverse=True)
    return rets


def actor_loss(
    actor: Actor,
    wm: WorldModel,
    critic: Critic,
    start: Latent,
    start_cont: jax.Array,
    return_scale: jax.Array,
    bins: jax.Array,
    cfg: DreamerV3Config,
    key: jax.Array,
):
    """Imagine ``horizon`` steps and maximize normalized lambda-returns plus entropy."""

    def step(s, k):
        action_key, latent_key = jax.random.split(k)
        f = wm.feat(s)
        _, std = actor(f)
        action = actor.sample(f, action_key)
        entropy = (jnp.log(std) + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1)
        return wm.img_step(s, action, latent_key), (f, entropy)

    last, (feats, entropy) = jax.lax.scan(step, start, jax.random.split(key, cfg.horizon))
    feats = jnp.concatenate([feats, wm.feat(last)[None]], 0)  # (H+1, N, F)
    reward = twohot_mean(wm.reward(feats[1:]), bins)
    cont = jax.nn.sigmoid(wm.cont(feats[1:])[..., 0])
    value = twohot_mean(critic(feats), bins)
    rets = lambda_returns(reward, cont, value, cfg.gamma, cfg.gae_lambda)
    weight = sg(jnp.cumprod(jnp.concatenate([start_cont[None], cfg.gamma * cont[:-1]], 0), 0))
    lo, hi = jnp.percentile(sg(rets), 5), jnp.percentile(sg(rets), 95)
    loss = -(weight * (rets / jnp.maximum(1.0, return_scale) + cfg.ent_coef * entropy)).mean()
    return loss, (sg(feats), sg(rets), weight, hi - lo, entropy.mean())


def critic_loss(
    critic: Critic,
    slow_critic: Critic,
    feats: jax.Array,
    rets: jax.Array,
    weight: jax.Array,
    bins: jax.Array,
    cfg: DreamerV3Config,
) -> jax.Array:
    """Two-hot cross-entropy towards lambda-returns, regularized towards the slow critic."""
    log_probs = jax.nn.log_softmax(critic(feats[:-1]))
    target = twohot(symlog(rets), bins)
    slow_target = jax.nn.softmax(slow_critic(feats[:-1]))
    loss = -(target * log_probs).sum(-1) - cfg.slow_critic_reg * (slow_target * log_probs).sum(-1)
    return (weight * loss).mean()


def update(
    nets: Networks,
    seq: dict[str, jax.Array],
    return_scale: jax.Array,
    bins: jax.Array,
    cfg: DreamerV3Config,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Update world model, actor and critic (in place); returns the new return scale."""
    model_key, imagine_key = jax.random.split(key)
    (_, (latents, metrics)), grads = nnx.value_and_grad(world_model_loss, has_aux=True)(
        nets.world_model, seq, bins, cfg, model_key
    )
    nets.model_opt.update(nets.world_model, grads)

    start = jax.tree.map(lambda x: sg(x.reshape(-1, x.shape[-1])), latents)
    start_cont = 1.0 - seq["terminal"].reshape(-1)
    (a_loss, (feats, rets, weight, ret_range, entropy)), grads = nnx.value_and_grad(
        actor_loss, has_aux=True
    )(
        nets.actor,
        nets.world_model,
        nets.critic,
        start,
        start_cont,
        return_scale,
        bins,
        cfg,
        imagine_key,
    )
    nets.actor_opt.update(nets.actor, grads)

    c_loss, grads = nnx.value_and_grad(critic_loss)(
        nets.critic, nets.slow_critic, feats, rets, weight, bins, cfg
    )
    nets.critic_opt.update(nets.critic, grads)
    slow = jax.tree.map(
        lambda s, p: (1 - cfg.slow_critic_rate) * s + cfg.slow_critic_rate * p,
        nnx.state(nets.slow_critic),
        nnx.state(nets.critic),
    )
    nnx.update(nets.slow_critic, slow)

    return_scale = cfg.return_norm_decay * return_scale + (1 - cfg.return_norm_decay) * ret_range
    metrics.update(
        {
            "loss/actor": a_loss,
            "loss/critic": c_loss,
            "train/return_scale": return_scale,
            "train/entropy": entropy,
            "train/imagined_return": rets.mean(),
        }
    )
    return return_scale, metrics


# ── Training ─────────────────────────────────────────────────────────────────


class DreamerV3State(NamedTuple):
    nets: nnx.State
    return_scale: jax.Array  # EMA of the 5th-95th percentile range of imagined returns
    buffer: replay.ReplayBuffer
    env_state: Any
    latent: Latent  # acting latent state before the current obs
    prev_action: jax.Array
    first: jax.Array
    key: jax.Array


def train(
    cfg: DreamerV3Config,
    env_factory: envs.EnvFactory | None = None,
    log_callback: LogCallback | None = None,
    resume_from: str | Path | None = None,
) -> Policy:
    """Train DreamerV3 and return the final policy.

    Args:
        cfg: Hyperparameters.
        env_factory: Optional callable returning a raw ``MjxEnv`` (replaces ``cfg.env_id``).
        log_callback: Optional ``callback(metrics, step)`` called on every log.
        resume_from: Run directory whose latest checkpoint initializes the networks
            (the replay buffer starts empty).
    """
    s = setup(cfg, NAME, env_factory, log_callback, autoreset=False)
    env, pk, n, A = s.env, s.policy_key, cfg.num_envs, s.action_size
    key, net_key = jax.random.split(s.key)
    nets = Networks(cfg, s.obs_shapes[pk][-1], A, net_key)
    graphdef, nets_state = nnx.split(nets)
    agent_graphdef = nnx.split(Agent(nets.world_model, nets.actor))[0]
    bins = jnp.linspace(-20.0, 20.0, cfg.num_bins)
    return_scale = jnp.ones(())
    if resume_from is not None:
        _, restored = checkpoint.load(
            resume_from, {"nets": nets_state, "return_scale": return_scale}
        )
        nets_state, return_scale = restored["nets"], restored["return_scale"]

    obs = envs.select_obs(s.env_state.obs, (pk,))[pk]
    row = {
        "obs": obs,
        "action": jnp.zeros((n, A)),
        "reward": jnp.zeros(n),
        "terminal": jnp.zeros(n),
        "last": jnp.zeros(n),
        "first": jnp.zeros(n),
    }
    buffer = replay.init(row, cfg.buffer_size // n)

    def collect(state: DreamerV3State, nets: Networks, random: bool) -> DreamerV3State:
        """Filter the current observation, store the step and act in all environments."""
        key, act_key, random_key, reset_key = jax.random.split(state.key, 4)
        env_state = state.env_state
        obs = envs.select_obs(env_state.obs, (pk,))[pk]
        action, latent = agent_step(
            Agent(nets.world_model, nets.actor),
            state.latent,
            state.prev_action,
            obs,
            state.first,
            act_key,
            False,
        )
        if random:
            action = jax.random.uniform(random_key, (n, A), minval=-1.0, maxval=1.0)
        buffer = replay.add(
            state.buffer,
            {
                "obs": obs,
                "action": action,
                "reward": env_state.reward,
                "terminal": envs.termination(env_state),
                "last": env_state.done,
                "first": state.first,
            },
        )
        next_state = envs.step_or_reset(env, env_state, action, reset_key)
        return state._replace(
            buffer=buffer,
            env_state=next_state,
            latent=latent,
            prev_action=action,
            first=env_state.done,
            key=key,
        )

    def learn(state: DreamerV3State, nets: Networks):
        key, return_scale, metrics = state.key, state.return_scale, {}
        for _ in range(cfg.updates_per_step):
            key, sample_key, update_key = jax.random.split(key, 3)
            seq = replay.sample_sequences(state.buffer, sample_key, cfg.batch_size, cfg.seq_len)
            return_scale, metrics = update(nets, seq, return_scale, bins, cfg, update_key)
        return state._replace(return_scale=return_scale, key=key), metrics

    def make_train_fn(num_iterations: int, random: bool):
        def iteration(state: DreamerV3State, _):
            nets = nnx.merge(graphdef, state.nets)
            state = collect(state, nets, random)
            metrics = {}
            if not random:
                state, metrics = learn(state, nets)
            return state._replace(nets=nnx.split(nets)[1]), metrics

        @jax.jit(donate_argnums=0)  # update the replay buffer in place
        def run(buffer: replay.ReplayBuffer, state: DreamerV3State):
            state, metrics = jax.lax.scan(
                iteration, state._replace(buffer=buffer), None, num_iterations
            )
            return state.buffer, state._replace(buffer=None), jax.tree.map(lambda x: x[-1], metrics)

        def train_fn(state: DreamerV3State):
            buffer, state, metrics = run(state.buffer, state._replace(buffer=None))
            return state._replace(buffer=buffer), metrics

        return train_fn

    state = DreamerV3State(
        nets_state,
        return_scale,
        buffer,
        s.env_state,
        nets.world_model.initial((n,)),
        jnp.zeros((n, A)),
        jnp.ones(n),
        key,
    )
    warmup_iterations = max(math.ceil(cfg.learning_starts / n), cfg.seq_len + 1)
    state, _ = make_train_fn(warmup_iterations, random=True)(state)
    iterations = max(1, min(cfg.total_timesteps // (cfg.num_logs * n), 1000))

    def policy_params(state: DreamerV3State) -> dict[str, Any]:
        nets = nnx.merge(graphdef, state.nets)
        return {"agent": nnx.state(Agent(nets.world_model, nets.actor))}

    def init_carry(batch_size: int):
        S = cfg.stoch * cfg.classes
        latent = Latent(jnp.zeros((batch_size, cfg.deter)), jnp.zeros((batch_size, S)))
        return latent, jnp.zeros((batch_size, A))

    apply = make_policy_apply(agent_graphdef, pk)
    evaluate = envs.make_evaluator(cfg, s.eval_env, apply, init_carry)
    state, _ = train_loop(
        cfg,
        s.logger,
        state,
        make_train_fn(iterations, random=False),
        iterations * n,
        evaluate,
        policy_params,
        lambda st: {"nets": st.nets, "return_scale": st.return_scale},
        jax.random.key(cfg.seed + 1),
        warmup_iterations * n,
    )
    s.logger.close()
    return Policy(NAME, policy_params(state), apply, init_carry)


def load_policy(run_dir: str | Path, step: int | None = None) -> Policy:
    """Rebuild the policy saved in ``run_dir`` (latest checkpoint unless ``step`` is given)."""
    meta = checkpoint.load_metadata(run_dir)
    cfg = DreamerV3Config.from_dict(meta["config"])
    spec = meta["spec"]
    pk, A = spec["policy_obs_key"], spec["action_size"]
    nets = Networks(cfg, spec["obs_shapes"][pk][-1], A, jax.random.key(0))
    graphdef, nets_state = nnx.split(nets)
    _, restored = checkpoint.load(run_dir, {"nets": nets_state, "return_scale": jnp.ones(())}, step)
    nets = nnx.merge(graphdef, restored["nets"])
    agent_graphdef, agent_state = nnx.split(Agent(nets.world_model, nets.actor))
    S = cfg.stoch * cfg.classes

    def init_carry(batch_size: int):
        latent = Latent(jnp.zeros((batch_size, cfg.deter)), jnp.zeros((batch_size, S)))
        return latent, jnp.zeros((batch_size, A))

    return Policy(NAME, {"agent": agent_state}, make_policy_apply(agent_graphdef, pk), init_carry)


Config = DreamerV3Config
