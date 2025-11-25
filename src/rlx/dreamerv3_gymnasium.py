"""
DreamerV3 single-file implementation using Flax NNX

- Follows the official DreamerV3 repo: https://github.com/danijar/dreamerv3
- Designed for Gymnasium vector environments (sync mode)
- Pure JAX, Flax NNX, Hydra config, and functional style
- Readable, maintainable, and research-friendly

References:
- Hafner et al. (2023) "Mastering Diverse Domains through World Models"
- https://github.com/danijar/dreamerv3
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from omegaconf import DictConfig, OmegaConf
import distrax


# Set GPU parameters
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# ---------------------------
# Config
# ---------------------------
@dataclass
class DreamerConfig:
    env_id: str = "HalfCheetah-v5"
    num_envs: int = 4
    total_steps: int = 500_000
    seed: int = 42
    # Model sizes
    rssm_hidden: int = 512
    rssm_deter: int = 512
    rssm_stoch: int = 32
    rssm_classes: int = 32
    encoder_depth: int = 32
    decoder_depth: int = 32
    # Training
    batch_size: int = 16
    batch_length: int = 64
    grad_heads: tuple[str, ...] = ("image", "reward")
    imag_horizon: int = 15
    actor_dist: str = "trunc_normal"
    actor_layers: int = 4
    actor_units: int = 512
    critic_layers: int = 4
    critic_units: int = 512
    discount: float = 0.997
    lambda_: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    model_lr: float = 6e-4
    kl_scale: float = 1.0
    kl_balance: float = 0.8
    free_nats: float = 3.0
    expl_amount: float = 0.0
    expl_min: float = 0.0
    expl_decay: float = 0.0
    expl_type: str = "additive"
    # Logging
    log_frequency: int = 10
    save_model: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    keep_checkpoints: int = 3
    use_wandb: bool = False
    wandb_project: str = "dreamerv3-gymnasium"


# ---------------------------
# Replay Buffer
# ---------------------------


class ReplayBuffer:
    def __init__(self, obs_shape, action_dim, size, batch_length, batch_size):
        self.size = size
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def add(self, obs, actions, rewards, dones):
        n = obs.shape[0] * obs.shape[1]
        obs = obs.reshape(-1, *obs.shape[2:])
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        idxs = np.arange(self.ptr, self.ptr + n) % self.size
        self.obs[idxs] = obs
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones
        self.ptr = (self.ptr + n) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self):
        max_idx = self.size if self.full else self.ptr
        idxs = np.random.randint(0, max_idx - self.batch_length, size=self.batch_size)
        obs = np.stack([self.obs[i : i + self.batch_length] for i in idxs])
        actions = np.stack([self.actions[i : i + self.batch_length] for i in idxs])
        rewards = np.stack([self.rewards[i : i + self.batch_length] for i in idxs])
        dones = np.stack([self.dones[i : i + self.batch_length] for i in idxs])
        return obs, actions, rewards, dones


# ---------------------------
# Model Components (Encoder, RSSM, Decoder, Actor, Critic)
# ---------------------------


def get_activation_fn(name: str):
    activations = {
        "relu": nnx.relu,
        "elu": nnx.elu,
        "swish": nnx.swish,
        "tanh": nnx.tanh,
        "gelu": nnx.gelu,
    }
    return activations[name]


# MLP Encoder for proprioceptive input
class MLPEncoder(nnx.Module):
    def __init__(self, input_dim, hidden_dim, layers=3, act=nnx.swish, **_):
        self.net = nnx.Sequential(
            *[nnx.Linear(hidden_dim, act=act) for _ in range(layers)]
        )
        self.input_dim = input_dim

    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = x.reshape((x.shape[0], -1))
        return self.net(x)


# MLP Decoder for proprioceptive input
class MLPDecoder(nnx.Module):
    def __init__(self, output_dim, hidden_dim, layers=3, act=nnx.swish, **_):
        self.net = nnx.Sequential(
            *[nnx.Linear(hidden_dim, act=act) for _ in range(layers)]
        )
        self.out = nnx.Linear(output_dim)

    def __call__(self, x):
        x = self.net(x)
        return self.out(x)


# Faithful RSSM with categorical latent, sampling, and KL
class RSSM(nnx.Module):
    def __init__(self, deter, stoch, classes, hidden, act=nnx.elu, **_):
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.act = act
        self.rnn = nnx.GRUCell(deter)
        self.fc1 = nnx.Linear(hidden, act=act)
        self.fc2 = nnx.Linear(hidden, act=act)
        self.fc3 = nnx.Linear(stoch * classes)

    def _cat_dist(self, logits):
        # logits: [B, stoch * classes]
        logits = logits.reshape((logits.shape[0], self.stoch, self.classes))
        return distrax.Categorical(logits=logits)

    def _sample(self, dist, key):
        # Sample one-hot from categorical
        idx = dist.sample(seed=key)
        return jax.nn.one_hot(idx, self.classes)

    def __call__(self, prev_stoch, prev_deter, action, embed, key):
        # Prior
        x = jnp.concatenate([prev_stoch.reshape(prev_stoch.shape[0], -1), action], -1)
        deter, _ = self.rnn(prev_deter, x)
        h = self.act(self.fc1(deter))
        prior_logits = self.fc3(h)
        prior_dist = self._cat_dist(prior_logits)
        # Posterior
        h2 = self.act(self.fc2(jnp.concatenate([deter, embed], -1)))
        post_logits = self.fc3(h2)
        post_dist = self._cat_dist(post_logits)
        # Sample posterior state
        post_key, prior_key = jax.random.split(key)
        post_stoch = self._sample(post_dist, post_key)
        prior_stoch = self._sample(prior_dist, prior_key)
        return {
            "deter": deter,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "prior_dist": prior_dist,
            "post_dist": post_dist,
            "prior_stoch": prior_stoch,
            "post_stoch": post_stoch,
        }


class DenseHead(nnx.Module):
    def __init__(self, shape, layers, units, act=nnx.elu, dist="trunc_normal", **_):
        self.shape = shape
        self.layers = nnx.Sequential(
            *[nnx.Linear(units, act=act) for _ in range(layers)]
        )
        self.out = nnx.Linear(np.prod(shape))
        self.dist = dist

    def __call__(self, x):
        x = self.layers(x)
        x = self.out(x)
        if self.dist == "trunc_normal":
            return distrax.Normal(x, 1.0).sample(seed=jax.random.PRNGKey(0))
        return x.reshape((-1,) + self.shape)


# Full DreamerV3 Model for proprioceptive input


class DreamerV3Model(nnx.Module):
    def __init__(self, obs_dim, action_dim, cfg: DreamerConfig, rngs: nnx.Rngs):
        self.encoder = MLPEncoder(obs_dim, cfg.encoder_depth)
        self.rssm = RSSM(
            cfg.rssm_deter, cfg.rssm_stoch, cfg.rssm_classes, cfg.rssm_hidden
        )
        self.decoder = MLPDecoder(obs_dim, cfg.decoder_depth)
        self.reward_head = DenseHead((1,), 2, 512)
        self.actor = DenseHead(
            (action_dim,), cfg.actor_layers, cfg.actor_units, dist=cfg.actor_dist
        )
        self.critic = DenseHead((1,), cfg.critic_layers, cfg.critic_units)

    def __call__(self, obs, action, prev_stoch, prev_deter, key):
        embed = self.encoder(obs)
        rssm_out = self.rssm(prev_stoch, prev_deter, action, embed, key)
        deter = rssm_out["deter"]
        recon = self.decoder(deter)
        reward = self.reward_head(deter)
        policy = self.actor(deter)
        value = self.critic(deter)
        return recon, reward, policy, value, rssm_out


# ---------------------------
# Loss Functions and Training Loop
# ---------------------------


def dreamer_loss(model, obs, actions, rewards, cfg: DreamerConfig, key):
    # This is a simplified version. The official repo uses a replay buffer and imag rollout.
    # Here, we just show the structure for a single batch.
    batch_size, batch_length = obs.shape[:2]
    # Initial state
    prev_stoch = jnp.zeros((batch_size, cfg.rssm_stoch * cfg.rssm_classes))
    prev_deter = jnp.zeros((batch_size, cfg.rssm_deter))
    loss = 0.0
    for t in range(batch_length):
        o = obs[:, t]
        a = actions[:, t]
        r = rewards[:, t]
        recon, reward_pred, policy, value, prior, post, deter = model(
            o, a, prev_stoch, prev_deter
        )
        # World model loss: reconstruction + reward + KL
        recon_loss = ((recon - o) ** 2).mean()
        reward_loss = ((reward_pred - r) ** 2).mean()
        kl_loss = jnp.sum(jax.scipy.stats.entropy(post, prior))
        # Actor-critic loss (simplified)
        actor_loss = -value.mean()
        critic_loss = ((value - r) ** 2).mean()
        loss += (
            recon_loss + reward_loss + cfg.kl_scale * kl_loss + actor_loss + critic_loss
        )
        prev_stoch, prev_deter = post, deter
    return loss / batch_length


def train(cfg: DreamerConfig):
    # Set up environment
    envs = gym.make_vec(cfg.env_id, num_envs=cfg.num_envs, vectorization_mode="sync")
    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = envs.single_action_space.shape[0]
    key = jax.random.key(cfg.seed)
    # Model
    model = DreamerV3Model(obs_dim, action_dim, cfg, nnx.Rngs(key))
    # Separate optimizers for world model, actor, critic
    model_opt = nnx.Optimizer(model, optax.adam(cfg.model_lr), wrt=nnx.Param)
    actor_opt = nnx.Optimizer(model.actor, optax.adam(cfg.actor_lr), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(model.critic, optax.adam(cfg.critic_lr), wrt=nnx.Param)
    # Replay buffer
    replay_buffer = ReplayBuffer(
        (obs_dim,),
        action_dim,
        size=100000,
        batch_length=cfg.batch_length,
        batch_size=cfg.batch_size,
    )
    obs, _ = envs.reset(seed=cfg.seed)
    for step in range(cfg.total_steps // cfg.batch_size):
        # Collect experience
        batch_obs = np.zeros(
            (cfg.batch_size, cfg.batch_length, obs_dim), dtype=np.float32
        )
        batch_actions = np.zeros(
            (cfg.batch_size, cfg.batch_length, action_dim), dtype=np.float32
        )
        batch_rewards = np.zeros(
            (cfg.batch_size, cfg.batch_length, 1), dtype=np.float32
        )
        batch_dones = np.zeros((cfg.batch_size, cfg.batch_length, 1), dtype=np.float32)
        for b in range(cfg.batch_size):
            o = obs
            prev_stoch = jnp.zeros((cfg.num_envs, cfg.rssm_stoch, cfg.rssm_classes))
            prev_deter = jnp.zeros((cfg.num_envs, cfg.rssm_deter))
            for t in range(cfg.batch_length):
                rollout_key, key = jax.random.split(key)
                _, _, policy, _, _ = model(
                    o,
                    np.zeros((cfg.num_envs, action_dim)),
                    prev_stoch.reshape(cfg.num_envs, -1),
                    prev_deter,
                    rollout_key,
                )
                a = np.tanh(np.asarray(policy))
                next_obs, r, terminated, truncated, _ = envs.step(a)
                done = np.logical_or(terminated, truncated).astype(np.float32)
                batch_obs[b, t] = o.reshape(-1)
                batch_actions[b, t] = a
                batch_rewards[b, t] = r
                batch_dones[b, t] = done
                o = next_obs
            obs = o
        replay_buffer.add(batch_obs, batch_actions, batch_rewards, batch_dones)
        # Sample batch
        obs_batch, act_batch, rew_batch, done_batch = replay_buffer.sample()

        # World model update (reconstruction, reward, KL)
        def categorical_kl(p_logits, q_logits):
            # p, q: [B, stoch, classes]
            p = jax.nn.log_softmax(p_logits, axis=-1)
            q = jax.nn.log_softmax(q_logits, axis=-1)
            p_prob = jnp.exp(p)
            kl = jnp.sum(p_prob * (p - q), axis=-1)  # [B, stoch]
            return kl.sum(-1).mean()  # mean over batch

        def world_model_loss(model):
            batch_size, batch_length, _ = obs_batch.shape
            prev_stoch = jnp.zeros((batch_size, cfg.rssm_stoch, cfg.rssm_classes))
            prev_deter = jnp.zeros((batch_size, cfg.rssm_deter))
            kl_loss, recon_loss, reward_loss = 0.0, 0.0, 0.0
            for t in range(batch_length):
                o = jnp.array(obs_batch[:, t])
                a = jnp.array(act_batch[:, t])
                r = jnp.array(rew_batch[:, t])
                rollout_key, _ = jax.random.split(key)
                recon, reward_pred, _, _, rssm_out = model(
                    o, a, prev_stoch.reshape(batch_size, -1), prev_deter, rollout_key
                )
                # KL balancing: alpha*KL(post||prior) + (1-alpha)*KL(prior||post)
                post_logits = rssm_out["post_logits"].reshape(
                    batch_size, cfg.rssm_stoch, cfg.rssm_classes
                )
                prior_logits = rssm_out["prior_logits"].reshape(
                    batch_size, cfg.rssm_stoch, cfg.rssm_classes
                )
                kl_post_prior = categorical_kl(post_logits, prior_logits)
                kl_prior_post = categorical_kl(prior_logits, post_logits)
                kl = (
                    cfg.kl_balance * kl_post_prior
                    + (1.0 - cfg.kl_balance) * kl_prior_post
                )
                kl = jnp.maximum(kl, cfg.free_nats)
                kl_loss += kl
                recon_loss += ((recon - o) ** 2).mean()
                reward_loss += ((reward_pred - r) ** 2).mean()
                prev_stoch = rssm_out["post_stoch"]
                prev_deter = rssm_out["deter"]
            return (kl_loss + recon_loss + reward_loss) / batch_length

        wm_loss, wm_grads = nnx.value_and_grad(world_model_loss)(model)
        model_opt.update(model=model, grads=wm_grads)

        # Imagination rollout for actor/critic
        def imagine_rollout(model, horizon, key):
            batch_size = obs_batch.shape[0]
            prev_stoch = jnp.zeros((batch_size, cfg.rssm_stoch, cfg.rssm_classes))
            prev_deter = jnp.zeros((batch_size, cfg.rssm_deter))
            features = []
            actions = []
            entropies = []
            for t in range(horizon):
                rollout_key, key = jax.random.split(key)
                deter = prev_deter
                feat = jnp.concatenate([prev_stoch.reshape(batch_size, -1), deter], -1)
                policy = model.actor(deter)
                # Add exploration noise (DreamerV3: additive Gaussian)
                noise = cfg.expl_amount * jax.random.normal(rollout_key, policy.shape)
                policy_expl = policy + noise
                a = jnp.tanh(policy_expl)
                actions.append(a)
                # Entropy (for regularization)
                entropy = -jnp.mean(jax.nn.log_softmax(policy_expl, axis=-1), axis=-1)
                entropies.append(entropy)
                # Rollout prior only (no obs)
                rssm_out = model.rssm(
                    prev_stoch,
                    prev_deter,
                    a,
                    jnp.zeros((batch_size, model.encoder.input_dim)),
                    rollout_key,
                )
                prev_stoch = rssm_out["prior_stoch"]
                prev_deter = rssm_out["deter"]
                features.append(feat)
            return (
                jnp.stack(features, axis=1),
                jnp.stack(actions, axis=1),
                jnp.stack(entropies, axis=1),
            )

        imag_feats, imag_actions, imag_entropies = imagine_rollout(
            model, cfg.imag_horizon, key
        )

        # Lambda-returns and actor/critic loss
        def lambda_returns(rewards, values, discount, lambda_):
            # rewards, values: [B, H]
            def scan_fn(carry, t):
                reward, value = rewards[:, t], values[:, t]
                next_return = carry
                ret = reward + discount * (
                    (1 - lambda_) * value + lambda_ * next_return
                )
                return ret, ret

            init = values[:, -1]
            _, returns = jax.lax.scan(
                scan_fn, init, jnp.arange(rewards.shape[1] - 1, -1, -1)
            )
            returns = returns[::-1].transpose(1, 0)  # [B, H]
            return returns

        # Compute imagined rewards and values
        imag_deters = imag_feats[..., imag_feats.shape[-1] // 2 :]
        imag_rewards = model.reward_head(imag_deters).squeeze(-1)  # [B, H]
        imag_values = model.critic(imag_deters).squeeze(-1)  # [B, H]
        lambda_ret = lambda_returns(
            imag_rewards, imag_values, cfg.discount, cfg.lambda_
        )

        def actor_loss_fn(actor):
            # Maximize expected value under imagined policy + entropy regularization
            ent = imag_entropies.mean()
            return (
                -jnp.mean(lambda_ret) - 1e-3 * ent
            )  # 1e-3: typical DreamerV3 entropy scale

        actor_loss, actor_grads = nnx.value_and_grad(actor_loss_fn)(model.actor)
        actor_opt.update(model.actor, actor_grads)

        def critic_loss_fn(critic):
            # Fit critic to lambda-returns
            return jnp.mean((imag_values - lambda_ret) ** 2)

        critic_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(model.critic)
        critic_opt.update(model.critic, critic_grads)
        if step % cfg.log_frequency == 0:
            mean_reward = float(np.mean(batch_rewards))
            mean_entropy = float(np.mean(imag_entropies))
            print(
                f"Step {step}: wm_loss={float(wm_loss):.4f}, actor_loss={float(actor_loss):.4f}, critic_loss={float(critic_loss):.4f}, mean_reward={mean_reward:.2f}, entropy={mean_entropy:.4f}"
            )


# ---------------------------
# Hydra entry point
# ---------------------------
@hydra.main(
    version_base=None, config_path="../../configs", config_name="dreamerv3_gymnasium"
)
def main(cfg: DictConfig):
    train(DreamerConfig(**OmegaConf.to_container(cfg, resolve=True)))


if __name__ == "__main__":
    main()
