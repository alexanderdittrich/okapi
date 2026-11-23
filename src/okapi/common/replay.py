"""GPU replay buffer storing each environment's steps in order.

Rows follow the convention of DreamerV3: row ``t`` of an environment holds the
observation ``obs[t]``, the action taken there, the ``reward`` received on
arriving at ``obs[t]``, and whether ``obs[t]`` is ``terminal`` (episode
terminated), ``last`` (episode ended, terminated or truncated) or ``first``
(episode started). Collect with ``make_env(..., autoreset=False)`` and
:func:`~okapi.common.envs.step_or_reset` so that terminal observations are stored.

Off-policy actor-critics sample n-step transitions (:func:`sample_transitions`),
world models sample sequences (:func:`sample_sequences`).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

PyTree = Any


class ReplayBuffer(NamedTuple):
    data: PyTree  # leaves of shape (capacity, num_envs, ...)
    count: jax.Array  # rows written per environment


def init(example_row: PyTree, capacity: int) -> ReplayBuffer:
    """Allocate a buffer for rows shaped like ``example_row`` (leaves ``(num_envs, ...)``)."""
    data = jax.tree.map(lambda x: jnp.zeros((capacity, *x.shape), x.dtype), example_row)
    return ReplayBuffer(data, jnp.zeros((), jnp.int32))


def add(buffer: ReplayBuffer, row: PyTree) -> ReplayBuffer:
    """Write one row (one step of every environment), overwriting the oldest when full."""
    capacity = jax.tree.leaves(buffer.data)[0].shape[0]
    i = buffer.count % capacity
    data = jax.tree.map(lambda d, x: d.at[i].set(x), buffer.data, row)
    return ReplayBuffer(data, buffer.count + 1)


def _shape(buffer: ReplayBuffer) -> tuple[int, int]:
    return jax.tree.leaves(buffer.data)[0].shape[:2]


def sample_transitions(
    buffer: ReplayBuffer, key: jax.Array, batch_size: int, n_step: int, gamma: float
) -> dict[str, PyTree]:
    """Sample n-step transitions ``(obs, action, reward, discount, next_obs)``.

    ``reward`` is the discounted sum of up to ``n_step`` rewards, truncated where
    the episode ends; ``next_obs`` is the observation to bootstrap from and
    ``discount`` is ``gamma ** k`` (``k`` steps taken), or 0 after a termination.
    Requires at least ``n_step + 2`` rows.
    """
    capacity, num_envs = _shape(buffer)
    d = buffer.data
    filled = jnp.minimum(buffer.count, capacity)
    time_key, env_key = jax.random.split(key)
    # start rows t in [oldest, count - n_step - 2]; a last row is shifted to the next (first) row
    t = buffer.count - filled + jax.random.randint(time_key, (batch_size,), 0, filled - n_step - 1)
    env = jax.random.randint(env_key, (batch_size,), 0, num_envs)
    at = lambda x, k: x[(t + k) % capacity, env]
    t = jnp.where(at(d["last"], 0) > 0, t + 1, t)

    ret = jnp.zeros(batch_size)
    discount = jnp.ones(batch_size)
    ended = jnp.zeros(batch_size, bool)
    next_t = t + n_step
    for k in range(1, n_step + 1):
        ret = ret + jnp.where(ended, 0.0, discount * at(d["reward"], k))
        discount = jnp.where(ended, discount, discount * gamma)
        newly = ~ended & (at(d["last"], k) > 0)
        next_t = jnp.where(newly, t + k, next_t)
        ended = ended | newly
    next_at = lambda x: x[next_t % capacity, env]
    return {
        "obs": jax.tree.map(lambda x: at(x, 0), d["obs"]),
        "action": at(d["action"], 0),
        "reward": ret,
        "discount": discount * (1.0 - next_at(d["terminal"])),
        "next_obs": jax.tree.map(next_at, d["obs"]),
    }


def sample_sequences(
    buffer: ReplayBuffer, key: jax.Array, batch_size: int, seq_len: int
) -> PyTree:
    """Sample ``batch_size`` windows of ``seq_len`` consecutive rows; leaves ``(seq_len, batch, ...)``."""
    capacity, num_envs = _shape(buffer)
    filled = jnp.minimum(buffer.count, capacity)
    time_key, env_key = jax.random.split(key)
    t0 = buffer.count - filled + jax.random.randint(
        time_key, (batch_size,), 0, filled - seq_len + 1
    )
    env = jax.random.randint(env_key, (batch_size,), 0, num_envs)
    rows = (t0[None, :] + jnp.arange(seq_len)[:, None]) % capacity
    return jax.tree.map(lambda x: x[rows, env[None, :]], buffer.data)
