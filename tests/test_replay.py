"""n-step targets and sequence sampling of the replay buffer."""

import jax
import jax.numpy as jnp
import numpy as np

from okapi.common import replay


def make_buffer(rewards, terminal, last, capacity=None):
    """One environment; obs[t] = t so sampled indices can be read back."""
    T = len(rewards)
    row = {"obs": {"state": jnp.zeros((1, 1))}, "action": jnp.zeros((1, 1)),
           "reward": jnp.zeros(1), "terminal": jnp.zeros(1), "last": jnp.zeros(1),
           "first": jnp.zeros(1)}
    buf = replay.init(row, capacity or T)
    for t in range(T):
        buf = replay.add(buf, {
            "obs": {"state": jnp.full((1, 1), float(t))}, "action": jnp.full((1, 1), float(t)),
            "reward": jnp.array([rewards[t]], jnp.float32),
            "terminal": jnp.array([terminal[t]], jnp.float32),
            "last": jnp.array([last[t]], jnp.float32), "first": jnp.zeros(1)})
    return buf


def test_nstep_stops_at_termination():
    # rows: 0 1 2 3(terminal) 4(first) 5 6 7 8 9
    rewards = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    terminal = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    buf = make_buffer(rewards, terminal, terminal)
    batch = replay.sample_transitions(buf, jax.random.key(0), 512, n_step=3, gamma=0.5)
    start = np.asarray(batch["obs"]["state"][:, 0]).astype(int)
    nxt = np.asarray(batch["next_obs"]["state"][:, 0]).astype(int)
    assert 3 not in start  # a last row is never a start
    for t, n, r, d in zip(start, nxt, np.asarray(batch["reward"]), np.asarray(batch["discount"])):
        if t <= 2:  # episode ends at 3: bootstrap obs 3 with discount 0
            assert n == 3 and d == 0.0
            assert np.isclose(r, sum(0.5**k for k in range(3 - t)))
        else:
            assert n == t + 3 and np.isclose(d, 0.125) and np.isclose(r, 1.75)


def test_nstep_bootstraps_after_truncation():
    rewards = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    last = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # truncated, not terminated
    buf = make_buffer(rewards, [0] * 10, last)
    batch = replay.sample_transitions(buf, jax.random.key(1), 512, n_step=3, gamma=0.5)
    start = np.asarray(batch["obs"]["state"][:, 0]).astype(int)
    d = np.asarray(batch["discount"])
    for t, disc in zip(start, d):
        if t <= 2:
            assert np.isclose(disc, 0.5 ** (3 - t))  # bootstraps from the final obs


def test_sequences_are_contiguous_after_wraparound():
    buf = make_buffer([0.0] * 25, [0] * 25, [0] * 25, capacity=8)
    seq = replay.sample_sequences(buf, jax.random.key(0), 64, seq_len=5)
    obs = np.asarray(seq["obs"]["state"][..., 0])  # (5, 64)
    assert obs.min() >= 25 - 8
    assert np.all(np.diff(obs, axis=0) == 1)
