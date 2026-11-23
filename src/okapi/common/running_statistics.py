"""Running mean / standard deviation for observation normalization (Welford)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RunningStatisticsState(NamedTuple):
    """Running statistics of a stream of arrays."""

    count: jax.Array  # samples seen
    mean: jax.Array
    summed_variance: jax.Array  # sum of squared deviations
    std: jax.Array


def init_state(shape: tuple[int, ...]) -> RunningStatisticsState:
    """Statistics of an empty stream (mean 0, std 1)."""
    return RunningStatisticsState(
        count=jnp.zeros((), jnp.float32),
        mean=jnp.zeros(shape, jnp.float32),
        summed_variance=jnp.zeros(shape, jnp.float32),
        std=jnp.ones(shape, jnp.float32),
    )


def update(
    state: RunningStatisticsState,
    batch: jax.Array,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
) -> RunningStatisticsState:
    """Merge a batch into the statistics. All leading axes are treated as batch axes."""
    batch = batch.reshape(-1, *state.mean.shape)
    count = state.count + batch.shape[0]
    diff_old = batch - state.mean
    mean = state.mean + diff_old.sum(0) / count
    summed_variance = state.summed_variance + (diff_old * (batch - mean)).sum(0)
    summed_variance = jnp.maximum(summed_variance, 0.0)
    std = jnp.clip(jnp.sqrt(summed_variance / count), std_min_value, std_max_value)
    return RunningStatisticsState(count, mean, summed_variance, std)


def normalize(
    batch: jax.Array,
    state: RunningStatisticsState,
    max_abs_value: float | None = None,
) -> jax.Array:
    """Standardize ``batch``; optionally clip to ``[-max_abs_value, max_abs_value]``."""
    out = (batch - state.mean) / state.std
    if max_abs_value is not None:
        out = jnp.clip(out, -max_abs_value, max_abs_value)
    return out
