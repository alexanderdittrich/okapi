"""The inference interface returned by every algorithm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp


def _no_carry(batch_size: int) -> tuple:
    return ()


@dataclass(frozen=True)
class Policy:
    """A trained policy acting on raw environment observations.

    Feed-forward and recurrent policies share one signature::

        carry = policy.init_carry(num_envs)
        action, carry = policy(obs, carry, first)

    ``first`` marks environments whose episode starts at ``obs`` (it resets the
    memory of recurrent policies and is ignored otherwise).

    Attributes:
        name: Algorithm name.
        params: Network parameters and observation statistics.
        apply: ``apply(params, carry, obs, first, key, deterministic) -> (action, carry)``.
        init_carry: ``init_carry(batch_size) -> carry`` (``()`` if feed-forward).
    """

    name: str
    params: Any
    apply: Callable[..., tuple[jax.Array, Any]]
    init_carry: Callable[[int], Any] = field(default=_no_carry)

    def __call__(
        self,
        obs: Any,
        carry: Any = (),
        first: jax.Array | None = None,
        key: jax.Array | None = None,
        deterministic: bool = True,
    ) -> tuple[jax.Array, Any]:
        if key is None:
            key = jax.random.key(0)
        if first is None:
            first = jnp.zeros(())
        return self.apply(self.params, carry, obs, first, key, deterministic)
