"""Environment creation, observation handling and evaluation.

Observations are handled as dictionaries everywhere: flat array observations
are wrapped as ``{"state": obs}``. Policies receive the raw environment
observation and normalize it themselves, so a trained :class:`~okapi.common.policy.Policy`
can be applied directly to ``env.step`` outputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training
from mujoco_playground import registry, wrapper
from mujoco_playground._src import mjx_env

from okapi.common import running_statistics
from okapi.common.config import TrainConfig

try:  # silence MuJoCo Warp's kernel-loading messages
    import warp

    warp.config.quiet = True
except ImportError:
    pass

Obs = dict[str, jax.Array]
ObsStats = dict[str, running_statistics.RunningStatisticsState]
PolicyApply = Callable[..., tuple[jax.Array, Any]]
EnvFactory = Callable[[], mjx_env.MjxEnv]


# ── Observations ─────────────────────────────────────────────────────────────


def obs_to_dict(obs: jax.Array | Mapping[str, jax.Array]) -> Obs:
    """Wrap flat observations as ``{"state": obs}``; dict observations pass through."""
    return dict(obs) if isinstance(obs, Mapping) else {"state": obs}


def obs_keys(cfg: TrainConfig, obs: Any) -> tuple[str, str]:
    """``(policy_key, value_key)``; both ``"state"`` for flat observations."""
    if not isinstance(obs, Mapping):
        return "state", "state"
    for k in (cfg.policy_obs_key, cfg.value_obs_key):
        if k not in obs:
            raise KeyError(f"Observation key {k!r} not in {list(obs)}")
    return cfg.policy_obs_key, cfg.value_obs_key


def select_obs(obs: Any, keys: tuple[str, ...]) -> Obs:
    """The observation entries named in ``keys`` (deduplicated)."""
    obs = obs_to_dict(obs)
    return {k: obs[k] for k in dict.fromkeys(keys)}


def init_obs_stats(obs: Obs) -> ObsStats:
    """Empty running statistics for each entry of an (unbatched-shape) observation."""
    return {k: running_statistics.init_state(v.shape[-1:]) for k, v in obs.items()}


def update_obs_stats(stats: ObsStats, obs: Obs) -> ObsStats:
    return {k: running_statistics.update(s, obs[k]) for k, s in stats.items()}


def normalize_obs(obs: Obs, stats: ObsStats) -> Obs:
    return {k: running_statistics.normalize(obs[k], s) for k, s in stats.items()}


# ── Environments ─────────────────────────────────────────────────────────────


def load_env(cfg: TrainConfig, env_factory: EnvFactory | None = None) -> mjx_env.MjxEnv:
    """Instantiate the raw environment from ``env_factory`` or the Playground registry."""
    if env_factory is not None:
        return env_factory()
    overrides = dict(cfg.env_overrides)
    if cfg.impl is not None:
        overrides["impl"] = cfg.impl
    return registry.load(cfg.env_id, config_overrides=overrides or None)


def make_env(
    cfg: TrainConfig,
    raw_env: mjx_env.MjxEnv,
    num_envs: int,
    key: jax.Array,
    autoreset: bool = True,
    randomize: bool = False,
) -> mjx_env.MjxEnv:
    """Vectorize ``raw_env`` with episode truncation.

    Args:
        autoreset: Reset finished episodes inside ``env.step`` (on-policy algorithms).
            Without it, use :func:`step_or_reset`, which keeps the final observation
            of an episode for one step (needed for correct off-policy bootstrapping).
        randomize: Apply the registry's domain randomizer (if one exists).
    """
    randomization_fn = None
    if randomize and cfg.domain_randomization:
        dr_fn = registry.get_domain_randomizer(cfg.env_id)
        if dr_fn is not None:
            dr_keys = jax.random.split(key, num_envs)
            randomization_fn = lambda model: dr_fn(model, dr_keys)
    if autoreset:
        return wrapper.wrap_for_brax_training(
            raw_env,
            episode_length=cfg.episode_length,
            action_repeat=cfg.action_repeat,
            randomization_fn=randomization_fn,
            full_reset=cfg.full_reset,
        )
    if randomization_fn is None:
        env = brax_training.VmapWrapper(raw_env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(raw_env, randomization_fn)
    return brax_training.EpisodeWrapper(env, cfg.episode_length, cfg.action_repeat)


def reset(env: mjx_env.MjxEnv, num_envs: int, key: jax.Array) -> mjx_env.State:
    """Reset ``num_envs`` environments (Brax wrappers expect raw ``uint32`` keys)."""
    keys = jax.random.split(key, num_envs)
    if jnp.issubdtype(keys.dtype, jax.dtypes.prng_key):
        keys = jax.random.key_data(keys)
    return env.reset(keys)


def step_or_reset(
    env: mjx_env.MjxEnv, state: mjx_env.State, action: jax.Array, key: jax.Array
) -> mjx_env.State:
    """Step all environments; those whose previous step ended an episode are reset instead.

    Used with ``make_env(..., autoreset=False)``: the terminal observation stays
    visible for one step, so replay buffers can store it with its reward and
    termination flag before the next episode starts.
    """
    prev_done = state.done > 0
    stepped = env.step(state, action)

    def with_resets() -> mjx_env.State:
        fresh = reset(env, prev_done.shape[0], key)

        def where(new, old):
            if old.ndim == 0 or old.shape[0] != prev_done.shape[0]:
                return old  # not per-environment (e.g. Warp's shared contact buffers)
            mask = prev_done.reshape(prev_done.shape + (1,) * (old.ndim - 1))
            return jnp.where(mask, new, old)

        return jax.tree.map(where, fresh, stepped)

    return jax.lax.cond(prev_done.any(), with_resets, lambda: stepped)


def termination(state: mjx_env.State) -> jax.Array:
    """1 where the episode terminated (not merely truncated by the time limit)."""
    truncation = state.info.get("truncation", jnp.zeros_like(state.done))
    return state.done * (1.0 - truncation)


# ── Evaluation ───────────────────────────────────────────────────────────────


def make_evaluator(
    cfg: TrainConfig,
    eval_env: mjx_env.MjxEnv,
    policy_apply: PolicyApply,
    init_carry: Callable[[int], Any],
) -> Callable[[Any, jax.Array], dict[str, jax.Array]]:
    """Build a jitted ``evaluate(params, key) -> metrics`` running one episode per env.

    ``policy_apply(params, carry, obs, first, key, deterministic)`` must return
    ``(action, carry)``. Episode returns and the per-episode sums of the
    environment's metrics are reported as ``eval/*``.
    """
    n = cfg.num_eval_envs
    num_steps = math.ceil(cfg.episode_length / cfg.action_repeat)

    @jax.jit
    def evaluate(params: Any, key: jax.Array) -> dict[str, jax.Array]:
        key, reset_key = jax.random.split(key)
        state = reset(eval_env, n, reset_key)
        metric_keys = sorted(state.metrics)
        zeros = jnp.zeros(n)

        def step(carry, t):
            state, policy_carry, ret, length, sums, active, key = carry
            key, act_key = jax.random.split(key)
            first = jnp.full(n, t == 0, jnp.float32)
            action, policy_carry = policy_apply(
                params, policy_carry, state.obs, first, act_key, True
            )
            state = eval_env.step(state, action)
            ret = ret + state.reward * active
            length = length + active
            sums = {k: sums[k] + state.metrics[k] * active for k in metric_keys}
            active = active * (1.0 - state.done)
            return (state, policy_carry, ret, length, sums, active, key), None

        init = (
            state,
            init_carry(n),
            zeros,
            zeros,
            {k: zeros for k in metric_keys},
            jnp.ones(n),
            key,
        )
        (_, _, ret, length, sums, _, _), _ = jax.lax.scan(step, init, jnp.arange(num_steps))
        return {
            "eval/episode_reward": ret.mean(),
            "eval/episode_reward_std": ret.std(),
            "eval/episode_length": length.mean(),
            **{f"eval/{k}": v.mean() for k, v in sums.items()},
        }

    return evaluate
