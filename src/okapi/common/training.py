"""Setup and outer training loop shared by all algorithms (evaluation, logging, checkpointing)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax

from okapi.common import checkpoint, envs
from okapi.common.config import TrainConfig
from okapi.common.logger import LogCallback, Logger, env_config_dict

TrainState = Any


def train_loop(
    cfg: TrainConfig,
    logger: Logger,
    state: TrainState,
    train_fn: Callable[[TrainState], tuple[TrainState, dict[str, jax.Array]]],
    steps_per_call: int,
    evaluate: Callable[[Any, jax.Array], dict[str, jax.Array]],
    policy_params: Callable[[TrainState], Any],
    checkpoint_state: Callable[[TrainState], Any],
    key: jax.Array,
    start_step: int = 0,
) -> tuple[TrainState, int]:
    """Call ``train_fn`` until ``cfg.total_timesteps`` environment steps are reached.

    Args:
        state: Algorithm-specific training state, threaded through ``train_fn``.
        train_fn: ``train_fn(state) -> (state, metrics)``; advances ``steps_per_call`` env steps.
        evaluate: ``evaluate(policy_params, key) -> metrics`` (see ``make_evaluator``).
        policy_params: Extracts the policy parameters from ``state``.
        checkpoint_state: Extracts the pytree to checkpoint from ``state``.
        start_step: Environment steps already taken (warm-up or resume).

    Returns:
        The final state and the number of environment steps taken.
    """
    step = start_step
    train_metrics: dict[str, jax.Array] = {}
    while True:
        metrics: dict[str, jax.Array] = {}
        if logger.should_eval(step):
            key, eval_key = jax.random.split(key)
            metrics.update(evaluate(policy_params(state), eval_key))
            if cfg.checkpoint:
                checkpoint.save(logger.run_dir, step, checkpoint_state(state))
        if logger.should_log(step):
            metrics.update(train_metrics)
        if metrics:
            logger.log(metrics, step)
        if step >= cfg.total_timesteps:
            break
        state, train_metrics = train_fn(state)
        step += steps_per_call
    return state, step


class Setup(NamedTuple):
    """Everything an algorithm needs before its first update (see :func:`setup`)."""

    env: Any  # vectorized training environment
    eval_env: Any  # vectorized evaluation environment (autoreset)
    env_state: Any  # initial batched training state
    policy_key: str  # observation entry of the policy
    value_key: str  # observation entry of the critic
    obs_shapes: dict[str, tuple[int, ...]]  # shapes of the used observation entries
    action_size: int
    logger: Logger
    key: jax.Array


def setup(
    cfg: TrainConfig,
    algorithm: str,
    env_factory: envs.EnvFactory | None = None,
    log_callback: LogCallback | None = None,
    autoreset: bool = True,
    start_step: int = 0,
) -> Setup:
    """Create environments, reset them, resolve observation keys and start the logger.

    Args:
        autoreset: ``True`` for on-policy algorithms; ``False`` for replay-based ones,
            which step with :func:`~okapi.common.envs.step_or_reset`.
    """
    key = jax.random.key(cfg.seed)
    key, dr_key, reset_key = jax.random.split(key, 3)
    raw_env = envs.load_env(cfg, env_factory)
    env = envs.make_env(cfg, raw_env, cfg.num_envs, dr_key, autoreset=autoreset, randomize=True)
    eval_env = envs.make_env(cfg, envs.load_env(cfg, env_factory), cfg.num_eval_envs, dr_key)
    env_state = jax.jit(lambda k: envs.reset(env, cfg.num_envs, k))(reset_key)
    policy_key, value_key = envs.obs_keys(cfg, env_state.obs)
    obs = envs.select_obs(env_state.obs, (policy_key, value_key))
    obs_shapes = {k: tuple(v.shape[1:]) for k, v in obs.items()}
    spec = {
        "policy_obs_key": policy_key,
        "value_obs_key": value_key,
        "obs_shapes": {k: list(s) for k, s in obs_shapes.items()},
        "action_size": int(env.action_size),
    }
    logger = Logger(cfg, algorithm, env_config_dict(raw_env), spec, log_callback, start_step)
    return Setup(env, eval_env, env_state, policy_key, value_key, obs_shapes,
                 int(env.action_size), logger, key)
