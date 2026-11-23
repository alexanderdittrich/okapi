"""Settings shared by every algorithm.

Each algorithm extends :class:`TrainConfig` with its own hyperparameters, so
environment, budget, evaluation and logging options are named the same
everywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class TrainConfig:
    """Options common to all algorithms.

    Attributes:
        env_id: MuJoCo Playground environment name (ignored with ``env_factory``).
        impl: Physics backend, ``"jax"`` (MJX) or ``"warp"``. ``None`` keeps the
            environment default.
        num_envs: Parallel environments used for data collection.
        episode_length: Steps before an episode is truncated.
        action_repeat: Physics steps per agent step.
        policy_obs_key: Observation entry fed to the policy (dict observations).
        value_obs_key: Observation entry fed to value functions / critics.
        domain_randomization: Use the registry's domain randomizer, if any.
        full_reset: On-policy auto-reset calls ``env.reset`` instead of restoring a cached
            first state; needed when the environment keeps per-episode state in ``info``.
        env_overrides: Extra ``config_overrides`` passed to ``registry.load``.
        total_timesteps: Environment steps to train for.
        seed: Random seed.
        num_evals: Evaluations spread evenly over training (plus one at step 0).
        num_eval_envs: Episodes per evaluation.
        num_logs: Training-metric log points spread evenly over training.
        log_dir: Parent directory of run directories.
        run_name: Run directory name; generated from env, algorithm and time if ``None``.
        use_wandb: Log to Weights & Biases.
        wandb_project: Weights & Biases project.
        checkpoint: Save a checkpoint at every evaluation.
        verbose: Print metrics to the console.
    """

    # Environment
    env_id: str = "CheetahRun"
    impl: str | None = None
    num_envs: int = 1024
    episode_length: int = 1000
    action_repeat: int = 1
    policy_obs_key: str = "state"
    value_obs_key: str = "state"
    domain_randomization: bool = False
    full_reset: bool = False
    env_overrides: dict[str, Any] = field(default_factory=dict)

    # Budget
    total_timesteps: int = 1_000_000
    seed: int = 0

    # Evaluation, logging, checkpointing
    num_evals: int = 20
    num_eval_envs: int = 128
    num_logs: int = 100
    log_dir: str = "runs"
    run_name: str | None = None
    use_wandb: bool = False
    wandb_project: str = "okapi"
    checkpoint: bool = False
    verbose: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]):
        """Build a config, rejecting unknown keys (typos would otherwise pass silently)."""
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
        return cls(**values)
