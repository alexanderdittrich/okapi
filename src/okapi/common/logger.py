"""Run directories, metadata and metric logging (console, JSONL, W&B, callback)."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import yaml

from okapi.common.config import TrainConfig

LogCallback = Callable[[dict[str, float], int], None]

_PACKAGES = (
    "jax",
    "flax",
    "optax",
    "orbax-checkpoint",
    "mujoco",
    "mujoco-mjx",
    "playground",
    "warp-lang",
    "brax",
    "numpy",
)


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _schedule(total: int, num_points: int, include_zero: bool) -> list[int]:
    points = [round(total * k / max(num_points, 1)) for k in range(1, num_points + 1)]
    return ([0] if include_zero else []) + points


class Logger:
    """Creates the run directory and dispatches metrics.

    Every run directory contains ``metadata.yaml`` (algorithm, configs, package
    versions, git commit) and ``metrics.jsonl`` (one JSON object per log call).

    Args:
        cfg: Algorithm config.
        algorithm: Algorithm name, stored in the metadata.
        env_config: Environment config dict, stored in the metadata.
        spec: Observation shapes and action size, stored in the metadata so that
            policies can be rebuilt without instantiating the environment.
        log_callback: Optional ``callback(metrics, step)`` called on every log.
        start_step: Skip evaluation / log points before this step (resuming).
    """

    def __init__(
        self,
        cfg: TrainConfig,
        algorithm: str,
        env_config: dict[str, Any] | None = None,
        spec: dict[str, Any] | None = None,
        log_callback: LogCallback | None = None,
        start_step: int = 0,
    ):
        self.cfg = cfg
        self.algorithm = algorithm
        self.callback = log_callback
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = cfg.run_name or f"{cfg.env_id.lower()}-{algorithm}-{stamp}-{uuid.uuid4().hex[:6]}"
        self.run_dir = (Path(cfg.log_dir) / name).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "algorithm": algorithm,
            "run_name": name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "packages": {p: _version(p) for p in _PACKAGES},
            "config": cfg.to_dict(),
            "spec": spec or {},
            "environment": env_config or {},
        }
        with open(self.run_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)
        self._jsonl = open(self.run_dir / "metrics.jsonl", "a")

        self._evals = _schedule(cfg.total_timesteps, cfg.num_evals, include_zero=True)
        self._logs = _schedule(cfg.total_timesteps, cfg.num_logs, include_zero=False)
        self._evals = [p for p in self._evals if p >= start_step]
        self._logs = [p for p in self._logs if p > start_step]
        self.start_time = self._last_time = time.time()
        self._last_step = start_step

        self._wandb = None
        if cfg.use_wandb:
            import wandb

            self._wandb = wandb.init(
                project=cfg.wandb_project,
                name=name,
                config={"algorithm": algorithm, **cfg.to_dict(), "environment": env_config},
            )
        if cfg.verbose:
            print(f"[okapi] {algorithm} on {cfg.env_id}  →  {self.run_dir}")

    @staticmethod
    def _due(points: list[int], step: int) -> bool:
        hit = False
        while points and step >= points[0]:
            points.pop(0)
            hit = True
        return hit

    def should_eval(self, step: int) -> bool:
        """True once per evaluation point that ``step`` has passed."""
        return self._due(self._evals, step)

    def should_log(self, step: int) -> bool:
        """True once per log point that ``step`` has passed."""
        return self._due(self._logs, step)

    def log(self, metrics: dict[str, Any], step: int) -> dict[str, float]:
        """Record metrics at ``step`` (environment steps). Adds ``time/*`` entries."""
        now = time.time()
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        if step > self._last_step:
            metrics["time/sps"] = (step - self._last_step) / max(now - self._last_time, 1e-9)
        metrics["time/walltime"] = now - self.start_time
        self._last_step, self._last_time = step, now

        self._jsonl.write(json.dumps({"step": int(step), **metrics}) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=int(step))
        if self.callback is not None:
            self.callback(metrics, int(step))
        if self.cfg.verbose:
            self._print(metrics, step)
        return metrics

    def _print(self, metrics: dict[str, float], step: int) -> None:
        total = self.cfg.total_timesteps
        head = f"[{step / 1e6:8.2f}M / {total / 1e6:.2f}M]"
        if "eval/episode_reward" in metrics:
            head += (
                f"  eval {metrics['eval/episode_reward']:9.2f}"
                f" ± {metrics['eval/episode_reward_std']:.1f}"
            )
        if "time/sps" in metrics:
            head += f"  sps {metrics['time/sps']:9.0f}"
        rest = [
            f"{k.replace('train/', '').replace('/', '_')}={v:.3g}"
            for k, v in sorted(metrics.items())
            if k.startswith(("loss/", "train/"))
        ]
        print(head + ("  " + " ".join(rest) if rest else ""), flush=True)

    def close(self) -> None:
        self._jsonl.close()
        if self.cfg.verbose:
            print(f"[okapi] done in {time.time() - self.start_time:.0f}s  →  {self.run_dir}")
        if self._wandb is not None:
            self._wandb.finish()


def env_config_dict(raw_env: Any) -> dict[str, Any]:
    """The environment's ConfigDict as plain YAML-safe data (empty if unavailable)."""
    try:
        return yaml.safe_load(str(raw_env._config)) or {}
    except Exception:
        return {}
