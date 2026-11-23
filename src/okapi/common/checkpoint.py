"""Orbax checkpoints of training state pytrees.

Checkpoints live in ``<run_dir>/checkpoints/ckpt_<step>``. Only arrays are
stored; network structure is rebuilt from the run's ``metadata.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import orbax.checkpoint as ocp
import yaml


def _manager(run_dir: str | Path) -> ocp.CheckpointManager:
    path = (Path(run_dir) / "checkpoints").resolve()  # Orbax requires absolute paths
    path.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(
        directory=str(path),
        options=ocp.CheckpointManagerOptions(step_prefix="ckpt", step_format_fixed_length=10),
    )


def save(run_dir: str | Path, step: int, state: Any) -> None:
    """Save ``state`` (any pytree of arrays) at environment step ``step``."""
    mgr = _manager(run_dir)
    mgr.save(step, args=ocp.args.StandardSave({"state": state, "step": jnp.array(step)}))
    mgr.wait_until_finished()


def steps(run_dir: str | Path) -> list[int]:
    """Saved checkpoint steps, ascending."""
    return sorted(_manager(run_dir).all_steps())


def load(run_dir: str | Path, target: Any, step: int | None = None) -> tuple[int, Any]:
    """Restore the checkpoint at ``step`` (latest if ``None``) into the structure of ``target``."""
    mgr = _manager(run_dir)
    step = mgr.latest_step() if step is None else step
    if step is None:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    out = mgr.restore(step, args=ocp.args.StandardRestore({"state": target, "step": jnp.array(0)}))
    return int(out["step"]), out["state"]


def load_metadata(run_dir: str | Path) -> dict[str, Any]:
    with open(Path(run_dir) / "metadata.yaml") as f:
        return yaml.safe_load(f)
