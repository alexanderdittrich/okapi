"""okapi: single-file deep RL algorithms in JAX / Flax NNX for MuJoCo Playground."""

from __future__ import annotations

import importlib
import os
from types import ModuleType

# Defaults for JAX / MuJoCo on GPU; values already set in the environment win.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
if "--xla_gpu_triton_gemm_any" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"

__version__ = "0.2.0"

#: Algorithm name → module. Every module defines ``Config``, ``train`` and ``load_policy``.
ALGORITHMS: dict[str, str] = {
    "ppo": "okapi.ppo",
    "ppo_gru": "okapi.ppo_gru",
    "sac": "okapi.sac",
    "simbav2": "okapi.simbav2",
    "bro": "okapi.bro",
    "dreamerv3": "okapi.dreamerv3",
    "tdmpc2": "okapi.tdmpc2",
}


def get_algorithm(name: str) -> ModuleType:
    """Import the module of algorithm ``name`` (see :data:`ALGORITHMS`)."""
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
    return importlib.import_module(ALGORITHMS[name])


def load_policy(run_dir: str, step: int | None = None):
    """Load the policy of any run directory (dispatches on its ``metadata.yaml``)."""
    from okapi.common.checkpoint import load_metadata

    return get_algorithm(load_metadata(run_dir)["algorithm"]).load_policy(run_dir, step)
