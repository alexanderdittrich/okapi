#!/usr/bin/env python3
"""Measure training throughput (environment steps per second) of each algorithm in isolation.

    uv run benchmarks/throughput.py

Runs every benchmark setting from ``run.py`` for a short budget, one job at a time, and
reports the median steps/s after compilation. Writes ``docs/benchmark_throughput.json``.
Run it on an otherwise idle GPU.
"""

from __future__ import annotations

import json
import shutil
import tempfile

import numpy as np

import okapi
from run import ROOT, SETTINGS

OUTPUT = ROOT / "docs" / "benchmark_throughput.json"
BUDGET = {"ppo": 30_000_000, "ppo_gru": 15_000_000, "sac": 2_000_000, "simbav2": 1_000_000,
          "bro": 1_000_000, "dreamerv3": 150_000, "tdmpc2": 100_000}


def measure(env_id: str, algo: str) -> float:
    module = okapi.get_algorithm(algo)
    log_dir = tempfile.mkdtemp()
    sps: list[float] = []
    cfg = module.Config.from_dict({
        **SETTINGS[env_id][algo], "env_id": env_id, "total_timesteps": BUDGET[algo],
        "num_evals": 1, "num_logs": 10, "log_dir": log_dir, "verbose": False})
    module.train(cfg, log_callback=lambda m, step: sps.append(m.get("time/sps", np.nan)))
    shutil.rmtree(log_dir)
    return float(np.nanmedian(sps[2:]))  # skip compilation and the first evaluation


def main() -> None:
    out: dict = {}
    for env_id in SETTINGS:
        for algo in SETTINGS[env_id]:
            out.setdefault(env_id, {})[algo] = measure(env_id, algo)
            print(f"{env_id:<24} {algo:<10} {out[env_id][algo]:>10,.0f} steps/s", flush=True)
    OUTPUT.write_text(json.dumps(out, indent=1))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
