#!/usr/bin/env python3
"""Benchmark okapi's algorithms on MuJoCo Playground tasks.

    uv run benchmarks/run.py                                # everything in SETTINGS, 4 seeds
    uv run benchmarks/run.py --algos sac bro --envs CheetahRun --seeds 0 1
    uv run benchmarks/run.py --lanes 2                      # run two jobs at a time
    uv run benchmarks/run.py --collect                      # only aggregate finished runs

Each job trains in its own process and writes ``runs/benchmark/<env>/<algo>/seed<k>``.
Finished jobs are skipped, so the script can be interrupted and restarted. Results
are aggregated into ``docs/benchmark_results.json``; plot them with ``plot.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "benchmark"
RESULTS = ROOT / "docs" / "benchmark_results.json"

# Per-environment hyperparameters on top of each algorithm's defaults.
# PPO uses MuJoCo Playground's tuned Brax PPO settings.
CHEETAH_PPO = dict(
    num_envs=2048, total_timesteps=60_000_000, num_steps=30, num_rollout_rounds=16,
    num_minibatches=32, update_epochs=16, learning_rate=1e-3, gamma=0.995, clip_coef=0.3,
    reward_scaling=10.0, max_grad_norm=None, actor_hidden_sizes=[32, 32, 32, 32],
    critic_hidden_sizes=[256, 256, 256, 256, 256], warmup_stats=False)
GO1 = dict(value_obs_key="privileged_state")
GO1_PPO = dict(GO1, num_envs=8192, total_timesteps=200_000_000, num_steps=20,
               num_minibatches=32, update_epochs=4, learning_rate=3e-4, gamma=0.97,
               clip_coef=0.3, max_grad_norm=1.0, warmup_stats=False)
CHEETAH_OFF_POLICY = dict(total_timesteps=5_000_000)
GO1_OFF_POLICY = dict(GO1, total_timesteps=10_000_000, num_envs=1024, batch_size=8192,
                      updates_per_step=4, gamma=0.97, n_step=3, buffer_size=1024 * 1024,
                      learning_starts=32_768)

SETTINGS: dict[str, dict[str, dict]] = {
    "CheetahRun": {
        "ppo": CHEETAH_PPO,
        "ppo_gru": dict(CHEETAH_PPO, update_epochs=8, learning_rate=3e-4, max_grad_norm=1.0,
                        actor_hidden_sizes=[128, 128], critic_hidden_sizes=[256, 256, 256],
                        gru_size=128),
        "sac": CHEETAH_OFF_POLICY,
        "simbav2": CHEETAH_OFF_POLICY,
        "bro": CHEETAH_OFF_POLICY,
        "dreamerv3": dict(total_timesteps=1_000_000),
        "tdmpc2": dict(total_timesteps=1_000_000),
    },
    "Go1JoystickFlatTerrain": {
        "ppo": GO1_PPO,
        "ppo_gru": dict(GO1_PPO, actor_hidden_sizes=[256, 128], critic_hidden_sizes=[256, 128],
                        gru_size=128),
        "sac": GO1_OFF_POLICY,
        "simbav2": GO1_OFF_POLICY,
        "bro": GO1_OFF_POLICY,
        "dreamerv3": dict(total_timesteps=1_000_000, gamma=0.97),
        "tdmpc2": dict(total_timesteps=1_000_000, gamma=0.97),
    },
}


def job_dir(env_id: str, algo: str, seed: int) -> Path:
    return RUN_DIR / env_id / algo / f"seed{seed}"


def run_job(env_id: str, algo: str, seed: int) -> None:
    """Train one configuration in this process."""
    import okapi

    module = okapi.get_algorithm(algo)
    out = job_dir(env_id, algo, seed)
    if out.exists():  # leftovers of an interrupted run would be appended to
        shutil.rmtree(out)
    cfg = module.Config.from_dict({
        **SETTINGS[env_id][algo], "env_id": env_id, "seed": seed, "num_evals": 20,
        "num_logs": 50, "log_dir": str(out.parent), "run_name": out.name, "checkpoint": True,
    })
    module.train(cfg)
    (out / "DONE").touch()


def collect() -> dict:
    """Aggregate the evaluation curves of all finished runs into ``RESULTS``."""
    results: dict = {}
    for done in sorted(RUN_DIR.glob("*/*/seed*/DONE")):
        run = done.parent
        env_id, algo, seed = run.parent.parent.name, run.parent.name, int(run.name[4:])
        rows = [json.loads(line) for line in open(run / "metrics.jsonl")]
        evals = [r for r in rows if "eval/episode_reward" in r]
        results.setdefault(env_id, {}).setdefault(algo, {})[seed] = {
            "steps": [r["step"] for r in evals],
            "reward": [r["eval/episode_reward"] for r in evals],
            "walltime": [r["time/walltime"] for r in evals],
        }
    RESULTS.write_text(json.dumps(results, indent=1))
    print(f"Wrote {RESULTS}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--envs", nargs="*", default=list(SETTINGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--lanes", type=int, default=1, help="Jobs running at the same time")
    ap.add_argument("--collect", action="store_true", help="Only aggregate finished runs")
    ap.add_argument("--job", nargs=3, metavar=("ENV", "ALGO", "SEED"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.job:
        env_id, algo, seed = args.job
        return run_job(env_id, algo, int(seed))
    if args.collect:
        collect()
        return

    queue = [(e, a, s) for s in args.seeds for e in args.envs
             for a in (args.algos or SETTINGS[e]) if not (job_dir(e, a, s) / "DONE").exists()]
    print(f"{len(queue)} jobs, {args.lanes} lane(s)")
    running: list[tuple[subprocess.Popen, tuple]] = []
    retries: dict[tuple, int] = {}
    while queue or running:
        while queue and len(running) < args.lanes:
            job = queue.pop(0)
            log = job_dir(*job).with_suffix(".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, __file__, "--job", job[0], job[1], str(job[2])]
            running.append((subprocess.Popen(cmd, stdout=open(log, "w"),
                                             stderr=subprocess.STDOUT, cwd=ROOT), job))
            print(f"started  {job}", flush=True)
        for proc, job in list(running):
            if proc.poll() is not None:
                running.remove((proc, job))
                print(f"finished {job} (exit {proc.returncode})", flush=True)
                if proc.returncode != 0 and retries.get(job, 0) < 2:  # e.g. GPU out of memory
                    retries[job] = retries.get(job, 0) + 1
                    queue.append(job)
                    print(f"requeued {job}", flush=True)
        time.sleep(5)
    collect()


if __name__ == "__main__":
    main()
