"""CleanRL fastlane benchmark -- PPO through worker + SHM frame streaming.

Same as worker.py but with FastLane enabled via environment variables.
This adds the overhead of:
  - FastLaneTelemetryWrapper around the gym env (env.render() every step)
  - SharedMemory writes to /dev/shm for live GUI visualization
  - Frame throttling and optional downscaling
"""

import os
import sys
import time
from pathlib import Path

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_fastlane_benchmark(config) -> BenchmarkResult:
    """Run CleanRL PPO via worker subprocess with FastLane frame streaming."""
    print_run_header(config.worker_name, "fastlane", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    from cleanrl_worker.config import CleanRLWorkerConfig
    from cleanrl_worker.runtime import CleanRLWorkerRuntime

    worker_config = CleanRLWorkerConfig(
        run_id=f"bench_fastlane_{config.seed}",
        algo="ppo",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        seed=config.seed,
        extras={"cuda": False, "num_envs": config.num_envs},
    )
    runtime = CleanRLWorkerRuntime(
        config=worker_config,
        use_grpc=False,
        grpc_target="",
        dry_run=True,
    )
    args = runtime.build_cleanrl_args()
    module_name = runtime.resolve_entrypoint()[0]

    cmd = [sys.executable, "-m", module_name] + args + [
        "--no-track", "--no-capture-video",
    ]

    # Enable fastlane
    env_overrides = {
        "GYM_GUI_FASTLANE_ONLY": "1",
        "GYM_GUI_FASTLANE_SLOT": "0",
        "GYM_GUI_FASTLANE_VIDEO_MODE": "single",
        "GYM_GUI_FASTLANE_GRID_LIMIT": str(config.num_envs),
    }

    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, env=env_overrides)

    final_return = _parse_last_return(stdout)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="cleanrl",
        scenario="fastlane",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=elapsed,
        steps_per_second=sps,
        peak_memory_mb=peak_mb,
        final_episode_return=final_return,
        seed=config.seed,
        num_envs=config.num_envs,
        iteration=getattr(config, "_current_iteration", 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print_run_result(result)
    return result


def _parse_last_return(stdout: str) -> float:
    last = 0.0
    for line in stdout.splitlines():
        if "episodic_return=" in line:
            try:
                last = float(line.split("episodic_return=")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                pass
    return last
