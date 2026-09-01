"""Ray/RLlib worker benchmark -- PPO through MOSAIC's RayWorkerRuntime.

Measures the overhead of the MOSAIC worker shim:
  - RayWorkerConfig construction and validation
  - Environment factory registration
  - Multi-agent policy mapping setup (even for single-agent)
  - Structured logging, telemetry, analytics manifest
  - Ray cluster env var propagation
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_worker_benchmark(config) -> BenchmarkResult:
    """Run RLlib PPO via MOSAIC RayWorkerRuntime."""
    print_run_header(config.worker_name, "worker", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = f"""\
import os
os.environ["WANDB_MODE"] = "disabled"
from ray_worker.config import (
    RayWorkerConfig, EnvironmentConfig, TrainingConfig,
    ResourceConfig, CheckpointConfig, PolicyConfiguration,
    PettingZooAPIType,
)
from ray_worker.runtime import RayWorkerRuntime

config = RayWorkerConfig(
    run_id="bench_ray_worker_{config.seed}",
    environment=EnvironmentConfig(
        family="gym",
        env_id="{config.env_id}",
        api_type=PettingZooAPIType.GYM,
    ),
    policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
    training=TrainingConfig(
        algorithm="PPO",
        total_timesteps={config.total_timesteps},
        algo_params={{"lr": {config.learning_rate}}},
    ),
    resources=ResourceConfig(num_workers=max({config.num_envs} - 1, 1), num_gpus=0),
    checkpoint=CheckpointConfig(),
    tensorboard=False,
    wandb=False,
    fastlane_enabled=False,
    seed={config.seed},
)

runtime = RayWorkerRuntime(config)
runtime.run()
"""
    cmd = [sys.executable, "-c", script]
    env_overrides = {"RAY_FASTLANE_ENABLED": "0"}

    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, env=env_overrides, timeout=1800)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="ray",
        scenario="worker",
        env_id=config.env_id,
        total_timesteps=config.total_timesteps,
        wall_time_seconds=elapsed,
        steps_per_second=sps,
        peak_memory_mb=peak_mb,
        seed=config.seed,
        num_envs=config.num_envs,
        iteration=getattr(config, "_current_iteration", 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print_run_result(result)
    return result
