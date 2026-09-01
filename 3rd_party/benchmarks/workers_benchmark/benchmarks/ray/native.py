"""Ray/RLlib native benchmark -- PPO directly via RLlib, no MOSAIC wrapping.

Matches every RayWorkerRuntime setting exactly (old API stack,
create_env_on_local_worker, hyperparameters) so the only variable
is the MOSAIC wrapper overhead itself.
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_native_benchmark(config) -> BenchmarkResult:
    """Run RLlib PPO directly via subprocess (no MOSAIC worker wrapping)."""
    print_run_header(config.worker_name, "native", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    # Match every RayWorkerRuntime setting for a fair comparison:
    # - Old API stack (enable_rl_module_and_learner=False)
    # - create_env_on_local_worker=True (worker sets this)
    # - Same hyperparameters from metadata/ray_rllib/0.1.0/schemas.json
    script = f"""\
import os, ray
os.environ["WANDB_MODE"] = "disabled"
ray.init(include_dashboard=False, log_to_driver=False)

from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("{config.env_id}")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .env_runners(
        num_env_runners=max({config.num_envs} - 1, 1),
        create_env_on_local_worker=True,
    )
    .training(
        lr=0.0003,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.3,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        num_epochs=30,
        minibatch_size=128,
        train_batch_size_per_learner=4000,
    )
    .resources(num_gpus=0)
)

algo = config.build()
total = 0
while total < {config.total_timesteps}:
    result = algo.train()
    total = result.get("num_env_steps_sampled_lifetime", 0)
    if total == 0:
        total = result.get("timesteps_total", 0)
    if total == 0:
        break
algo.stop()
ray.shutdown()
"""
    cmd = [sys.executable, "-c", script]
    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, timeout=1800)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="ray",
        scenario="native",
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
