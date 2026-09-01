"""Ray/RLlib native benchmark -- New API stack (RLlib 2.x default).

Uses the new RLModule + Learner API stack (default in Ray 2.x).
This measures what users get out-of-the-box with modern RLlib,
in contrast to the old API stack that MOSAIC's RayWorkerRuntime uses.
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_native_new_api_benchmark(config) -> BenchmarkResult:
    """Run RLlib PPO with new API stack (default) via subprocess."""
    print_run_header(config.worker_name, "native_new_api", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    # New API stack: RLlib defaults (enable_rl_module_and_learner=True).
    # Same hyperparameters as old API benchmark for fair comparison.
    script = f"""\
import os, ray
os.environ["WANDB_MODE"] = "disabled"
ray.init(include_dashboard=False, log_to_driver=False)

from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("{config.env_id}")
    .env_runners(num_env_runners=max({config.num_envs} - 1, 1))
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
        scenario="native_new_api",
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
