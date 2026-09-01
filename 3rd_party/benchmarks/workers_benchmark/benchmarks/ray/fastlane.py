"""Ray/RLlib fastlane benchmark -- worker + real SHM frame streaming.

Uses RLlib's on_environment_created callback to replace the sub-env on
worker 1 with a render_mode='rgb_array' version wrapped by
FastLaneTelemetryWrapper. The callback fires on every remote worker
after gym.make_vec() succeeds.

Key findings from investigating the RLlib source (rllib/env/single_agent_env_runner.py):
- RLlib calls gym.make_vec() inside each remote worker process
- The env is created WITHOUT render_mode (defaults to None)
- on_environment_created callback provides access to the env AFTER creation
- env_runner.worker_index identifies which remote worker (1, 2, 3, ...)
- We target worker_index=1 and recreate its sub-env with render_mode='rgb_array'
"""

import sys
import time

from workers_benchmark.utils import (
    BenchmarkResult, run_subprocess_timed, print_run_header, print_run_result,
)


def run_fastlane_benchmark(config) -> BenchmarkResult:
    """Run RLlib PPO with FastLane via on_environment_created callback."""
    print_run_header(config.worker_name, "fastlane", config.env_id,
                     config.total_timesteps, config.num_envs, config.seed,
                     getattr(config, "_current_iteration", 1),
                     config.iterations)

    script = f"""\
import os, gymnasium as gym
os.environ["WANDB_MODE"] = "disabled"
os.environ["GYM_GUI_FASTLANE_ONLY"] = "1"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

class FastLaneCallback(RLlibCallback):
    def on_environment_created(self, *, env_runner, env, env_context, **kwargs):
        # env_runner.worker_index: 1,2,3 for remote workers. Target worker 1 only.
        worker_idx = getattr(env_runner, "worker_index", -1)
        if worker_idx != 1:
            return

        try:
            from cleanrl_worker.fastlane import FastLaneTelemetryWrapper

            class _FLConfig:
                enabled = True
                slot = 0
                run_id = "bench_ray_fl_{config.seed}"
                agent_id = "ray-agent"
                worker_id = "ray-worker"
                seed = {config.seed}
                total_envs = 1
                video_mode = "single"
                grid_limit = 4

            # Navigate to the SyncVectorEnv
            underlying = env
            while hasattr(underlying, "env"):
                underlying = underlying.env

            if hasattr(underlying, "envs") and len(underlying.envs) > 0:
                # Close the original sub-env (render_mode=None)
                old_sub = underlying.envs[0]
                old_sub.close()

                # Recreate with render_mode="rgb_array"
                new_sub = gym.make("{config.env_id}", render_mode="rgb_array")

                # Wrap with FastLane
                wrapped = FastLaneTelemetryWrapper(new_sub, _FLConfig(), 0)
                underlying.envs[0] = wrapped

                print(f"[FastLane] Wrapped env on worker {{worker_idx}} with render_mode=rgb_array", flush=True)
            else:
                print(f"[FastLane] No sub-envs found on worker {{worker_idx}}", flush=True)
        except Exception as e:
            print(f"[FastLane] Failed to wrap env on worker {{worker_idx}}: {{e}}", flush=True)

ray.init(include_dashboard=False, log_to_driver=False)

rllib_config = (
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
    .callbacks(FastLaneCallback)
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

algo = rllib_config.build()
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
    env_overrides = {
        "GYM_GUI_FASTLANE_ONLY": "1",
        "GYM_GUI_FASTLANE_SLOT": "0",
        "GYM_GUI_FASTLANE_VIDEO_MODE": "single",
    }

    elapsed, peak_mb, stdout, _ = run_subprocess_timed(cmd, env=env_overrides, timeout=1800)
    sps = config.total_timesteps / elapsed if elapsed > 0 else 0.0

    result = BenchmarkResult(
        worker_name="ray",
        scenario="fastlane",
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
