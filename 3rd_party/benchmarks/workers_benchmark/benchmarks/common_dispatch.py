from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional

from workers_benchmark.configs.common import BenchmarkConfig
from workers_benchmark.utils import BenchmarkResult

DEFAULT_FN_NAMES = {
    "native": "run_native_benchmark",
    "worker": "run_worker_benchmark",
    "fastlane": "run_fastlane_benchmark",
}


def dispatch_scenario(
    worker_cfg,
    scenario_cfg,
    env_cfg,
    output_cfg,
    seed: int,
    fn_names: Optional[dict] = None,
) -> List[BenchmarkResult]:
    """Run one scenario iteration loop for any worker following the standard signature.

    Every worker exposes `run_{native,worker,fastlane}_benchmark(config: BenchmarkConfig) -> BenchmarkResult`
    in a module named `workers_benchmark.benchmarks.{worker_name}.{scenario}`. This helper resolves
    that module, calls it env_cfg.iterations times with proper iteration numbering, and saves each
    result under `output_cfg.root/{scenario}/`.

    Pass fn_names to override the function name when a worker deviates from the default
    (e.g. jumanji has run_fastlane_gymnasium_benchmark instead of run_fastlane_benchmark).
    """
    worker_name = worker_cfg._name
    scenario = scenario_cfg._name
    resolved_fn_names = fn_names or DEFAULT_FN_NAMES
    if scenario not in resolved_fn_names:
        raise ValueError(f"unknown {worker_name} scenario: {scenario}")

    module_name = f"workers_benchmark.benchmarks.{worker_name}.{scenario}"
    module = importlib.import_module(module_name)
    bench_fn = getattr(module, resolved_fn_names[scenario])

    output_dir = Path(output_cfg.root) / scenario
    total_iters = env_cfg.iterations
    results: List[BenchmarkResult] = []
    for i in range(1, total_iters + 1):
        config = BenchmarkConfig(
            env_id=env_cfg.gym_id,
            total_timesteps=env_cfg.total_timesteps,
            num_envs=env_cfg.num_envs,
            seed=seed,
            worker_name=worker_name,
            scenario=scenario,
            iterations=env_cfg.iterations,
            enable_fastlane=(scenario == "fastlane"),
        )
        setattr(config, "_current_iteration", i)
        result = bench_fn(config)
        result.save(output_dir)
        results.append(result)
    return results
