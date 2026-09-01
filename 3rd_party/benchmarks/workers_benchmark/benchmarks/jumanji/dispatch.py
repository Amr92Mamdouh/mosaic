from __future__ import annotations

from workers_benchmark.benchmarks.common_dispatch import dispatch_scenario

JUMANJI_FN_NAMES = {
    "native": "run_native_benchmark",
    "worker": "run_worker_benchmark",
    "fastlane": "run_fastlane_gymnasium_benchmark",
}


def run(worker_cfg, scenario_cfg, env_cfg, output_cfg, seed):
    return dispatch_scenario(
        worker_cfg, scenario_cfg, env_cfg, output_cfg, seed,
        fn_names=JUMANJI_FN_NAMES,
    )


__all__ = ["run"]
