from __future__ import annotations

import shutil
from pathlib import Path

from workers_benchmark.benchmarks.common_dispatch import dispatch_scenario


def _clean_stale_ray_sessions() -> None:
    """Ray's GCS server can hang on cold-start when /tmp/ray/session_* accumulates.
    Wipe stale sessions before spawning a new one.
    """
    ray_tmp = Path("/tmp/ray")
    if not ray_tmp.is_dir():
        return
    for stale in ray_tmp.glob("session_*"):
        shutil.rmtree(stale, ignore_errors=True)


def run(worker_cfg, scenario_cfg, env_cfg, output_cfg, seed: int):
    _clean_stale_ray_sessions()
    return dispatch_scenario(worker_cfg, scenario_cfg, env_cfg, output_cfg, seed)


__all__ = ["run"]
