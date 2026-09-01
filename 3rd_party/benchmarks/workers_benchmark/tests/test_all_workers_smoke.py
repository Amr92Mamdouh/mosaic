"""Smoke test: every worker's Hydra dispatch runs end-to-end on env=quick_test.

Runs 9 workers × 3 scenarios (each with env.iterations=1 for speed), checks
that the CLI exits 0 and produces a JSON with the expected schema. Does NOT
compare against reference values; that's test_cleanrl_dispatch.py's job.

xfail markers explain pre-existing upstream bugs in sbx and rltools that also
fail on the pre-Hydra CLI.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path("/home/hamid/Desktop/software/mosaic")

WORKERS = [
    "cleanrl",
    "xuance",
    "ray",  # slow (~100s per iteration due to RLlib actor spawn), but works
    "tianshou",
    "sb3",
    pytest.param(
        "sbx",
        marks=pytest.mark.xfail(
            reason="Upstream bug: sbx>=0.19 imports sbx.crossq at package init, "
            "which imports tensorflow_probability.substrates.jax, which calls "
            "jax.interpreters.xla.pytype_aval_mappings — removed in current jax. "
            "Fails identically on pre-Hydra CLI. Fix requires jax/tfp/sbx version alignment.",
            strict=False,
        ),
    ),
    "torchrl",
    pytest.param(
        "rltools",
        marks=pytest.mark.xfail(
            reason="Upstream bug: rltools.SAC(env_factory, ...) raises "
            "IndexError: tuple index out of range at C++ binding construction. "
            "Fails identically on pre-Hydra CLI. Fix requires rltools rebuild or upstream patch.",
            strict=False,
        ),
    ),
    "jumanji",
]
SCENARIOS = ["native", "worker", "fastlane"]

REQUIRED_JSON_FIELDS = (
    "worker_name",
    "scenario",
    "env_id",
    "total_timesteps",
    "wall_time_seconds",
    "steps_per_second",
    "seed",
    "num_envs",
    "iteration",
)


@pytest.mark.parametrize("worker", WORKERS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_hydra_dispatch_runs_and_saves_json(worker, scenario, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = "3rd_party/benchmarks:" + env.get("PYTHONPATH", "")
    if worker == "ray":
        # Ray cold-start is flaky when /tmp/ray/session_* accumulates from prior runs.
        import shutil
        for stale in Path("/tmp/ray").glob("session_*"):
            shutil.rmtree(stale, ignore_errors=True)
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-m", "workers_benchmark.train",
            f"worker={worker}",
            f"scenario={scenario}",
            "env=quick_test",
            "env.iterations=1",
            "seed=42",
            f"output.root={tmp_path}",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI failed for worker={worker} scenario={scenario}:\n"
        f"stdout={result.stdout[-1500:]}\nstderr={result.stderr[-1500:]}"
    )

    jsons = sorted((tmp_path / scenario).glob(f"{worker}_{scenario}_*_i1.json"))
    assert jsons, f"no {worker}_{scenario}_*_i1.json under {tmp_path}/{scenario}"
    output = json.loads(jsons[-1].read_text())
    for field in REQUIRED_JSON_FIELDS:
        assert field in output, f"missing field {field!r} in output JSON: {output}"
    assert output["worker_name"] == worker
    assert output["scenario"] == scenario
    assert output["seed"] == 42
