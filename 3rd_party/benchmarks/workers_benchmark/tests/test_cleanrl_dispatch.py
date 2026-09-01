from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path("/home/hamid/Desktop/software/mosaic")
REFERENCE_ROOT = REPO_ROOT / "var/frameworks/benchmarks"

CONFIG_FIELDS = ("env_id", "total_timesteps", "num_envs", "seed", "worker_name", "scenario")


@pytest.mark.parametrize("scenario", ["native", "worker", "fastlane"])
def test_hydra_cleanrl_matches_reference_configfields(scenario, tmp_path):
    """New Hydra CLI must produce config-fields matching the existing reference JSON for cleanrl."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "3rd_party/benchmarks:" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-m", "workers_benchmark.train",
            "worker=cleanrl",
            f"scenario={scenario}",
            "env=cartpole",
            "seed=42",
            f"output.root={tmp_path}",
            "env.iterations=1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout={result.stdout[-1500:]}\nstderr={result.stderr[-1500:]}"

    jsons = sorted(tmp_path.rglob(f"cleanrl_{scenario}_*_i1.json"))
    assert jsons, f"no cleanrl_{scenario}_*_i1.json under {tmp_path}"
    hydra_output = json.loads(jsons[-1].read_text())

    reference_file = REFERENCE_ROOT / scenario / f"cleanrl_{scenario}_CartPole_v1_i1.reference.json"
    assert reference_file.exists(), f"reference missing: {reference_file}"
    reference_output = json.loads(reference_file.read_text())

    for field in CONFIG_FIELDS:
        assert hydra_output.get(field) == reference_output.get(field), (
            f"config field {field} drifted for {scenario}: "
            f"hydra={hydra_output.get(field)} reference={reference_output.get(field)}"
        )


def test_hydra_snapshot_files_exist(tmp_path):
    """Verify Hydra .hydra/ snapshot files land alongside per-run outputs."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "3rd_party/benchmarks:" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "-m", "workers_benchmark.train",
            "worker=cleanrl",
            "scenario=native",
            "env=quick_test",
            "seed=42",
            "env.iterations=1",
            f"output.root={tmp_path}",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-1500:]}"

    hydra_dirs = list(tmp_path.rglob(".hydra"))
    assert hydra_dirs, f"no .hydra subdir under {tmp_path}"
    for hd in hydra_dirs:
        assert (hd / "config.yaml").exists(), f"missing {hd}/config.yaml"
        assert (hd / "overrides.yaml").exists(), f"missing {hd}/overrides.yaml"
