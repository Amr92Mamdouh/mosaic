from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from workers_benchmark.structured_configs import register

register()

CONF_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_config_store_has_all_groups():
    cs = ConfigStore.instance()
    repo = cs.repo
    assert "worker" in repo
    assert "scenario" in repo
    assert "env" in repo
    assert "output" in repo


def test_root_config_composes():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config")
        assert cfg.worker._name == "cleanrl"
        assert cfg.scenario._name == "native"
        assert cfg.env._name == "quick_test"
        assert cfg.output._name == "var"
        assert cfg.seed == 42


def test_all_nine_workers_compose():
    workers = ["cleanrl", "xuance", "ray", "tianshou", "sb3", "sbx", "torchrl", "rltools", "jumanji"]
    for w in workers:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=[f"worker={w}"])
            assert cfg.worker._name == w, f"worker={w} did not compose"
            assert cfg.worker._target_ == f"workers_benchmark.benchmarks.{w}.dispatch.run"


def test_all_three_scenarios_compose():
    for s in ["native", "worker", "fastlane"]:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=[f"scenario={s}"])
            assert cfg.scenario._name == s


def test_all_three_envs_compose():
    for e in ["quick_test", "cartpole", "pendulum"]:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=[f"env={e}"])
            assert cfg.env._name == e


def test_output_var_points_to_absolute_var_dir():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["output=var"])
        assert cfg.output.root == "/home/hamid/Desktop/software/mosaic/var/frameworks/benchmarks"


def test_fastlane_scenario_sets_env_var():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["scenario=fastlane"])
        assert cfg.scenario.env_vars.get("GYM_GUI_FASTLANE_ONLY") == "1"


def test_jumanji_worker_sets_xla_preallocate_false():
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["worker=jumanji"])
        assert cfg.worker.extra_env.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"


def test_chdir_is_false():
    """Hydra must NOT chdir into output dir, otherwise worker relative paths break."""
    config_yaml = Path(CONF_DIR) / "config.yaml"
    text = config_yaml.read_text()
    assert "chdir: false" in text, "configs/config.yaml must pin hydra.job.chdir=false"
