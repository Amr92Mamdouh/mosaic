from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class WorkerConfig:
    _name: str = "???"
    _target_: str = "???"
    supports: dict[str, Any] = field(default_factory=lambda: {"scenarios": []})
    timeout_s: int = 3600
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    _name: str = "???"
    module_suffix: str = "???"
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class EnvConfig:
    _name: str = "???"
    gym_id: str = "???"
    total_timesteps: int = 10000
    num_envs: int = 1
    iterations: int = 3


@dataclass
class OutputConfig:
    _name: str = "???"
    root: str = "???"


@dataclass
class BenchmarkConfig:
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    log_level: str = "INFO"


def register() -> None:
    cs = ConfigStore.instance()
    cs.store(group="worker", name="base_worker", node=WorkerConfig)
    cs.store(group="scenario", name="base_scenario", node=ScenarioConfig)
    cs.store(group="env", name="base_env", node=EnvConfig)
    cs.store(group="output", name="base_output", node=OutputConfig)
    cs.store(name="base_config", node=BenchmarkConfig)


register()
