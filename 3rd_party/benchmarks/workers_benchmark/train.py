from __future__ import annotations

import logging
import os

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from workers_benchmark import structured_configs

log = logging.getLogger(__name__)

structured_configs.register()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    for k, v in cfg.scenario.env_vars.items():
        os.environ[k] = str(v)
    for k, v in cfg.worker.extra_env.items():
        os.environ[k] = str(v)

    dispatch_fn = get_method(cfg.worker._target_)
    result = dispatch_fn(cfg.worker, cfg.scenario, cfg.env, cfg.output, cfg.seed)
    log.info("Benchmark result: %s", result)


if __name__ == "__main__":
    main()
