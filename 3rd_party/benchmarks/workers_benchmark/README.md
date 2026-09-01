# MOSAIC Worker Overhead Benchmark Suite

Measures the overhead of wrapping RL frameworks with MOSAIC's worker
abstraction and FastLane visual streaming pipeline.

Three scenarios per framework:

| Scenario | What it measures |
|----------|------------------|
| **Native** | Framework running directly, no MOSAIC wrapping |
| **Worker** | Framework wrapped by MOSAIC worker (logging, config, telemetry) |
| **FastLane** | Worker + real-time frame streaming via shared memory |

## Frameworks benchmarked

| Worker | Framework | Architecture | Status |
|--------|-----------|-------------|--------|
| `cleanrl` | CleanRL | Single-process | Complete |
| `xuance` | XuanCe | Single-process | Complete |
| `ray` | Ray/RLlib | Distributed (multi-process) | Complete |
| `tianshou` | Tianshou | Single-process | Complete |
| `sb3` | Stable-Baselines3 | Single-process | Complete |
| `sbx` | SBX (JAX) | Single-process | Complete |
| `torchrl` | TorchRL | Single-process | Complete |
| `rltools` | RLtools (C++) | Single-process | Complete |
| `jumanji` | Jumanji (JAX) | Single-process (JIT) | Complete |

## Prerequisites

```bash
# From the mosaic root directory
export PYTHONPATH="3rd_party/benchmarks:$PYTHONPATH"
```

### Per-worker dependencies

Each worker has its own requirements file in `requirements/`:

```bash
pip install -r requirements/base.txt           # shared deps (includes hydra-core, omegaconf)
pip install -r requirements/cleanrl_worker.txt  # CleanRL
pip install -r requirements/ray_worker.txt      # Ray/RLlib
# ... etc.
```

### System requirements

- **RAM**: 8 GB minimum, 16 GB recommended. Ray/RLlib spawns multiple
  worker processes (~1.1 GB each). Running under memory pressure inflates
  wall times due to swap thrashing.
- **CPU**: 4+ cores. Ray benchmarks use `num_cpus=4`.

## Running benchmarks

The sole entry point is `workers_benchmark.train`, driven by
[Hydra](https://hydra.cc/) configuration. Every run saves a
`.hydra/config.yaml` snapshot alongside the result JSON, making
results fully reproducible from config alone.

### Single run

```bash
# CleanRL, native scenario, CartPole, seed 42
python -m workers_benchmark.train worker=cleanrl scenario=native env=cartpole seed=42

# Quick smoke test (10 K steps, 3 iterations, ~5 s)
python -m workers_benchmark.train worker=cleanrl scenario=native env=quick_test
```

### Grid sweep (all scenarios for one worker)

```bash
python -m workers_benchmark.train -m \
  worker=cleanrl \
  scenario=native,worker,fastlane \
  env=cartpole \
  seed=42
```

### Full 9-worker matrix

```bash
python -m workers_benchmark.train -m \
  worker=cleanrl,xuance,ray,tianshou,sb3,sbx,torchrl,rltools,jumanji \
  scenario=native,worker,fastlane \
  env=cartpole \
  seed=42
```

### Override individual config values

```bash
# Increase total timesteps for one run
python -m workers_benchmark.train worker=cleanrl env.total_timesteps=500000

# Change output root
python -m workers_benchmark.train worker=cleanrl output=local

# Combine overrides
python -m workers_benchmark.train worker=sb3 scenario=fastlane env=pendulum seed=1
```

## Analyzing results

### Summary tables

```bash
python -m workers_benchmark.scripts.analyze_results \
  --results-dir /path/to/mosaic/var/frameworks/benchmarks
```

Prints per-worker overhead tables and cross-worker native comparison.

### Publication charts

```bash
python -m workers_benchmark.scripts.plot_publication \
  --results-dir /path/to/mosaic/var/frameworks/benchmarks \
  --env CartPole-v1
```

Generates:
- `var/frameworks/benchmarks/plots/combined_overhead.png`: all frameworks side by side
- `var/frameworks/benchmarks/plots/overhead_ratios.png`: horizontal overhead ratio bars

To include pre-migration frozen baselines in the plot:

```bash
python -m workers_benchmark.scripts.plot_publication \
  --results-dir /path/to/mosaic/var/frameworks/benchmarks \
  --env CartPole-v1 \
  --include-reference
```

## Environment presets

| Preset | Env | Steps | Envs | Iterations | Purpose |
|--------|-----|-------|------|------------|---------|
| `quick_test` | CartPole-v1 | 10,000 | 1 | 3 | Smoke test (~5 s) |
| `cartpole` | CartPole-v1 | 100,000 | 4 | 5 | Standard benchmark |
| `pendulum` | Pendulum-v1 | 300,000 | 1 | 10 | Matches rl-tools paper |

## Metrics collected

- **Wall time**: `time.perf_counter()` (high-resolution, not CPU time)
- **Steps per second**: `total_timesteps / wall_time`
- **Peak memory**: Background thread samples `VmPeak` from `/proc/self/status`
  every 0.5 s. Fallback: `resource.getrusage` or `psutil`.
- **Overhead ratio**: `scenario_mean / native_mean` (1.0x = no overhead)

## Output files

Results land under `var/frameworks/benchmarks/` (configurable via `output=var`
or `output=local`):

```
var/frameworks/benchmarks/
  native/
    cleanrl_native_CartPole_v1_i1.json
    cleanrl_native_CartPole_v1_i2.json
    ...
  worker/
    cleanrl_worker_CartPole_v1_i1.json
    ...
  fastlane/
    cleanrl_fastlane_CartPole_v1_i1.json
    ...
  plots/
    combined_overhead.png
    overhead_ratios.png
  .hydra_snapshots/
    multirun/YYYY-MM-DD/HH-MM-SS/
      cleanrl_native_cartpole_seed42/
        .hydra/config.yaml      # full resolved config snapshot
        .hydra/overrides.yaml   # CLI overrides used
```

Each JSON contains:

```json
{
  "worker_name": "cleanrl",
  "scenario": "native",
  "env_id": "CartPole-v1",
  "total_timesteps": 100000,
  "wall_time_seconds": 32.1,
  "steps_per_second": 3115,
  "peak_memory_mb": 312.4,
  "seed": 42,
  "num_envs": 4,
  "iteration": 1,
  "timestamp": "2026-08-31 10:21:49"
}
```

## Reproducing results from scratch

```bash
# 1. Set PYTHONPATH from mosaic root
export PYTHONPATH="3rd_party/benchmarks:$PYTHONPATH"

# 2. Run the full matrix (single seed, 5 iterations each)
python -m workers_benchmark.train -m \
  worker=cleanrl,xuance,ray,tianshou,sb3,sbx,torchrl,rltools,jumanji \
  scenario=native,worker,fastlane \
  env=cartpole \
  seed=42

# 3. Generate publication charts
python -m workers_benchmark.scripts.plot_publication \
  --results-dir var/frameworks/benchmarks \
  --env CartPole-v1
```

**Estimated time**: ~2 hours for 7 functional workers on a 4-core machine
with 16 GB RAM. Ray/RLlib is the slowest (~15 min for 3 scenarios).

## Directory structure

```
workers_benchmark/
  train.py                    # Hydra entry point (python -m workers_benchmark.train)
  structured_configs.py       # Dataclass schemas registered with ConfigStore
  utils.py                    # BenchmarkResult, BenchmarkTimer, run_subprocess_timed
  configs/
    config.yaml               # Root config with defaults list
    worker/
      cleanrl.yaml            # _target_: workers_benchmark.benchmarks.cleanrl.dispatch.run
      xuance.yaml
      ray.yaml
      tianshou.yaml
      sb3.yaml
      sbx.yaml
      torchrl.yaml
      rltools.yaml
      jumanji.yaml
    scenario/
      native.yaml
      worker.yaml
      fastlane.yaml           # sets GYM_GUI_FASTLANE_ONLY=1
    env/
      quick_test.yaml
      cartpole.yaml
      pendulum.yaml
    output/
      var.yaml                # root: var/frameworks/benchmarks (default)
      local.yaml              # root: 3rd_party/benchmarks/workers_benchmark/results
  benchmarks/
    __init__.py               # AVAILABLE_WORKERS list
    cleanrl/
      dispatch.py             # routes to native.py / worker.py / fastlane.py
      native.py
      worker.py
      fastlane.py
    xuance/
    ray/
    tianshou/
    sb3/
    sbx/
    torchrl/
    rltools/
    jumanji/
  scripts/
    analyze_results.py        # Load JSONs, compute stats, print tables
    plot_publication.py       # Publication-quality combined charts
    compare_workers.py        # Multi-subplot comparison
    compare_workers_lines.py  # Line graphs across iterations
  tests/
    test_structured_configs.py
    test_cleanrl_dispatch.py
    test_*_dispatch.py        # one per worker
```

## Adding a new worker

1. Create `benchmarks/<name>/` with `__init__.py`, `native.py`,
   `worker.py`, `fastlane.py`.
2. Each scenario file exports a `run_{scenario}_benchmark(...)` function
   returning a `BenchmarkResult`.
3. Create `benchmarks/<name>/dispatch.py` following this pattern:

   ```python
   from __future__ import annotations

   def run(worker_cfg, scenario_cfg, env_cfg, output_cfg, seed: int):
       if scenario_cfg._name == "native":
           from .native import run_native_benchmark as fn
       elif scenario_cfg._name == "worker":
           from .worker import run_worker_benchmark as fn
       elif scenario_cfg._name == "fastlane":
           from .fastlane import run_fastlane_benchmark as fn
       else:
           raise ValueError(f"unknown scenario: {scenario_cfg._name}")
       return fn(
           env_id=env_cfg.gym_id,
           total_timesteps=env_cfg.total_timesteps,
           num_envs=env_cfg.num_envs,
           seed=seed,
           iterations=env_cfg.iterations,
       )

   __all__ = ["run"]
   ```

4. Add `configs/worker/<name>.yaml`:

   ```yaml
   _name: <name>
   _target_: workers_benchmark.benchmarks.<name>.dispatch.run
   supports:
     scenarios: [native, worker, fastlane]
   timeout_s: 3600
   extra_env: {}
   ```

5. Add the worker name to `AVAILABLE_WORKERS` in `benchmarks/__init__.py`.
6. See `benchmarks/cleanrl/` as the reference implementation.
