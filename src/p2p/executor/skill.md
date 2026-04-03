# Executor Module — Implementation Guide

## Purpose
CLI + API entry point for single training runs. Thin wrapper around existing
`runner.py` / `ppo.py` / `env.py`. Person 1 works here.

## Scope
- Load a reward function from a `.py` file (via `reward_loader`)
- Run training with configurable `TrainConfig`
- Save all artifacts to an iteration directory (via `run_training`)
- Expose as CLI (`python -m p2p.executor`) and API endpoint

## Dependencies (existing files — DO NOT modify)

| File               | What you use                                    |
|--------------------|-------------------------------------------------|
| `reward_loader.py` | `load_from_file(path)` → RewardFunction         |
| `runner.py`        | `run_training(config, reward_fn, source, dir)`   |
| `config.py`        | `TrainConfig`, `TrainConfig.from_json()`         |
| `iteration_record.py` | `IterationRecord(path)` for post-iteration inspection |

## Files to Create

### `src/p2p/executor/__init__.py`
Empty or re-export `run` function.

### `src/p2p/executor/__main__.py`
```python
"""CLI: python -m p2p.executor --reward-fn reward.py [--config config.json] [--runs-dir runs/]"""

import argparse
import json
from pathlib import Path

from p2p.config import TrainConfig
from p2p.training.reward_loader import load_from_file
from p2p.training.runner import run_training


def main():
    parser = argparse.ArgumentParser(description="Prompt2Policy Executor — single training run")
    parser.add_argument("--reward-fn", required=True, help="Path to reward function .py file")
    parser.add_argument("--config", default=None, help="Path to config JSON (optional)")
    parser.add_argument("--runs-dir", default="runs", help="Base directory for runs")
    args = parser.parse_args()

    # Load reward
    reward = load_from_file(Path(args.reward_fn))
    source = Path(args.reward_fn).read_text()

    # Load config
    if args.config:
        config = TrainConfig.from_json(Path(args.config).read_text())
    else:
        config = TrainConfig()

    # Run
    iteration_dir = run_training(config, reward, reward_source=source, runs_dir=args.runs_dir)
    print(f"Completed: {iteration_dir}")


if __name__ == "__main__":
    main()
```

## Tests (`tests/test_executor_cli.py`)

1. `test_executor_cli_help` — `--help` exits 0
2. `test_executor_cli_missing_reward_fn` — exits non-zero without --reward-fn
3. `test_executor_cli_loads_reward_and_trains` — mock `run_training`, verify called with correct args
4. `test_executor_cli_with_config_file` — verify config.json is loaded

## Interface Contract

**Input:**
- `--reward-fn`: Path to a `.py` file containing either:
  - A `RewardFunction` subclass (new style), OR
  - A `reward_fn(obs, action, next_obs, info)` function (legacy)
- `--config`: Optional JSON file matching `TrainConfig` schema

**Output:**
- Iteration directory at `{runs-dir}/{iteration_id}/` with standard layout
- Exit code 0 on success, 1 on error

## Cautions
- Don't move or refactor `runner.py`, `ppo.py`, `env.py` — just import them
- The CLI should be a thin wrapper (< 50 lines)
- Config defaults should match existing `TrainConfig()` defaults
- Reward loading goes through `reward_loader.py`, NOT direct import
