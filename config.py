"""Configuration loader (stdlib ``tomllib``; no third-party dependency).

Reads ``config.toml`` at the repo root and merges it over built-in defaults,
so every CLI works out of the box even without a config file present.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, dict[str, Any]] = {
    "api": {
        "base_url": "https://api.jolpi.ca/ergast/f1",
        "user_agent": (
            "f1-result-predictor/0.1.0 "
            "(https://github.com/example/f1-result-predictor; contact: dev@example.com)"
        ),
        "sleep_seconds": 0.25,
        "timeout": 30.0,
        "max_retries": 3,
    },
    "data": {
        "cache_dir": "data/raw",
        "dataset": "data/features.parquet",
        "start_season": 2010,
        "end_season": 2025,
    },
    "model": {
        "checkpoint": "data/model/hurdle.joblib",
        "calibrators": "data/model/calibrators.joblib",
        "seed": 42,
    },
    "report": {
        "backtest": "reports/backtest.md",
        "prediction": "reports/prediction.md",
    },
}


def load_config(path: str | Path = "config.toml") -> dict[str, dict[str, Any]]:
    """Load ``config.toml`` merged over :data:`DEFAULTS` (section-wise)."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path)
    if not p.exists():
        return cfg
    with p.open("rb") as fh:
        user = tomllib.load(fh)
    for section, values in user.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg
