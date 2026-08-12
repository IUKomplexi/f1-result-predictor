"""Walk-forward-validated hyperparameter search for the hurdle models.

The search samples gradient-boosting configurations, evaluates each one on a
reduced-window walk-forward (to keep it fast), and ranks them by a combined
score of expected-points MAE and ranking correlation (Spearman).

Usage::

    python model/search.py [--n 16] [--max-test-season 2019] [--seed 0]

``--max-test-season`` caps the latest test season (the default 2019 validates
on 2018-2019 with training on everything before, which is a fast but
representative window). The best configuration is printed; paste it into
``model/train.py``'s ``DEFAULT_PARAMS`` to ship it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from f1core.config import load_config
from f1data import F1Client
from features.build import build_dataset
from features.registry import enabled_features
from model.evaluate import race_metrics
from model.train import (
    HurdleModels,
    prepare,
    quantize_points,
    walk_forward_seasons,
)

PARAM_RANGES = {
    "max_iter": [200, 300, 400, 600],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "max_depth": [3, 5, 7],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.5, 1.0, 2.0],
}


def sample_configs(n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` random hyperparameter configurations (seeded, reproducible).

    Integer parameters are cast to int explicitly (numpy's ``choice`` may
    return floats for a list input) because sklearn validates parameter types.
    """
    rng = np.random.default_rng(seed)
    int_params = {"max_iter", "max_depth", "min_samples_leaf"}
    configs = []
    for _ in range(n):
        config = {}
        for name, values in PARAM_RANGES.items():
            value = rng.choice(values)
            config[name] = int(value) if name in int_params else float(value)
        configs.append(config)
    return configs


def evaluate_config(
    df: pd.DataFrame,
    params: dict[str, Any],
    max_test_season: int | None = None,
    features: list[str] | None = None,
) -> tuple[float, float]:
    """Mean per-race MAE and Spearman of one config on a walk-forward window.

    Metrics are computed per race (rankings are only meaningful within a
    single race — pooling rounds would corrupt them), then averaged. Expected
    points are quantized, matching the deployed output. ``features`` selects
    the model columns (default: the full set).
    """
    mae, spearman = [], []
    for train, test, season in walk_forward_seasons(df):
        if max_test_season is not None and season > max_test_season:
            break
        model = HurdleModels(seed=42, params=params).fit(*prepare(train, features))
        X_test, _ = prepare(test, features)
        test = test.copy()
        test["pred_points"] = quantize_points(model.predict_expected_points(X_test))
        for _, race in test.groupby(["season", "round"]):
            m = race_metrics(race)
            mae.append(m["mae"])
            spearman.append(m["spearman"])
    if not mae:
        raise ValueError("no walk-forward splits to evaluate")
    return float(np.mean(mae)), float(np.mean(spearman))


def search(
    df: pd.DataFrame,
    n: int = 16,
    seed: int = 0,
    max_test_season: int | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate ``n`` configs and rank them (best = lowest combined score)."""
    rows = []
    for params in sample_configs(n, seed):
        mae, spearman = evaluate_config(df, params, max_test_season, features)
        rows.append({**params, "mae": mae, "spearman": spearman})
    results = pd.DataFrame(rows)
    # Lower is better: MAE rank + inverse-Spearman rank.
    results["score"] = results["mae"].rank() + (1.0 - results["spearman"]).rank()
    return results.sort_values("score").reset_index(drop=True)


def run(
    *,
    n: int = 16,
    seed: int = 0,
    max_test_season: int = 2019,
    start: int = 2010,
    end: int = 2025,
    refresh: bool = False,
    cache_dir: str = "data/raw",
    dataset: str = "data/features.parquet",
    enable_features: Sequence[str] = (),
    disable_features: Sequence[str] = (),
    cfg: dict | None = None,
    log=None,
) -> dict:
    """Run a walk-forward hyperparameter search and return JSON-safe results.

    ``log`` is an optional progress callback (web job runner). The returned
    dict carries the full ranked result table plus the ``best`` configuration
    (ready to be written to ``[model.params]`` by the dashboard).
    """
    log = log or (lambda msg: print(msg, flush=True))
    cfg = cfg or load_config()
    client = F1Client(cache_dir=cache_dir, refresh=refresh)
    log(f"Building dataset {start}-{end} ...")
    df = build_dataset(client, range(start, end + 1), cache_path=dataset)
    feats = enabled_features(
        cfg, enable=list(enable_features), disable=list(disable_features)
    )
    log(f"Searching {n} configs (test seasons <= {max_test_season}, {len(feats)} features) ...")
    results = search(df, n=n, seed=seed, max_test_season=max_test_season, features=feats)

    best_row = results.iloc[0].drop(["mae", "spearman", "score"]).to_dict()
    best = {
        k: (int(v) if k in {"max_iter", "max_depth", "min_samples_leaf"} else float(v))
        for k, v in best_row.items()
    }
    log(f"Best config: {best}")
    int_keys = {"max_iter", "max_depth", "min_samples_leaf"}
    return {
        "results": [
            {k: (int(v) if k in int_keys else float(v)) for k, v in row.items()}
            for row in results.round(4).to_dict(orient="records")
        ],
        "best": best,
        "n": n,
        "seed": seed,
        "max_test_season": max_test_season,
        "n_features": len(feats),
        "features": feats,
    }


