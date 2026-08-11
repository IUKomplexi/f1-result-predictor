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

import argparse
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16, help="configs to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-test-season", type=int, default=2019,
                        help="latest test season in the search window")
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--dataset", default="data/features.parquet")
    parser.add_argument(
        "--enable-features", default="",
        help="comma-separated features to enable on top of config",
    )
    parser.add_argument(
        "--disable-features", default="",
        help="comma-separated features to disable on top of config",
    )
    args = parser.parse_args()

    client = F1Client(cache_dir=args.cache_dir, refresh=args.refresh)
    df = build_dataset(client, range(args.start, args.end + 1), cache_path=args.dataset)
    feats = enabled_features(
        load_config(),
        enable=[f for f in args.enable_features.split(",") if f],
        disable=[f for f in args.disable_features.split(",") if f],
    )
    results = search(df, n=args.n, seed=args.seed,
                     max_test_season=args.max_test_season, features=feats)

    print(f"Walk-forward search (test seasons <= {args.max_test_season}):")
    print(results.round(4).to_string(index=False))
    best_row = results.iloc[0].drop(["mae", "spearman", "score"]).to_dict()
    best = {
        k: (int(v) if k in {"max_iter", "max_depth", "min_samples_leaf"} else float(v))
        for k, v in best_row.items()
    }
    print("\nBest configuration (paste into model/train.py DEFAULT_PARAMS):")
    print(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
