"""Tests for model/search.py: config sampling and walk-forward evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.build import add_features
from model.search import evaluate_config, sample_configs, search

from test_model import _synthetic_df


def test_sample_configs_seeded_and_in_range():
    a = sample_configs(4, seed=1)
    b = sample_configs(4, seed=1)
    assert a == b  # deterministic
    for cfg in a:
        assert set(cfg) == {"max_iter", "learning_rate", "max_depth",
                            "min_samples_leaf", "l2_regularization"}
        assert cfg["max_depth"] in (3, 5, 7)
        assert 0.03 <= cfg["learning_rate"] <= 0.1


def test_evaluate_config_returns_mae_and_spearman():
    df = add_features(_synthetic_df(n_seasons=8))
    mae, spearman = evaluate_config(df, {}, max_test_season=2018)
    assert 0.0 < mae < 10.0
    assert 0.0 < spearman <= 1.0
    with pytest.raises(ValueError, match="no walk-forward splits"):
        evaluate_config(df, {}, max_test_season=2014)


def test_search_ranks_configs():
    df = add_features(_synthetic_df(n_seasons=8))
    results = search(df, n=4, seed=0, max_test_season=2018)
    assert len(results) == 4
    assert {"mae", "spearman", "score"} <= set(results.columns)
    assert results["score"].is_monotonic_increasing
    # Configs are cast to proper types for sklearn (check column dtypes:
    # pandas iterrows upcasts int columns to float, so check the column).
    assert results["max_iter"].dtype.kind == "i"
    assert results["learning_rate"].dtype.kind == "f"


def test_evaluate_config_matches_run_backtest_aggregation():
    """The search's MAE/Spearman must equal run_backtest's per-race model row.

    Regression: evaluate_config used to pool whole seasons into race_metrics
    (corrupting Spearman); it must aggregate per race like run_backtest.
    """
    from model.evaluate import run_backtest

    df = add_features(_synthetic_df(n_seasons=8))
    overall, _ = run_backtest(df)
    mae, spearman = evaluate_config(df, {}, max_test_season=None)
    # overall is rounded to 4 decimals by run_backtest.
    assert mae == pytest.approx(overall.loc["model", "mae"], abs=5e-5)
    assert spearman == pytest.approx(overall.loc["model", "spearman"], abs=5e-5)
