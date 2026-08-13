"""Tests for model/features_eval.py (the Feature Lab job)."""

from __future__ import annotations

import pytest
from test_model import _synthetic_df

from features.build import add_features
from model.features_eval import evaluate_feature_deltas, run
from model.train import model_params


def test_evaluate_feature_deltas_reports_baseline_and_sorted_deltas():
    df = add_features(_synthetic_df(n_seasons=6))
    feats = ["grid", "driver_prev_points_mean"]
    result = evaluate_feature_deltas(df, feats, model_params())
    assert result["baseline"]["n_features"] == 2
    assert result["additions"] and result["removals"]
    for row in result["additions"] + result["removals"]:
        assert set(row) == {
            "feature", "mae", "spearman", "delta_mae", "delta_spearman",
        }
    # Sorted best-first by delta_mae (most negative = most helpful).
    add_deltas = [row["delta_mae"] for row in result["additions"]]
    assert add_deltas == sorted(add_deltas)
    # Additions are features outside the baseline set; removals come from it.
    assert all(row["feature"] not in feats for row in result["additions"])
    assert {row["feature"] for row in result["removals"]} == set(feats)


def test_run_validates_season_window_before_building_data():
    with pytest.raises(ValueError, match="at least 4 seasons"):
        run(start=2024, end=2026)
    with pytest.raises(ValueError, match="before start season"):
        run(start=2024, end=2020)
