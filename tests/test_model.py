"""Tests for the hurdle model, walk-forward splits, baselines, and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.build import add_features
from model.evaluate import (
    baseline_champ_points,
    baseline_grid_points,
    race_metrics,
    run_backtest,
)
from model.train import (
    HurdleModels,
    load_checkpoint,
    points_for_position,
    prepare,
    save_checkpoint,
    train_final_model,
    walk_forward_seasons,
)


def _synthetic_df(n_seasons: int = 8, rounds: int = 4, drivers: int = 10) -> pd.DataFrame:
    """Deterministic synthetic history: better grid => better finishing position.

    Finishing positions are a strict permutation of ``1..drivers`` (as in real
    F1 results), correlated with grid via small noise.
    """
    rng = np.random.default_rng(7)
    rows = []
    date_idx = 0
    for season in range(2015, 2015 + n_seasons):
        for round_ in range(1, rounds + 1):
            date_idx += 1
            date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=date_idx * 14)
            grid_order = rng.permutation(drivers)
            grid_pos = np.empty(drivers, dtype=int)  # driver_id -> 0-based grid slot
            for slot, driver in enumerate(grid_order):
                grid_pos[driver] = slot
            noise = rng.normal(0, 1.2, size=drivers)
            # Rank drivers by (grid slot + noise); lower is better. The double
            # argsort yields 0..drivers-1, giving each position exactly once.
            finish_rank = np.argsort(np.argsort(grid_pos + noise))
            for slot, driver in enumerate(grid_order):
                finish = int(finish_rank[driver]) + 1
                points = points_for_position(finish)
                rows.append(
                    {
                        "season": season,
                        "round": round_,
                        "date": date,
                        "circuit_id": f"c{round_}",
                        "driver_id": f"d{driver}",
                        "constructor_id": f"t{driver % 3}",
                        "grid": slot + 1,
                        "position": finish,
                        "points": points,
                        "status": "Finished" if points or rng.random() > 0.3 else "Engine",
                        "sprint_points": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_points_for_position():
    assert points_for_position(1) == 25.0
    assert points_for_position(10) == 1.0
    assert points_for_position(11) == 0.0
    assert points_for_position(None) == 0.0


def test_prepare_converts_categoricals():
    df = add_features(_synthetic_df(n_seasons=4))
    X, y = prepare(df)
    for col in ("driver_id", "constructor_id", "circuit_id", "points_era"):
        assert str(X[col].dtype) == "category"
    assert set(y.columns) == {"points", "scored", "top3", "win"}
    assert y["points"].iloc[0] == df["points"].iloc[0]


def test_hurdle_model_learns_grid_signal():
    df = add_features(_synthetic_df())
    X, y = prepare(df)
    model = HurdleModels(seed=1).fit(X, y)

    pred = model.predict_expected_points(X)
    # Expected points must correlate positively with actual points.
    corr = np.corrcoef(pred, y["points"])[0, 1]
    assert corr > 0.3, f"corr too low: {corr:.3f}"
    # Better grid => higher expected points on average.
    mean_by_grid = pd.Series(pred, index=df.index).groupby(df["grid"]).mean()
    assert mean_by_grid.loc[1] > mean_by_grid.loc[drivers := max(mean_by_grid.index)]
    # Probabilities are sane.
    probs = model.predict_probs(X)
    assert ((probs["p_scored"] >= 0) & (probs["p_scored"] <= 1)).all()
    assert ((probs["p_win"] >= 0) & (probs["p_win"] <= 1)).all()


def test_walk_forward_uses_only_prior_seasons():
    df = add_features(_synthetic_df(n_seasons=6))
    seen = list(walk_forward_seasons(df, min_train_seasons=3))
    assert seen, "no splits produced"
    for train, test, season in seen:
        assert test["season"].eq(season).all()
        assert train["season"].max() < season
    assert seen[0][2] == 2018  # first test season with >=3 training seasons


def test_checkpoint_roundtrip(tmp_path):
    df = add_features(_synthetic_df(n_seasons=4))
    model = train_final_model(df)
    path = tmp_path / "hurdle.joblib"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    X, _ = prepare(df)
    np.testing.assert_allclose(
        model.predict_expected_points(X), loaded.predict_expected_points(X)
    )


def test_race_metrics_perfect_and_reversed():
    df = pd.DataFrame(
        {
            "driver_id": [f"d{i}" for i in range(5)],
            "position": [1, 2, 3, 4, 5],
            "points": [25.0, 18.0, 15.0, 12.0, 10.0],
            "grid": [1, 2, 3, 4, 5],
            "champ_pos_entering": [1, 2, 3, 4, 5],
            "pred_points": [25.0, 18.0, 15.0, 12.0, 10.0],
        }
    )
    perfect = race_metrics(df)
    assert perfect["winner_hit"] == 1.0
    assert perfect["top3_overlap"] == 1.0
    assert perfect["spearman"] == pytest.approx(1.0)
    assert perfect["mae"] == 0.0

    df["pred_points"] = [10.0, 12.0, 15.0, 18.0, 25.0]  # fully reversed
    reversed_ = race_metrics(df)
    assert reversed_["winner_hit"] == 0.0
    assert reversed_["spearman"] < 0.0


def test_run_backtest_produces_tables():
    df = add_features(_synthetic_df(n_seasons=8))
    overall, by_season = run_backtest(df)
    assert set(overall.index) == {"model", "grid", "championship", "zero"}
    assert set(overall.columns) == {"winner_hit", "top3_overlap", "spearman", "mae"}
    assert "model" in by_season
    # The model should at least beat the zero baseline on MAE.
    assert overall.loc["model", "mae"] < overall.loc["zero", "mae"]
    # Per-season tables must hold one row per test season (regression: metrics
    # used to be computed on the pooled season, yielding 0.0/1.0 values).
    n_test_seasons = len(list(walk_forward_seasons(df)))
    assert len(by_season["model"]) == n_test_seasons
    assert len(by_season["grid"]) == n_test_seasons
    winner_hits = by_season["model"]["winner_hit"]
    assert winner_hits.between(0.0, 1.0).all()
    assert (winner_hits > 0.0).any() and (winner_hits < 1.0).any()


def test_baselines():
    df = add_features(_synthetic_df(n_seasons=3))
    grid = baseline_grid_points(df)
    champ = baseline_champ_points(df)
    assert (grid >= 0).all()
    assert (champ >= 0).all()
    # Pole position maps to 25 points.
    assert grid.loc[df["grid"] == 1].iloc[0] == 25.0
