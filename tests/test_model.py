"""Tests for the hurdle model, walk-forward splits, baselines, and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1core.predict import predict_race
from features.build import add_features
from model.evaluate import (
    baseline_champ_points,
    baseline_grid_points,
    race_metrics,
    run_backtest,
)
from model.train import (
    FEATURES,
    HurdleModels,
    checkpoint_meta,
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
    assert mean_by_grid.loc[1] > mean_by_grid.loc[max(mean_by_grid.index)]
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
    assert set(overall.columns) == {"winner_hit", "top3_overlap", "top10_overlap", "spearman", "mae"}
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


def test_race_metrics_missing_winner_does_not_crash():
    df = pd.DataFrame(
        {
            "driver_id": ["d0", "d1"],
            "position": [2, 3],  # nobody finished P1 (partial data)
            "points": [18.0, 15.0],
            "grid": [1, 2],
            "pred_points": [10.0, 8.0],
        }
    )
    m = race_metrics(df)
    assert m["winner_hit"] == 0.0
    assert 0.0 <= m["top3_overlap"] <= 1.0


def test_load_checkpoint_rejects_mismatched_features(tmp_path):
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"models": None, "features": ["old_feature"]}, path)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path)


def test_checkpoint_meta_roundtrips_season_range(tmp_path):
    """The training window is stored with the checkpoint and readable back."""
    df = add_features(_synthetic_df(n_seasons=4))
    feats = FEATURES
    model = train_final_model(df, feats)
    path = tmp_path / "hurdle-2015-2018.joblib"
    save_checkpoint(model, path, features=feats, season_range=(2015, 2018))
    meta = checkpoint_meta(path)
    assert meta["season_range"] == [2015, 2018]
    assert meta["features"] == feats
    assert "fingerprint" in meta
    # Legacy checkpoints (no season_range) degrade gracefully.
    legacy = tmp_path / "legacy.joblib"
    save_checkpoint(model, legacy, features=feats)
    assert "season_range" not in checkpoint_meta(legacy)
    assert checkpoint_meta(tmp_path / "missing.joblib") is None


def test_quantize_points_rounds_to_table_values():
    from model.train import QUANTIZED_POINTS, quantize_points

    out = quantize_points(np.array([0.1, 0.4, 2.3, 9.5, 24.0, 25.5, 25.9]))
    expected = [0.0, 0.0, 2.0, 10.0, 25.0, 25.0, 26.0]
    np.testing.assert_allclose(out, expected)
    # All outputs are points-table values.
    assert set(np.unique(out)) <= set(QUANTIZED_POINTS)


def test_run_backtest_quantize_option():
    df = add_features(_synthetic_df(n_seasons=8))
    overall_q, _ = run_backtest(df, quantize=True)
    assert overall_q.loc["model", "mae"] < overall_q.loc["zero", "mae"]


def test_run_backtest_deployed_checkpoint_scores_all_seasons():
    """model= scores every season with the fixed checkpoint (no walk-forward)."""
    df = add_features(_synthetic_df(n_seasons=8))
    model = train_final_model(df)
    overall, by_season = run_backtest(df, model=model)
    assert set(overall.index) == {"model", "grid", "championship", "zero"}
    assert set(overall.columns) == {"winner_hit", "top3_overlap", "top10_overlap", "spearman", "mae"}
    assert overall.loc["model", "mae"] < overall.loc["zero", "mae"]
    # The fixed model applies to every season — unlike the walk-forward mode,
    # which needs min_train_seasons prior seasons before the first test.
    assert len(by_season["model"]) == 8


def test_run_model_paths_compares_checkpoints(tmp_path, monkeypatch):
    """evaluate.run(model_paths=[...]) scores several checkpoints on one shared
    dataset and writes a snapshot with a per-model 'models' key."""
    import json

    import model.evaluate as evaluate_module

    df = add_features(_synthetic_df(n_seasons=8))
    early = df[df["season"] <= 2018]
    path_a = tmp_path / "alpha.joblib"
    path_b = tmp_path / "beta.joblib"
    save_checkpoint(train_final_model(df), path_a)
    save_checkpoint(train_final_model(early), path_b)

    # Offline: serve the synthetic frame instead of touching the raw cache.
    monkeypatch.setattr(evaluate_module, "build_dataset", lambda *a, **kw: df)
    result = evaluate_module.run(
        start=2015, end=2022, cache_dir=str(tmp_path / "raw"),
        dataset=str(tmp_path / "features.parquet"),
        out=str(tmp_path / "backtest.md"),
        out_json=str(tmp_path / "backtest.json"),
        model_paths=[str(path_a), str(path_b)],
    )

    assert result["models"] == ["alpha", "beta"]
    snap = result["snapshot"]
    assert set(snap["models"]) == {"alpha", "beta"}
    for name in ("alpha", "beta"):
        entry = snap["models"][name]
        assert set(entry["overall"]) == {"model", "grid", "championship", "zero"}
        assert {"winner_hit", "top3_overlap", "top10_overlap", "spearman", "mae"} <= set(
            entry["overall"]["model"]
        )
        assert "model" in entry["by_season"]
    # The primary tables come from the first compared checkpoint.
    assert result["checkpoint"] == str(path_a)
    # The snapshot on disk carries the models key too.
    on_disk = json.loads((tmp_path / "backtest.json").read_text(encoding="utf-8"))
    assert set(on_disk["models"]) == {"alpha", "beta"}
    assert on_disk["overall"]["model"]["mae"] == snap["models"]["alpha"]["overall"]["model"]["mae"]


def test_update_model_index_roundtrip(tmp_path):
    import json

    from model.train import update_model_index

    named = tmp_path / "data" / "model" / "hurdle-2022-2026-abc.joblib"
    other = tmp_path / "data" / "model" / "experiment.joblib"
    index_path = update_model_index(named, {"checkpoint": str(named), "rows": 100})
    assert index_path == tmp_path / "data" / "model" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert "hurdle-2022-2026-abc" in index
    assert index["hurdle-2022-2026-abc"]["rows"] == 100

    # A new name is added; re-training an existing name overwrites its entry.
    update_model_index(other, {"rows": 2})
    update_model_index(named, {"checkpoint": str(named), "rows": 101})
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["hurdle-2022-2026-abc"]["rows"] == 101
    assert index["experiment"]["rows"] == 2
    assert len(index) == 2


def test_quantize_points_nan_maps_to_zero():
    from model.train import quantize_points

    np.testing.assert_allclose(quantize_points(np.array([np.nan])), [0.0])


def test_run_backtest_no_quantize_differs():
    df = add_features(_synthetic_df(n_seasons=8))
    overall_c, _ = run_backtest(df, quantize=True)
    overall_r, _ = run_backtest(df, quantize=False)
    assert overall_c.loc["model", "mae"] != overall_r.loc["model", "mae"]
    assert overall_r.loc["model", "mae"] < overall_r.loc["zero", "mae"]


def test_model_drops_constant_numeric_columns():
    """Constant numeric columns are dropped at fit (HGB binning crashes on them).

    Regression: training on a pre-sprint season range (e.g. 2010-2015) makes
    ``is_sprint_round``/``team_switch`` constant, which used to crash
    HistGradientBoosting with a sliding_window_view error. The drop is
    recorded on the model so prediction drops the same columns.
    """
    df = add_features(_synthetic_df(n_seasons=4))
    df["is_sprint_round"] = 0.0  # force a constant numeric column

    model = train_final_model(df)  # used to crash during binning
    assert model is not None
    assert "is_sprint_round" in model.column_drop_
    assert "grid" not in model.column_drop_

    # The same model predicts on an untouched frame (column present again).
    last = df["season"].max()
    last_round = df.loc[df["season"] == last, "round"].max()
    result = predict_race(df, model, last, last_round)
    assert result["expected_points"].notna().all()
