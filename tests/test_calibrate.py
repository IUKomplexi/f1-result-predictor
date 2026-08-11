"""Tests for model/calibrate.py: OOS collection, isotonic calibration, Brier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_model import _synthetic_df

from model.calibrate import (
    TARGETS,
    apply_calibration,
    brier,
    collect_oos_scores,
    fit_calibrators,
    load_calibrators,
    reliability_table,
    save_calibrators,
)


def test_brier_and_reliability_table():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.75, 0.25, 0.75, 0.25])
    assert brier(y, p) == pytest.approx(((0.25) ** 2 * 4) / 4)
    table = reliability_table(y, p, bins=4)
    assert {"mean_pred", "observed", "n"} <= set(table.columns)


def test_fit_calibrators_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.5).astype(float)
    # Overconfident raw scores: 0.92 for actual positives, 0.08 for negatives
    # (the true rate is 0.5), so isotonic should pull them toward 0.5.
    raw = np.where(y == 1, 0.92, 0.08) + rng.normal(0, 0.005, n)
    oos = pd.DataFrame(
        {
            "p_scored": raw, "scored": y.astype(int),
            "p_top3": raw, "top3": y.astype(int),
            "p_win": raw, "win": y.astype(int),
            "season": np.full(n, 2015),
        }
    )
    cal = fit_calibrators(oos)
    cal_scored = cal["scored"].predict(raw)
    assert brier(y, cal_scored) < brier(y, raw)
    assert abs(float(np.mean(cal_scored)) - 0.5) < 0.05


def test_apply_calibration_maps_all_targets():
    rng = np.random.default_rng(1)
    raw = {"p_scored": rng.random(5), "p_top3": rng.random(5), "p_win": rng.random(5)}
    oos = pd.DataFrame(
        {
            **raw,
            "scored": np.ones(5, dtype=int), "top3": np.zeros(5, dtype=int),
            "win": np.zeros(5, dtype=int),
        }
    )
    cal = fit_calibrators(oos)
    out = apply_calibration(raw, cal)
    assert set(out) == set(raw)
    for key in raw:
        assert ((out[key] >= 0) & (out[key] <= 1)).all()
    # Applying no calibrators leaves scores untouched.
    assert apply_calibration(raw, {}) == raw


def test_collect_oos_scores_walk_forward():
    from features.build import add_features

    df = add_features(_synthetic_df(n_seasons=6))
    oos = collect_oos_scores(df)
    # Only test seasons (train-seasons are never scored out-of-sample).
    test_seasons = sorted(df["season"].unique())[3:]
    assert sorted(oos["season"].unique()) == test_seasons
    assert set(TARGETS) <= set(oos.columns)
    for col in ("p_scored", "p_top3", "p_win"):
        assert ((oos[col] >= 0) & (oos[col] <= 1)).all()
    # Labels match the source data.
    idx = oos.index[0]
    assert oos.loc[idx, "scored"] == df.loc[idx, "scored"]


def test_calibrator_roundtrip_and_missing(tmp_path):
    oos = pd.DataFrame(
        {
            "p_scored": [0.9, 0.1, 0.8, 0.2],
            "p_top3": [0.9, 0.1, 0.8, 0.2],
            "p_win": [0.9, 0.1, 0.8, 0.2],
            "scored": [1, 0, 1, 0], "top3": [1, 0, 1, 0], "win": [1, 0, 1, 0],
        }
    )
    cal = fit_calibrators(oos)
    path = tmp_path / "cal.joblib"
    save_calibrators(cal, path)
    loaded = load_calibrators(path)
    assert set(loaded) == set(cal)
    assert load_calibrators(tmp_path / "missing.joblib") is None

    with pytest.raises(ValueError, match="calibrator file"):
        bad = tmp_path / "bad.joblib"
        import joblib

        joblib.dump({"other": 1}, bad)
        load_calibrators(bad)


def test_calibrators_keyed_to_feature_fingerprint(tmp_path):
    """A calibrator file fit on a different feature set must be rejected."""
    from features.registry import all_feature_ids

    oos = pd.DataFrame(
        {
            "p_scored": [0.9, 0.1, 0.8, 0.2],
            "p_top3": [0.9, 0.1, 0.8, 0.2],
            "p_win": [0.9, 0.1, 0.8, 0.2],
            "scored": [1, 0, 1, 0], "top3": [1, 0, 1, 0], "win": [1, 0, 1, 0],
        }
    )
    cal = fit_calibrators(oos)
    full = all_feature_ids()
    subset = full[:-1]
    path = tmp_path / "cal.joblib"
    save_calibrators(cal, path, features=full)
    assert load_calibrators(path, expected=full) is not None
    with pytest.raises(ValueError, match="calibrator feature set does not match"):
        load_calibrators(path, expected=subset)
    # Legacy files without a fingerprint load unchecked (grandfathered).
    import joblib

    legacy = tmp_path / "legacy.joblib"
    joblib.dump({"calibrators": cal}, legacy)
    assert load_calibrators(legacy, expected=subset) is not None
