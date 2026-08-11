"""Tests for predict.py: synthetic-row features, ranked output, report."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from helpers import ok_response
from test_model import _synthetic_df

from f1core.predict import _synthetic_rows, format_report, predict_race
from features.build import add_features, assemble
from model.train import train_final_model


def _mini_df() -> pd.DataFrame:
    """Two drivers (a, b), same team, two races in 2020."""
    return pd.DataFrame(
        [
            [2020, 1, "2020-03-01", "c1", "a", "t1", 1, 1, 25.0, "Finished"],
            [2020, 1, "2020-03-01", "c1", "b", "t1", 2, 2, 18.0, "Finished"],
            [2020, 2, "2020-03-08", "c2", "a", "t1", 2, 3, 15.0, "Finished"],
            [2020, 2, "2020-03-08", "c2", "b", "t1", 1, 1, 25.0, "Finished"],
        ],
        columns=["season", "round", "date", "circuit_id", "driver_id",
                 "constructor_id", "grid", "position", "points", "status"],
    )


def _to_season_datas(df: pd.DataFrame) -> list:
    datas = {}
    for (season, round_), g in df.groupby(["season", "round"], sort=False):
        d = datas.setdefault(
            int(season), {"calendar": [], "results": {}, "qualifying": {}, "sprints": {}}
        )
        d["calendar"].append(
            {
                "round": int(round_),
                "date": str(g["date"].iloc[0]),
                "circuit_id": g["circuit_id"].iloc[0],
                "race_name": f"R{int(round_)}",
                "is_sprint_round": False,
            }
        )
        d["results"][int(round_)] = [
            {
                "season": int(row.season), "round": int(row.round),
                "position": row.position, "grid": row.grid,
                "points": float(row.points), "status": row.status,
                "driver_id": row.driver_id, "constructor_id": row.constructor_id,
            }
            for row in g.itertuples()
        ]
    return [datas[s] for s in sorted(datas)]


def test_synthetic_rows_have_pre_race_features_only():
    datas = _to_season_datas(_mini_df())
    rows = _synthetic_rows(
        calendar=[{"season": 2020, "round": 3, "circuit_id": "c1",
                   "race_name": "R3", "date": "2020-03-15", "is_sprint_round": False}],
        target_round=3,
        entries=[("a", "t1"), ("b", "t1")],
        grid_map=None,
    )
    assert len(rows) == 2
    datas[0]["calendar"].append({"round": 3, "date": "2020-03-15", "circuit_id": "c1",
                                 "race_name": "R3", "is_sprint_round": False})
    datas[0]["results"][3] = rows
    df = add_features(assemble(datas))

    syn = df[(df["season"] == 2020) & (df["round"] == 3)]
    assert len(syn) == 2
    assert syn["points"].isna().all()
    assert syn["position"].isna().all()
    assert not syn["scored"].any() and not syn["top3"].any() and not syn["win"].any()

    a = syn[syn["driver_id"] == "a"].iloc[0]
    # Features come strictly from rounds 1-2 (the synthetic row itself is not
    # part of its own rolling history), and grid is unknown (NaN).
    assert a["n_prior_races"] == 2
    assert a["driver_prev_points_mean"] == pytest.approx((25.0 + 15.0) / 2)
    assert a["champ_points_entering"] == pytest.approx(40.0)
    assert np.isnan(a["grid"])


def test_synthetic_rows_accept_grid_override():
    rows = _synthetic_rows(
        calendar=[{"season": 2020, "round": 3, "circuit_id": "c1",
                   "race_name": "R3", "date": "2020-03-15", "is_sprint_round": False}],
        target_round=3,
        entries=[("a", "t1"), ("b", "t1")],
        grid_map={"a": 1, "b": 2},
    )
    by_driver = {r["driver_id"]: r for r in rows}
    assert by_driver["a"]["grid"] == 1
    assert by_driver["b"]["grid"] == 2


def test_predict_race_returns_ranked_table():
    df = add_features(_synthetic_df(n_seasons=8))
    model = train_final_model(df)
    out = predict_race(df, model, 2021, 2)  # a real (non-synthetic) round

    assert list(out.columns) == [
        "pred_rank", "driver_id", "constructor_id", "grid", "expected_points",
        "p_scored", "p_top3", "p_win", "actual_points", "actual_position",
    ]
    assert out["pred_rank"].tolist() == list(range(1, len(out) + 1))
    assert out["expected_points"].is_monotonic_decreasing
    assert ((out["p_win"] >= 0) & (out["p_win"] <= 1)).all()
    assert out["actual_points"].notna().all()  # real race carries actuals


def test_predict_race_unknown_round_raises():
    df = add_features(_synthetic_df(n_seasons=2))
    model = train_final_model(df)
    with pytest.raises(ValueError, match="no rows"):
        predict_race(df, model, 2015, 99)


def test_find_next_race_beyond_cached_seasons(tmp_path):
    import pandas as pd
    from helpers import ok_response

    from f1core.predict import find_next_race
    from f1data import F1Client

    class FakeSession:
        headers: ClassVar[dict[str, str]] = {}

        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append((url, dict(params or {})))
            if url.endswith("/2025.json"):
                payload = {"MRData": {"RaceTable": {"Races": [
                    {"round": str(r), "raceName": f"R{r}"} for r in range(1, 25)]}}}
            elif url.endswith("/2026.json"):
                payload = {"MRData": {"RaceTable": {"Races": [
                    {"round": str(r), "raceName": f"R{r}"} for r in range(1, 5)]}}}
            elif url.endswith("/2026/last/results.json"):
                payload = {"MRData": {"RaceTable": {"Races": [{"round": "2"}]}}}
            else:
                raise AssertionError(f"unexpected URL {url}")
            return ok_response(payload)

    # 2025 season fully raced in the cached dataset; 2026 is found via the API
    # with the latest completed round being 2 -> the next race is round 3.
    df = pd.DataFrame({"season": [2025] * 24, "round": list(range(1, 25))})
    client = F1Client(cache_dir=tmp_path, session=FakeSession(), sleep_seconds=0)
    season, round_ = find_next_race(client, df, [2025])
    assert (season, round_) == (2026, 3)


def test_format_report_includes_verification_when_verified():
    result = pd.DataFrame(
        {
            "pred_rank": [1, 2, 3],
            "driver_id": ["d0", "d1", "d2"],
            "constructor_id": ["t0", "t1", "t0"],
            "grid": [1, 2, 3],
            "expected_points": [25.0, 18.0, 15.0],
            "p_scored": [0.9, 0.8, 0.7],
            "p_top3": [0.5, 0.4, 0.3],
            "p_win": [0.4, 0.2, 0.1],
            "actual_points": [25.0, 18.0, 15.0],
            "actual_position": [1, 2, 3],
        }
    )
    report = format_report(result, 2024, 22,
                           {"race_name": "Las Vegas", "circuit_id": "vegas",
                            "date": "2024-11-23"},
                           verified=True, checkpoint="data/model/hurdle.joblib")
    assert "# Prediction: Las Vegas (2024 Round 22)" in report
    assert "## Actual results" in report
    assert "winner_hit" in report


def _fake_session(routes):
    class FakeSession:
        headers: ClassVar[dict[str, str]] = {}

        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append(url)
            for suffix, payload in routes.items():
                if url.endswith(suffix):
                    return ok_response(payload)
            raise AssertionError(f"unexpected URL {url}")

    return FakeSession


def test_entry_list_uses_last_completed_race_grid(tmp_path):
    from f1core.predict import _entry_list
    from f1data import F1Client

    session = _fake_session(
        {
            "/2026/last/results.json": {
                "MRData": {"RaceTable": {"Races": [{"round": "11"}]}}
            },
            "/2026/11/results.json": {
                "MRData": {"RaceTable": {"Races": [{"Results": [
                    {"Driver": {"driverId": "a"}, "Constructor": {"constructorId": "t1"}},
                    {"Driver": {"driverId": "b"}, "Constructor": {"constructorId": "t2"}},
                ]}]}}
            },
            "/2026/drivers.json": {
                "MRData": {"DriverTable": {"Drivers": [
                    {"driverId": "a"}, {"driverId": "b"}, {"driverId": "c"},
                ]}}
            },
        }
    )
    client = F1Client(cache_dir=tmp_path, session=session(), sleep_seconds=0)
    # "c" missed the last race but holds a cached team -> still on the grid.
    df = pd.DataFrame(
        {"driver_id": ["c"], "constructor_id": ["t3"], "date": ["2026-06-01"]}
    )
    assert _entry_list(client, 2026, df) == [("a", "t1"), ("b", "t2"), ("c", "t3")]


def test_entry_list_falls_back_to_cached_teams(tmp_path):
    from f1core.predict import _entry_list
    from f1data import F1Client

    session = _fake_session(
        {
            # Season with no completed race yet: last/results.json parses to 0,
            # so the season driver list + cached teams are used.
            "/2026/last/results.json": {"MRData": {"RaceTable": {"Races": []}}},
            "/2026/drivers.json": {
                "MRData": {"DriverTable": {"Drivers": [{"driverId": "a"}]}}
            },
        }
    )
    client = F1Client(cache_dir=tmp_path, session=session(), sleep_seconds=0)
    df = pd.DataFrame(
        {"driver_id": ["a"], "constructor_id": ["t9"], "date": ["2025-12-01"]}
    )
    assert _entry_list(client, 2026, df) == [("a", "t9")]


def test_synthetic_rows_rejects_unknown_round():
    with pytest.raises(SystemExit, match="not in the season's calendar"):
        _synthetic_rows(
            calendar=[{"season": 2026, "round": 12}],
            target_round=99,
            entries=[("a", "t1")],
            grid_map=None,
        )


def test_predict_race_applies_calibrators():
    import numpy as np

    from model.calibrate import fit_calibrators

    df = add_features(_synthetic_df(n_seasons=8))
    model = train_final_model(df)
    # A calibrator that compresses every score toward the base rate.
    oos = pd.DataFrame(
        {
            "p_scored": [0.9, 0.1, 0.8, 0.2, 0.95, 0.05],
            "p_top3": [0.9, 0.1, 0.8, 0.2, 0.95, 0.05],
            "p_win": [0.9, 0.1, 0.8, 0.2, 0.95, 0.05],
            "scored": [1, 0, 1, 0, 1, 0],
            "top3": [1, 0, 1, 0, 1, 0],
            "win": [1, 0, 1, 0, 1, 0],
        }
    )
    cal = fit_calibrators(oos)
    raw = predict_race(df, model, 2021, 2)
    cald = predict_race(df, model, 2021, 2, cal)
    assert not np.allclose(raw["p_win"], cald["p_win"])
    assert ((cald["p_scored"] >= 0) & (cald["p_scored"] <= 1)).all()
    assert ((cald["p_win"] >= 0) & (cald["p_win"] <= 1)).all()


def test_format_report_labels_calibration_status():
    result = pd.DataFrame(
        {
            "pred_rank": [1, 2],
            "driver_id": ["d0", "d1"],
            "constructor_id": ["t0", "t1"],
            "grid": [1, 2],
            "expected_points": [25.0, 18.0],
            "p_scored": [0.9, 0.8],
            "p_top3": [0.5, 0.4],
            "p_win": [0.4, 0.2],
            "actual_points": [25.0, 18.0],
            "actual_position": [1, 2],
        }
    )
    meta = {"race_name": "X", "circuit_id": "c", "date": "2026-01-01"}
    cal_report = format_report(result, 2026, 1, meta, verified=False,
                               checkpoint="c", calibrated=True)
    raw_report = format_report(result, 2026, 1, meta, verified=False,
                               checkpoint="c", calibrated=False)
    assert "isotonic-calibrated model scores" in cal_report
    assert "raw model scores" in raw_report
    assert "isotonic-calibrated" not in raw_report


def test_rank_expected_pit_lane_last():
    """Ties in expected points: pit-lane (grid=0) starts rank last."""
    from f1core.predict import _rank_expected

    out = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c"],
            "grid": [1, 0, 3],
            "expected_points": [10.0, 10.0, 10.0],
        }
    )
    ranked = _rank_expected(out)
    assert ranked["pred_rank"].tolist() == [1, 2, 3]
    assert ranked["driver_id"].tolist() == ["a", "c", "b"]  # pit lane last


def test_predict_race_ranking_agrees_with_backtest_rank_by():
    """predict_race's ordering must match the shared rank_by on the same
    quantized predictions (regression: grid=0 tiebreak diverged)."""
    from f1core.reporting import rank_by

    df = add_features(_synthetic_df(n_seasons=8))
    model = train_final_model(df)
    out = predict_race(df, model, 2021, 2)

    v = out.rename(
        columns={"actual_points": "points", "actual_position": "position"}
    ).copy()
    v["pred_points"] = v["expected_points"]
    ranks = rank_by(v, "pred_points", "grid").tolist()
    assert ranks == list(range(1, len(out) + 1))


def test_main_clean_error_without_round(monkeypatch):
    """predict.main() turns get_prediction's ValueError into a clean SystemExit."""
    import sys

    import f1core.predict as predict_module

    def boom(**kw):
        raise ValueError("round_ is required when season is given")

    monkeypatch.setattr(predict_module, "get_prediction", boom)
    monkeypatch.setattr(sys, "argv", ["predict.py", "--season", "2024"])
    with pytest.raises(SystemExit) as exc:
        predict_module.main()
    assert "round_ is required" in str(exc.value)


def test_get_prediction_requires_season_for_round():
    from f1core.predict import get_prediction

    with pytest.raises(ValueError, match="season is required when round is given"):
        get_prediction(round_=22, quiet=True)


# --------------------------------------------------------------------------
# Disk-backed prediction cache
# --------------------------------------------------------------------------

def test_prediction_cache_key_distinguishes_season_round_fingerprint_params():
    from f1core.predict import prediction_cache_key

    k = prediction_cache_key(2024, 1, "fpA", "pA")
    assert prediction_cache_key(2024, 1, "fpA", "pA") == k  # deterministic
    distinct = {
        prediction_cache_key(2024, 1, "fpA", "pA"),  # base
        prediction_cache_key(2025, 1, "fpA", "pA"),  # different season
        prediction_cache_key(2024, 2, "fpA", "pA"),  # different round
        prediction_cache_key(2024, 1, "fpB", "pA"),  # different feature fingerprint
        prediction_cache_key(2024, 1, "fpA", "pB"),  # different params hash
    }
    assert len(distinct) == 5


def test_prediction_cache_roundtrip_rebuilds_result(tmp_path):
    import json

    from f1core.predict import (
        _pred_from_payload,
        load_cached_prediction,
        prediction_cache_key,
        prediction_payload,
        save_cached_prediction,
    )

    result = pd.DataFrame(
        {
            "pred_rank": [1, 2], "driver_id": ["a", "b"],
            "constructor_id": ["t1", "t2"], "grid": [1, 2],
            "expected_points": [25.0, 18.0], "p_scored": [0.9, 0.8],
            "p_top3": [0.6, 0.5], "p_win": [0.4, 0.3],
            "actual_points": [25.0, 18.0], "actual_position": [1, 2],
        }
    )
    pred = {
        "result": result,
        "meta": {"race_name": "R1", "circuit_id": "c1", "date": pd.Timestamp("2024-03-01")},
        "season": 2024, "round": 1, "synthetic": False, "verified": True,
        "calibrated": False, "checkpoint": "c", "features": ["grid"],
    }
    key = prediction_cache_key(2024, 1, "fp", "ph")
    payload = prediction_payload(pred)
    # The cache writer uses raw json.dumps (not FastAPI's encoder), so the
    # payload must be plain-JSON even when meta carries a Timestamp date.
    json.dumps(payload)
    save_cached_prediction(tmp_path, key, payload)

    cached = load_cached_prediction(tmp_path, key)
    assert cached is not None
    assert cached["season"] == 2024 and cached["round"] == 1
    assert cached["race"]["date"] == "2024-03-01T00:00:00"
    assert cached["drivers"][0]["driver_id"] == "a"

    rebuilt = _pred_from_payload(cached)
    pd.testing.assert_frame_equal(rebuilt["result"], result)
    assert rebuilt["meta"] == {"race_name": "R1", "circuit_id": "c1", "date": "2024-03-01T00:00:00"}
    assert rebuilt["verified"] is True


def test_prediction_cache_miss_and_fingerprint_invalidation(tmp_path):
    from f1core.predict import load_cached_prediction, prediction_cache_key

    save_key = prediction_cache_key(2024, 1, "fpA", "ph")
    # A different round (miss), or a changed feature fingerprint (invalidation),
    # must not read the entry written for save_key.
    assert load_cached_prediction(tmp_path, save_key) is None
    assert load_cached_prediction(
        tmp_path, prediction_cache_key(2024, 1, "fpB", "ph")
    ) is None
    assert load_cached_prediction(
        tmp_path, prediction_cache_key(2024, 1, "fpA", "ph2")
    ) is None


def test_get_prediction_disk_cache_skips_recompute(monkeypatch, tmp_path):
    """A cache hit for an explicit (season, round) skips dataset assembly + scoring."""
    import f1core.predict as p
    from f1core.config import load_config

    calls = {"assembly": 0, "predict": 0}
    base = pd.DataFrame(
        {
            "season": [2024, 2024], "round": [1, 1],
            "race_name": ["R1", "R1"], "circuit_id": ["c1", "c1"],
            "date": ["2024-03-01", "2024-03-01"],
            "driver_id": ["a", "b"], "constructor_id": ["t1", "t2"],
            "grid": [1, 2], "position": [1, 2], "points": [25.0, 18.0],
            "scored": [1, 1], "top3": [1, 1], "win": [1, 0],
        }
    )
    result = pd.DataFrame(
        {
            "pred_rank": [1, 2], "driver_id": ["a", "b"],
            "constructor_id": ["t1", "t2"], "grid": [1, 2],
            "expected_points": [25.0, 18.0], "p_scored": [0.9, 0.8],
            "p_top3": [0.6, 0.5], "p_win": [0.4, 0.3],
            "actual_points": [25.0, 18.0], "actual_position": [1, 2],
        }
    )

    def fake_frame(client, seasons):
        calls["assembly"] += 1
        return [], base

    def fake_target(client, base_df, season_datas, seasons, ts, tr, grid_csv, quiet):
        return base_df, False

    def fake_models(cfg, model_path, feats):
        return "checkpoint", object(), {}

    def fake_predict(df_, model, season, round_, cal, feats):
        calls["predict"] += 1
        return result

    monkeypatch.setattr(p, "_featured_frame", fake_frame)
    monkeypatch.setattr(p, "_target_frame", fake_target)
    monkeypatch.setattr(p, "_load_models", fake_models)
    monkeypatch.setattr(p, "predict_race", fake_predict)

    cfg = load_config()
    p.get_prediction(season=2024, round_=1, cfg=cfg, cache_dir=tmp_path)
    p.get_prediction(season=2024, round_=1, cfg=cfg, cache_dir=tmp_path)
    # The dataset is assembled and scored exactly once across both calls.
    assert calls == {"assembly": 1, "predict": 1}


def test_get_prediction_cache_invalidates_on_feature_change(monkeypatch, tmp_path):
    """Changing the feature selection produces a different cache key -> recompute."""
    import copy

    import f1core.predict as p
    from f1core.config import load_config

    calls = {"assembly": 0}
    base = pd.DataFrame(
        {
            "season": [2024, 2024], "round": [1, 1],
            "race_name": ["R1", "R1"], "circuit_id": ["c1", "c1"],
            "date": ["2024-03-01", "2024-03-01"],
            "driver_id": ["a", "b"], "constructor_id": ["t1", "t2"],
            "grid": [1, 2], "position": [1, 2], "points": [25.0, 18.0],
            "scored": [1, 1], "top3": [1, 1], "win": [1, 0],
        }
    )
    result = base[["driver_id", "grid"]].copy()

    def fake_frame(client, seasons):
        calls["assembly"] += 1
        return [], base

    monkeypatch.setattr(p, "_featured_frame", fake_frame)
    monkeypatch.setattr(p, "_target_frame",
                        lambda client, bdf, sd, ss, ts, tr, gc, q: (bdf, False))
    monkeypatch.setattr(p, "_load_models",
                        lambda cfg, mp, feats: ("checkpoint", object(), {}))
    monkeypatch.setattr(p, "predict_race",
                        lambda df_, m, s, r, c, f: result.assign(
                            pred_rank=range(1, len(result) + 1)))

    cfg = load_config()
    p.get_prediction(season=2024, round_=1, cfg=cfg, cache_dir=tmp_path)
    assert calls["assembly"] == 1
    # A different feature set (core defaults vs an explicit single feature)
    # must not reuse the cached payload.
    cfg2 = copy.deepcopy(cfg)
    cfg2["features"] = {"enabled": ["grid"]}
    p.get_prediction(season=2024, round_=1, cfg=cfg2, cache_dir=tmp_path)
    assert calls["assembly"] == 2
