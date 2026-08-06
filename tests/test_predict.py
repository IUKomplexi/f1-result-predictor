"""Tests for predict.py: synthetic-row features, ranked output, report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.build import add_features, assemble
from helpers import ok_response
from model.train import train_final_model
from predict import _synthetic_rows, format_report, predict_race

from test_model import _synthetic_df


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

    from f1data import F1Client
    from helpers import ok_response
    from predict import find_next_race

    class FakeSession:
        headers = {}

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
        headers = {}

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
    from f1data import F1Client
    from predict import _entry_list

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
    from f1data import F1Client
    from predict import _entry_list

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
