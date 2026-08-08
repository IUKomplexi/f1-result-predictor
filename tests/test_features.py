"""Tests for feature engineering: leakage safety, assembly, caching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import QueueSession, ok_response

from features.build import (
    NUMERIC_FEATURES,
    add_features,
    assemble,
    build_dataset,
    coverage_report,
)


def _mini_df() -> pd.DataFrame:
    """Two drivers (a, b), same team, three races."""
    return pd.DataFrame(
        [
            # season round date circuit driver team grid pos points status sprint_points
            [2020, 1, "2020-03-01", "c1", "a", "t1", 1, 1, 25.0, "Finished", 0.0],
            [2020, 1, "2020-03-01", "c1", "b", "t1", 2, 2, 18.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "a", "t1", 2, 3, 15.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "b", "t1", 1, 1, 25.0, "Finished", 0.0],
            [2020, 3, "2020-03-15", "c1", "a", "t1", 1, 1, 25.0, "Finished", 0.0],
            [2020, 3, "2020-03-15", "c1", "b", "t1", 2, 2, 18.0, "Finished", 0.0],
        ],
        columns=[
            "season", "round", "date", "circuit_id", "driver_id",
            "constructor_id", "grid", "position", "points", "status", "sprint_points",
        ],
    )


def _row_for(df: pd.DataFrame, driver: str, round_: int) -> pd.Series:
    return df[(df["driver_id"] == driver) & (df["round"] == round_)].iloc[0]


def test_no_leakage_rolling_features_use_prior_races_only():
    out = add_features(_mini_df())

    # Driver "a" at round 3: rolling features only from rounds 1-2.
    row = _row_for(out, "a", 3)
    assert row["n_prior_races"] == 2
    assert row["driver_prev_points_mean"] == pytest.approx((25.0 + 15.0) / 2)
    assert row["driver_prev_finish_mean"] == pytest.approx((1 + 3) / 2)
    assert row["champ_points_entering"] == pytest.approx(40.0)
    assert row["champ_pos_entering"] == 2  # b enters round 3 with 43 pts
    # Team form uses the team's prior races (43 = a25+b18, 40 = a15+b25).
    assert row["team_prev_points_mean"] == pytest.approx((43.0 + 40.0) / 2)

    # Rookie at season start: no history.
    first = _row_for(out, "b", 1)
    assert first["n_prior_races"] == 0
    assert np.isnan(first["driver_prev_points_mean"])
    assert first["champ_points_entering"] == 0.0


def test_no_leakage_future_changes_do_not_affect_features():
    base = _mini_df()
    out1 = add_features(base)
    altered = base.copy()
    altered.loc[altered["round"] == 3, "points"] = 0.0  # change the race itself
    out2 = add_features(altered)

    # Features for the round-3 rows must not depend on round-3 outcomes.
    for driver in ("a", "b"):
        r1 = _row_for(out1, driver, 3)
        r2 = _row_for(out2, driver, 3)
        for col in NUMERIC_FEATURES:
            a, b = r1[col], r2[col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == b, f"{driver} round3 {col}: {a} != {b}"


def test_champ_points_entering_includes_current_round_sprint():
    df = _mini_df()
    # Round 3 becomes a sprint round: drivers earned sprint points on Saturday.
    df.loc[df["round"] == 3, "sprint_points"] = [4.0, 5.0]
    out = add_features(df)

    a3 = _row_for(out, "a", 3)
    # Entering points = rounds 1-2 main points (25+15) + round-3 sprint (4).
    assert a3["champ_points_entering"] == pytest.approx(40.0 + 4.0)
    # Round-3 sprint points are NOT part of the main-race points target.
    assert a3["points"] == 25.0
    b3 = _row_for(out, "b", 3)
    assert b3["champ_points_entering"] == pytest.approx(18.0 + 25.0 + 5.0)


def test_targets_and_era():
    out = add_features(_mini_df())
    a3 = _row_for(out, "a", 3)
    assert bool(a3["scored"]) is True
    assert bool(a3["top3"]) is True
    assert bool(a3["win"]) is True
    b1 = _row_for(out, "b", 1)
    assert bool(b1["scored"]) is True and bool(b1["top3"]) is True and bool(b1["win"]) is False
    assert (out["points_era"] == "post2019").all()


def test_assemble_merges_calendar_qualifying_sprint():
    season_datas = [
        {
            "calendar": [
                {"round": 1, "date": "2020-03-01", "circuit_id": "c1",
                 "race_name": "Race 1", "is_sprint_round": True}
            ],
            "results": {
                1: [{"season": 2020, "round": 1, "position": 1, "grid": 2,
                     "points": 25.0, "status": "Finished", "driver_id": "a",
                     "constructor_id": "t"}]
            },
            "qualifying": {
                1: [{"season": 2020, "round": 1, "position": 3, "driver_id": "a",
                     "constructor_id": "t"}]
            },
            "sprints": {
                1: [{"season": 2020, "round": 1, "position": 2, "grid": 4,
                     "points": 7.0, "status": "Finished", "driver_id": "a",
                     "constructor_id": "t"}]
            },
        }
    ]
    df = assemble(season_datas)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["qual_pos"] == 3
    assert row["sprint_points"] == 7.0
    assert row["circuit_id"] == "c1"
    assert bool(row["is_sprint_round"])
    assert row["race_name"] == "Race 1"


def test_assemble_defaults_missing_qualifying_to_grid():
    season_datas = [
        {
            "calendar": [
                {"round": 1, "date": "2020-03-01", "circuit_id": "c1",
                 "race_name": "Race 1", "is_sprint_round": False}
            ],
            "results": {
                1: [{"season": 2020, "round": 1, "position": 5, "grid": 7,
                     "points": 0.0, "status": "Finished", "driver_id": "a",
                     "constructor_id": "t"}]
            },
            "qualifying": {},
            "sprints": {},
        }
    ]
    df = assemble(season_datas)
    assert df.iloc[0]["qual_pos"] == 7  # falls back to grid
    assert df.iloc[0]["sprint_points"] == 0.0


def test_build_dataset_caches(tmp_path):
    session = QueueSession()
    mini_cal = {
        "MRData": {
            "limit": "30", "offset": "0", "total": "1",
            "RaceTable": {
                "season": "2020", "Races": [
                    {"season": "2020", "round": "1", "raceName": "Race 1",
                     "Circuit": {"circuitId": "c1", "circuitName": "C1",
                                 "Location": {"country": "X"}},
                     "date": "2020-03-01", "time": "13:00:00Z"},
                ],
            },
        }
    }
    results = {
        "MRData": {
            "limit": "30", "offset": "0", "total": "1",
            "RaceTable": {
                "season": "2020", "round": "1", "Races": [
                    {"season": "2020", "round": "1", "raceName": "Race 1",
                     "Circuit": {"circuitId": "c1"}, "date": "2020-03-01",
                     "Results": [
                         {"position": "1", "grid": "1", "points": "25",
                          "laps": "50", "status": "Finished",
                          "Driver": {"driverId": "a"},
                          "Constructor": {"constructorId": "t"}},
                     ]},
                ],
            },
        }
    }
    empty_qual = {
        "MRData": {
            "limit": "30", "offset": "0", "total": "1",
            "RaceTable": {
                "season": "2020", "round": "1", "Races": [
                    {"season": "2020", "round": "1", "raceName": "Race 1",
                     "Circuit": {"circuitId": "c1"}, "date": "2020-03-01",
                     "QualifyingResults": []},
                ],
            },
        }
    }

    def route(url, params=None, timeout=None):
        session.calls.append((url, dict(params or {})))
        if url.endswith("/2020.json"):
            return ok_response(mini_cal)
        if url.endswith("/2020/results.json"):
            return ok_response(results)
        if url.endswith("/2020/qualifying.json"):
            return ok_response(empty_qual)
        raise AssertionError(f"No route for {url}")

    session.get = route  # type: ignore[method-assign]
    from f1data import F1Client

    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)
    cache_path = tmp_path / "feat.parquet"

    df1 = build_dataset(client, [2020], cache_path=cache_path)
    n_calls = len(session.calls)
    assert n_calls == 3  # calendar + results + qualifying

    df2 = build_dataset(client, [2020], cache_path=cache_path)
    assert len(session.calls) == n_calls  # served from parquet, no network
    pd.testing.assert_frame_equal(df1, df2)
    assert "champ_points_entering" in df1.columns
    assert df1["points"].iloc[0] == 25.0


def test_coverage_report_shape():
    out = add_features(_mini_df())
    report = coverage_report(out)
    assert list(report["season"]) == [2020]
    assert report.loc[0, "starts"] == 6
    assert report.loc[0, "scored_rate"] == 1.0


def test_team_switch_tenure_and_lag_features():
    """A driver switching teams mid-season: switch flag, pairing tenure, lag."""
    df = pd.DataFrame(
        [
            # season round date circuit driver team grid qual pos points status sprint_points
            [2020, 1, "2020-03-01", "c1", "x", "t1", 1, 1, 1, 25.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "x", "t2", 2, 4, 2, 18.0, "Finished", 0.0],
            [2020, 3, "2020-03-15", "c1", "x", "t2", 1, 1, 1, 25.0, "Finished", 0.0],
        ],
        columns=[
            "season", "round", "date", "circuit_id", "driver_id",
            "constructor_id", "grid", "qual_pos", "position", "points",
            "status", "sprint_points",
        ],
    )
    out = add_features(df)
    r2 = out[out["round"] == 2].iloc[0]
    assert r2["team_switch"] == 1.0       # moved from t1 to t2
    assert r2["team_tenure"] == 0         # first race with t2
    assert r2["last_race_points"] == 25.0  # points in round 1
    assert r2["grid_qual_gap"] == -2       # qualified 4th, started 2nd
    r3 = out[out["round"] == 3].iloc[0]
    assert r3["team_switch"] == 0.0
    assert r3["team_tenure"] == 1         # one prior race with t2
    assert r3["last_race_points"] == 18.0
    assert r3["circuit_prev_points_mean"] == pytest.approx(25.0)  # at c1, round 1
    assert out["is_sprint_round"].eq(0.0).all()  # default when no calendar




def test_constructor_champ_position_two_teams():
    df = pd.DataFrame(
        [
            # season round date circuit driver team grid pos points status sprint_points
            [2020, 1, "2020-03-01", "c1", "a", "t1", 1, 1, 25.0, "Finished", 0.0],
            [2020, 1, "2020-03-01", "c1", "c", "t2", 2, 2, 18.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "a", "t1", 2, 2, 18.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "c", "t2", 1, 1, 25.0, "Finished", 0.0],
        ],
        columns=[
            "season", "round", "date", "circuit_id", "driver_id",
            "constructor_id", "grid", "position", "points", "status", "sprint_points",
        ],
    )
    out = add_features(df)
    # Round 2 entering: t1 has 25 pts, t2 has 18 -> t1 leads.
    a2 = out[(out["driver_id"] == "a") & (out["round"] == 2)].iloc[0]
    c2 = out[(out["driver_id"] == "c") & (out["round"] == 2)].iloc[0]
    assert a2["constructor_champ_pos_entering"] == 1
    assert c2["constructor_champ_pos_entering"] == 2


def test_build_dataset_invalidates_stale_feature_cache(tmp_path, monkeypatch):
    """A cached parquet missing a currently-defined feature must be rebuilt."""
    from features import build as fb

    mini = {
        "calendar": [{"round": 1, "date": "2020-03-01", "circuit_id": "c1",
                      "race_name": "R1", "is_sprint_round": False}],
        "results": {1: [{"season": 2020, "round": 1, "position": 1, "grid": 1,
                         "points": 25.0, "status": "Finished", "driver_id": "a",
                         "constructor_id": "t1"}]},
        "qualifying": {}, "sprints": {},
    }
    monkeypatch.setattr(fb, "fetch_season", lambda client, s: mini)
    cache = tmp_path / "feat.parquet"

    df1 = fb.build_dataset(object(), [2020], cache_path=cache)
    assert "team_tenure" in df1.columns
    # Simulate a stale cache from an older feature set.
    df1.drop(columns=["team_tenure"]).to_parquet(cache, index=False)

    df2 = fb.build_dataset(object(), [2020], cache_path=cache)
    assert "team_tenure" in df2.columns  # rebuilt, not silently loaded stale


def test_build_dataset_invalidates_stale_season_cache(tmp_path, monkeypatch):
    """A cache built for fewer seasons than requested must be rebuilt."""
    from features import build as fb

    mini = {
        "calendar": [{"round": 1, "date": "2020-03-01", "circuit_id": "c1",
                      "race_name": "R1", "is_sprint_round": False}],
        "results": {1: [{"season": 2020, "round": 1, "position": 1, "grid": 1,
                         "points": 25.0, "status": "Finished", "driver_id": "a",
                         "constructor_id": "t1"}]},
        "qualifying": {}, "sprints": {},
    }
    seen = []
    monkeypatch.setattr(fb, "fetch_season",
                        lambda client, s: (seen.append(s), mini)[1])
    cache = tmp_path / "feat.parquet"

    fb.build_dataset(object(), [2020], cache_path=cache)
    assert seen == [2020]
    # Rebuild request for a wider range must NOT be served from the 2020 cache.
    fb.build_dataset(object(), [2020, 2021], cache_path=cache)
    assert seen == [2020, 2020, 2021]  # refetched both seasons


def test_teammate_gap_features():
    """Rolling gap vs teammate: negative = ahead; strictly prior (NaN first race)."""
    df = pd.DataFrame(
        [
            # season round date circuit driver team grid qual pos points status sprint_points
            [2020, 1, "2020-03-01", "c1", "a", "t1", 1, 1, 1, 25.0, "Finished", 0.0],
            [2020, 1, "2020-03-01", "c1", "b", "t1", 3, 4, 5, 6.0, "Finished", 0.0],
            [2020, 1, "2020-03-01", "c1", "c", "t2", 2, 2, 2, 18.0, "Finished", 0.0],
            [2020, 1, "2020-03-01", "c1", "d", "t2", 4, 3, 3, 15.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "a", "t1", 1, 1, 1, 25.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "b", "t1", 3, 5, 6, 2.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "c", "t2", 2, 2, 2, 18.0, "Finished", 0.0],
            [2020, 2, "2020-03-08", "c2", "d", "t2", 4, 3, 3, 15.0, "Finished", 0.0],
        ],
        columns=[
            "season", "round", "date", "circuit_id", "driver_id",
            "constructor_id", "grid", "qual_pos", "position", "points",
            "status", "sprint_points",
        ],
    )
    out = add_features(df)

    a1 = out[(out["driver_id"] == "a") & (out["round"] == 1)].iloc[0]
    assert np.isnan(a1["finish_gap_vs_teammate"])  # no prior race yet

    a2 = out[(out["driver_id"] == "a") & (out["round"] == 2)].iloc[0]
    b2 = out[(out["driver_id"] == "b") & (out["round"] == 2)].iloc[0]
    # Round 1: a finished 1st vs b 5th -> gap -4; qualified 1st vs 4th -> -3.
    assert a2["finish_gap_vs_teammate"] == pytest.approx(-4.0)
    assert a2["qual_gap_vs_teammate"] == pytest.approx(-3.0)
    assert b2["finish_gap_vs_teammate"] == pytest.approx(4.0)
    assert b2["qual_gap_vs_teammate"] == pytest.approx(3.0)
