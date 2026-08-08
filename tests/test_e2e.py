"""End-to-end test: the whole pipeline, offline, on synthetic API data.

Runs fetch -> assemble -> add_features -> train -> predict -> format_report
through the real modules with a fake API session, so CI exercises the full
chain (not just the pieces). The synthetic data is small (3 seasons x 4
rounds x 12 drivers) to keep the test fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import _response, ok_response

from f1data import F1Client, fetch_season
from features.build import add_features, assemble
from model.train import POINTS_TABLE, save_checkpoint, train_final_model
from predict import format_report, predict_race

SEASONS = [2010, 2011, 2012]
ROUNDS = 4
# 12 drivers -> positions 11/12 score 0, so the hurdle classifier sees both
# classes. Six teams of two, so teammate features are defined.
DRIVERS = [f"d{i}" for i in range(1, 13)]
CONSTRUCTORS = [f"c{(i - 1) // 2 + 1}" for i in range(1, 13)]


def _race_meta(season: int, round_: int) -> dict:
    return {
        "season": str(season),
        "round": str(round_),
        "raceName": f"GP {season} R{round_}",
        "date": f"{season}-03-0{round_}",
        "Circuit": {
            "circuitId": f"circuit_{season}_{round_}",
            "circuitName": "Circuit",
            "Location": {"country": "X"},
        },
    }


def _results_race(season: int, round_: int) -> dict:
    race = _race_meta(season, round_)
    race["Results"] = []
    for i, driver in enumerate(DRIVERS, start=1):
        # Finish position rotates with the round so drivers move through the
        # field and the points target takes both zero and non-zero values.
        pos = (i - 1 + round_) % len(DRIVERS) + 1
        race["Results"].append(
            {
                "position": str(pos),
                "grid": str(i),
                "points": str(POINTS_TABLE.get(pos, 0)),
                "laps": "50",
                "status": "Finished",
                "Driver": {"driverId": driver},
                "Constructor": {"constructorId": CONSTRUCTORS[i - 1]},
            }
        )
    return race


def _qualifying_race(season: int, round_: int) -> dict:
    race = _race_meta(season, round_)
    race["QualifyingResults"] = [
        {
            "position": str(i),
            "Driver": {"driverId": driver},
            "Constructor": {"constructorId": CONSTRUCTORS[i - 1]},
        }
        for i, driver in enumerate(DRIVERS, start=1)
    ]
    return race


class SyntheticSession:
    """Serves Jolpica-shaped MRData payloads for the synthetic seasons.

    Seasons 2010-2012 have a full calendar/results/qualifying; any other
    season has a calendar but no results (an upcoming season), and the
    ``last`` endpoint 404s like a real API with no results cached.
    """

    headers = {"User-Agent": "test"}

    def _respond(self, payload: dict):
        return ok_response(payload)

    def get(self, url: str, params=None, timeout=None):
        # url = <base>/f1/<season>.json or <base>/f1/<season>/<endpoint>.json
        path = url.split("/f1/", 1)[1]
        parts = path.split("/")
        season = int(parts[0].split(".")[0])
        endpoint = parts[1].split(".")[0] if len(parts) > 1 else "calendar"
        if endpoint == "calendar":
            races = [_race_meta(season, r) for r in range(1, ROUNDS + 1)]
            return self._respond(
                {"MRData": {"total": ROUNDS,
                            "RaceTable": {"season": str(season), "Races": races}}}
            )
        if endpoint == "drivers":
            return self._respond(
                {"MRData": {"DriverTable": {"season": str(season),
                                            "Drivers": [{"driverId": d} for d in DRIVERS]}}}
            )
        if endpoint == "last":
            return _response(404, text="no results")
        if len(parts) > 2 and parts[2].split(".")[0] == "sprint":
            # Sprint results for round <parts[1]>: synthetic calendars never
            # mark sprint rounds, so this serves an empty sprint (matching the
            # real API's shape for a round without a sprint).
            return self._respond(
                {"MRData": {"total": 0,
                            "RaceTable": {"season": str(season), "Races": []}}}
            )
        if season not in SEASONS:
            # Upcoming season: no results/qualifying to serve.
            return self._respond(
                {"MRData": {"total": 0,
                            "RaceTable": {"season": str(season), "Races": []}}}
            )
        if endpoint == "results":
            races = [_results_race(season, r) for r in range(1, ROUNDS + 1)]
            total = ROUNDS * len(DRIVERS)
        elif endpoint == "qualifying":
            races = [_qualifying_race(season, r) for r in range(1, ROUNDS + 1)]
            total = ROUNDS * len(DRIVERS)
        else:
            raise AssertionError(f"unexpected endpoint: {url}")
        return self._respond(
            {"MRData": {"total": total, "RaceTable": {"season": str(season), "Races": races}}}
        )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """One shared build of the whole chain for both E2E tests."""
    tmp = tmp_path_factory.mktemp("e2e")
    client = F1Client(
        base_url="https://api.jolpi.ca/ergast/f1",
        cache_dir=tmp / "raw",
        sleep_seconds=0,
        session=SyntheticSession(),
    )
    season_datas = [fetch_season(client, s) for s in SEASONS]
    df = add_features(assemble(season_datas))
    model = train_final_model(df)
    checkpoint = tmp / "model.joblib"
    save_checkpoint(model, checkpoint)
    return {"tmp": tmp, "client": client, "df": df, "model": model,
            "checkpoint": str(checkpoint)}


def test_full_pipeline_end_to_end(pipeline):
    df = pipeline["df"]
    assert len(df) == len(SEASONS) * ROUNDS * len(DRIVERS)
    assert set(df["season"]) == set(SEASONS)
    # The target takes both classes (zeros and scorers).
    assert (df["points"] == 0).any() and (df["points"] > 0).any()

    model = pipeline["model"]
    result = predict_race(df, model, SEASONS[-1], ROUNDS)
    assert len(result) == len(DRIVERS)
    assert result["pred_rank"].tolist() == list(range(1, len(DRIVERS) + 1))
    assert result["expected_points"].notna().all()
    assert result["expected_points"].ge(0).all()
    assert result["p_scored"].between(0, 1).all()
    assert result["p_win"].between(0, 1).all()

    # report (verified mode shows actuals and the honesty metrics)
    report = format_report(
        result,
        SEASONS[-1],
        ROUNDS,
        {"race_name": "GP 2012 R4", "circuit_id": "circuit_2012_4",
         "date": "2012-03-04"},
        verified=True,
        checkpoint="synthetic",
        calibrated=False,
    )
    assert "GP 2012 R4" in report
    assert "winner_hit" in report
    assert "d1" in report


def test_pipeline_predicts_upcoming_round_with_synthetic_entries(pipeline):
    """The next-race path (synthetic entry rows) also works end to end.

    Predict a round the cache has no results for: get_prediction injects
    entry rows and still returns a ranked prediction.
    """
    from predict import get_prediction

    tmp = pipeline["tmp"]
    pred = get_prediction(
        season=SEASONS[-1] + 1,  # 2013 - not in SEASONS
        round_=1,
        cfg={
            "data": {"cache_dir": str(tmp / "raw"),
                     "start_season": SEASONS[0], "end_season": SEASONS[-1],
                     "dataset": str(tmp / "features.parquet")},
            "model": {"checkpoint": pipeline["checkpoint"],
                      "calibrators": str(tmp / "calibrators.joblib"),
                      "seed": 42},
            "api": {"base_url": "https://api.jolpi.ca/ergast/f1",
                    "user_agent": "test", "sleep_seconds": 0.0,
                    "timeout": 30.0, "max_retries": 0},
            "report": {"backtest": str(tmp / "backtest.md"),
                       "prediction": str(tmp / "prediction.md")},
        },
        quiet=True,
        client=pipeline["client"],
    )
    assert pred["synthetic"] is True
    assert pred["verified"] is False
    assert len(pred["result"]) == len(DRIVERS)
    assert pred["result"]["expected_points"].notna().all()
