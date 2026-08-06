"""Tests for the domain fetchers against recorded API fixtures (no network)."""

from __future__ import annotations

import json

import pytest

from f1data import (
    F1Client,
    fetch_calendar,
    fetch_constructor_standings,
    fetch_driver_standings,
    fetch_qualifying,
    fetch_results,
    fetch_season,
    fetch_sprint,
)
from f1data.fetchers import is_classified

from helpers import FIXTURES, QueueSession, RouteSession, ok_response


@pytest.fixture
def client(tmp_path):
    session = RouteSession.for_fixture_names(
        {
            "/2024.json": "calendar_2024.json",
            "/2024/1/results.json": "results_2024_r1.json",
            "/2024/1/qualifying.json": "qualifying_2024_r1.json",
            "/2024/5/sprint.json": "sprint_2024_r5.json",
        }
    )
    return F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)


def test_fetch_calendar(client):
    calendar = fetch_calendar(client, 2024)
    assert len(calendar) == 24
    first = calendar[0]
    assert first["round"] == 1
    assert first["race_name"] == "Bahrain Grand Prix"
    assert first["circuit_id"] == "bahrain"
    assert first["date"] == "2024-03-02"
    assert first["is_sprint_round"] is False
    # Round 5 (China) is a sprint round in 2024.
    sprint_round = next(r for r in calendar if r["round"] == 5)
    assert sprint_round["is_sprint_round"] is True


def test_fetch_results(client):
    results = fetch_results(client, 2024, 1)
    assert len(results) == 20
    winner = results[0]
    assert winner["driver_id"] == "max_verstappen"
    assert winner["position"] == 1
    assert winner["grid"] == 1
    assert winner["points"] == 26.0  # 25 for the win + 1 fastest-lap point
    assert winner["status"] == "Finished"
    # Exactly 10 drivers scored points in the 2024 points era.
    assert sum(1 for r in results if r["points"] > 0) == 10
    assert sum(r["points"] for r in results) == 102.0  # 25+18+15+12+10+8+6+4+2+1+1


def test_fetch_qualifying(client):
    qualifying = fetch_qualifying(client, 2024, 1)
    assert len(qualifying) == 20
    assert qualifying[0]["driver_id"] == "max_verstappen"
    assert qualifying[0]["position"] == 1
    # Qualifying has no grid/points fields -> parsed safely.
    assert all("grid" in q for q in qualifying)


def test_fetch_sprint(client):
    sprint = fetch_sprint(client, 2024, 5)
    assert len(sprint) == 20
    assert sprint[0]["driver_id"] == "max_verstappen"
    assert sprint[0]["points"] == 8.0  # sprint winner: 8 points in 2024
    # Sprint points are separate from the main race (target uses main race only).
    assert sum(r["points"] for r in sprint) == 36.0  # 8+7+6+5+4+3+2+1


def test_fetch_sprint_absent_round_returns_empty(tmp_path):
    # Round 1 of 2024 has no sprint; the API returns an empty SprintResults list.
    # Route round 1 sprint to a dedicated empty fixture envelope.
    session = RouteSession.for_fixture_names(
        {"/2024/1/sprint.json": "sprint_2024_r1_empty.json"}
    )
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)
    sprint = fetch_sprint(client, 2024, 1)
    assert sprint == []


def test_fetch_season(tmp_path):
    from f1data import F1Client

    session = QueueSession()

    # Two-round synthetic calendar (round 2 is a sprint round).
    calendar = json.loads((FIXTURES / "calendar_2024.json").read_text(encoding="utf-8"))
    race_tpl = calendar["MRData"]["RaceTable"]["Races"][0]
    round1 = json.loads(json.dumps(race_tpl))
    round2 = json.loads(json.dumps(race_tpl))
    round1["round"] = "1"
    round2["round"] = "2"
    round2["raceName"] = "Sprint Grand Prix"
    round2["Sprint"] = {"date": "2024-04-20", "time": "03:00:00Z"}
    mini_cal = json.loads(json.dumps(calendar))
    mini_cal["MRData"]["RaceTable"]["Races"] = [round1, round2]
    mini_cal["MRData"]["total"] = "2"

    # Season-level results / qualifying: attach the recorded round-1 arrays to
    # both synthetic races.
    results = json.loads((FIXTURES / "results_2024_r1.json").read_text(encoding="utf-8"))
    qualifying = json.loads((FIXTURES / "qualifying_2024_r1.json").read_text(encoding="utf-8"))
    sprint = json.loads((FIXTURES / "sprint_2024_r5.json").read_text(encoding="utf-8"))
    season_results = json.loads(json.dumps(mini_cal))
    season_qual = json.loads(json.dumps(mini_cal))
    for race in season_results["MRData"]["RaceTable"]["Races"]:
        race["Results"] = results["MRData"]["RaceTable"]["Races"][0]["Results"]
    for race in season_qual["MRData"]["RaceTable"]["Races"]:
        race["QualifyingResults"] = qualifying["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]

    def route(url, params=None, timeout=None):
        session.calls.append((url, dict(params or {})))
        if url.endswith("/2024.json"):
            return ok_response(mini_cal)
        if url.endswith("/2024/results.json"):
            return ok_response(season_results)
        if url.endswith("/2024/qualifying.json"):
            return ok_response(season_qual)
        if url.endswith("/2024/2/sprint.json"):
            return ok_response(sprint)
        raise AssertionError(f"No route for {url}")

    session.get = route  # type: ignore[method-assign]
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    season = fetch_season(client, 2024)

    assert [r["round"] for r in season["calendar"]] == [1, 2]
    assert set(season["results"].keys()) == {1, 2}
    assert set(season["qualifying"].keys()) == {1, 2}
    assert set(season["sprints"].keys()) == {2}  # only the sprint round
    assert len(season["results"][1]) == 20
    assert len(season["sprints"][2]) == 20
    # Season-wide fetching: one request per endpoint, not one per round.
    season_urls = [u for u, _ in session.calls if u.endswith("/2024.json")]
    results_urls = [u for u, _ in session.calls if u.endswith("/2024/results.json")]
    assert len(season_urls) == 1 and len(results_urls) == 1


def test_fetch_driver_standings(tmp_path):
    session = RouteSession.for_fixture_names(
        {"/2024/driverstandings.json": "standings_2024.json"}
    )
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    standings = fetch_driver_standings(client, 2024)

    assert len(standings) == 24
    assert standings[0]["driver_id"] == "max_verstappen"
    assert standings[0]["position"] == 1
    assert standings[0]["points"] == 437.0
    assert standings[0]["constructor_id"] == "red_bull"


def test_is_classified():
    assert is_classified("Finished") is True
    assert is_classified("+1 Lap") is True
    assert is_classified("+4 Laps") is True
    assert is_classified("Collision") is False
    assert is_classified("Engine") is False
    assert is_classified("") is False
