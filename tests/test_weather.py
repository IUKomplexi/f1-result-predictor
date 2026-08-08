"""Tests for the weather data layer (f1weather) and its feature wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from helpers import _response, ok_response

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1weather import (
    WEATHER_COLUMNS,
    WeatherClient,
    build_weather_frame,
    fetch_race_weather,
    weather_frame,
)

DAILY = {
    "time": ["2024-08-25"],
    "temperature_2m_max": [24.5],
    "temperature_2m_min": [15.2],
    "precipitation_sum": [0.0],
    "wind_speed_10m_max": [28.4],
    "relative_humidity_2m_mean": [68.0],
    "cloud_cover_mean": [42.0],
}

RAINY_DAILY = {
    "time": ["2024-08-25"],
    "temperature_2m_max": [18.1],
    "temperature_2m_min": [11.0],
    "precipitation_sum": [7.5],
    "wind_speed_10m_max": [55.0],
    "relative_humidity_2m_mean": [92.0],
    "cloud_cover_mean": [98.0],
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class RouteSession:
    """Fake requests.Session routing archive/forecast URLs to fixed payloads."""

    def __init__(self, archive: dict | None = None, forecast: dict | None = None):
        self.archive = archive
        self.forecast = forecast
        self.calls: list = []
        self.headers: dict = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if "archive" in url:
            payload = self.archive
        else:
            payload = self.forecast
        if payload is None:
            return _response(500)
        return ok_response({"daily": payload})


def _client(session, tmp_path, **kw):
    return WeatherClient(cache_dir=tmp_path, sleep_seconds=0, session=session, **kw)


def test_daily_archive_parses(tmp_path):
    session = RouteSession(archive=DAILY)
    client = _client(session, tmp_path)
    daily = client.daily(52.07, 4.31, "2024-08-25", "2024-08-25")
    assert daily["time"] == ["2024-08-25"]
    assert daily["temperature_2m_max"] == [24.5]
    # timezone=auto requested so the local race date matches the daily bucket.
    assert session.calls[0][1]["timezone"] == "auto"
    assert session.calls[0][1]["daily"] == (
        "temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "wind_speed_10m_max,relative_humidity_2m_mean,cloud_cover_mean"
    )


def test_daily_forecast_uses_forecast_url(tmp_path):
    session = RouteSession(forecast=RAINY_DAILY)
    client = _client(session, tmp_path)
    client.daily(52.07, 4.31, "2026-08-23", "2026-08-23", forecast=True)
    assert session.calls[0][0] == FORECAST_URL


def test_forecast_is_never_cached(tmp_path):
    """Forecasts change over time; a stale cached payload must never be served."""
    session = RouteSession(forecast=RAINY_DAILY)
    client = _client(session, tmp_path)
    client.daily(52.07, 4.31, "2026-08-23", "2026-08-23", forecast=True)
    assert list(tmp_path.iterdir()) == []  # no cache file written
    client.daily(52.07, 4.31, "2026-08-23", "2026-08-23", forecast=True)
    assert len(session.calls) == 2  # every forecast call hits the API again


def test_cache_reuses_response_offline(tmp_path):
    session = RouteSession(archive=DAILY)
    client = _client(session, tmp_path)
    client.daily(52.07, 4.31, "2024-08-25", "2024-08-25")
    client.daily(52.07, 4.31, "2024-08-25", "2024-08-25")
    assert len(session.calls) == 1  # second call served from cache


def test_retry_on_server_error(tmp_path):
    failures = {"count": 0}

    class Flaky(RouteSession):
        def get(self, url, params=None, timeout=None):
            failures["count"] += 1
            if failures["count"] < 3:
                return _response(500)
            return ok_response({"daily": DAILY})

    client = _client(Flaky(), tmp_path, max_retries=3)
    daily = client.daily(52.07, 4.31, "2024-08-25", "2024-08-25")
    assert daily["time"] == ["2024-08-25"]
    assert failures["count"] == 3


def test_fetch_race_weather_normalizes(tmp_path):
    client = _client(RouteSession(archive=RAINY_DAILY), tmp_path)
    row = fetch_race_weather(
        client, 2024, 1, "2024-08-25", 52.07, 4.31
    )
    assert row["season"] == 2024 and row["round"] == 1
    assert row["temperature_max"] == 18.1
    assert row["precipitation_sum"] == 7.5
    assert row["wet"] == 1.0
    # missing date -> no row values (caller falls back to NaN)
    fresh = tmp_path / "fresh"
    client2 = _client(RouteSession(archive={**RAINY_DAILY, "time": ["2024-08-24"]}), fresh)
    assert fetch_race_weather(client2, 2024, 1, "2024-08-25", 52.07, 4.31) == {
        "season": 2024, "round": 1
    }


def test_build_weather_frame_skips_missing_coordinates(tmp_path):
    client = _client(RouteSession(archive=DAILY), tmp_path)
    calendar = [
        {"season": 2024, "round": 1, "date": "2024-08-25",
         "circuit_lat": "52.07", "circuit_long": "4.31"},
        {"season": 2024, "round": 2, "date": "2024-09-01"},  # no coords
    ]
    frame = build_weather_frame(client, calendar)
    assert len(frame) == 1
    assert frame.iloc[0]["round"] == 1
    assert set(WEATHER_COLUMNS) <= set(frame.columns)


def test_weather_frame_columns():
    frame = weather_frame(
        [{"season": 2024, "round": 1, "temperature_max": 24.5, "wet": 0.0}]
    )
    assert frame.iloc[0]["temperature_max"] == 24.5


# --- feature wiring ---------------------------------------------------------

def test_add_features_always_adds_nan_weather_columns():
    from features.build import WEATHER_FEATURES, add_features

    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "round": [1, 1],
            "position": [1, 2],
            "grid": [1, 2],
            "points": [25.0, 18.0],
            "status": ["Finished", "Finished"],
            "driver_id": ["a", "b"],
            "constructor_id": ["t1", "t1"],
            "date": ["2024-08-25", "2024-08-25"],
            "circuit_id": ["zandvoort", "zandvoort"],
            "race_name": ["Dutch GP", "Dutch GP"],
            "is_sprint_round": [False, False],
        }
    )
    out = add_features(df)
    for col in WEATHER_FEATURES:
        assert col in out.columns
        assert out[col].isna().all()


def test_merge_weather_is_race_level():
    from features.build import add_features, merge_weather

    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "round": [1, 1, 2],
            "position": [1, 2, 1],
            "grid": [1, 2, 1],
            "points": [25.0, 18.0, 25.0],
            "status": ["Finished", "Finished", "Finished"],
            "driver_id": ["a", "b", "a"],
            "constructor_id": ["t1", "t1", "t1"],
            "date": ["2024-08-25"] * 3,
            "circuit_id": ["zandvoort"] * 3,
            "race_name": ["Dutch GP"] * 3,
            "is_sprint_round": [False] * 3,
        }
    )
    out = add_features(df)
    wx = weather_frame(
        [{"season": 2024, "round": 1, "temperature_max": 24.5,
          "temperature_min": 15.0, "precipitation_sum": 0.0, "wind_max": 28.0,
          "humidity_mean": 68.0, "cloud_cover_mean": 42.0, "wet": 0.0}]
    )
    merged = merge_weather(out, wx)
    race1 = merged[merged["round"] == 1]
    assert (race1["temperature_max"] == 24.5).all()  # constant within the race
    assert merged[merged["round"] == 2]["temperature_max"].isna().all()  # untouched race


def test_build_dataset_merges_weather(tmp_path, monkeypatch):
    from f1data import F1Client
    from features.build import WEATHER_FEATURES, build_dataset

    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "round": [1, 1],
            "position": [1, 2],
            "grid": [1, 2],
            "points": [25.0, 18.0],
            "status": ["Finished", "Finished"],
            "driver_id": ["a", "b"],
            "constructor_id": ["t1", "t1"],
            "date": ["2024-08-25", "2024-08-25"],
            "circuit_id": ["zandvoort", "zandvoort"],
            "race_name": ["Dutch GP", "Dutch GP"],
            "is_sprint_round": [False, False],
        }
    )
    from features import build as fb

    monkeypatch.setattr(
        fb, "add_features", lambda d: d.assign(**{c: np.nan for c in WEATHER_FEATURES})
    )
    monkeypatch.setattr(fb, "_build_fresh", lambda client, seasons, cache: df.copy())

    wx = weather_frame(
        [{"season": 2024, "round": 1, "temperature_max": 24.5,
          "temperature_min": 15.0, "precipitation_sum": 0.0, "wind_max": 28.0,
          "humidity_mean": 68.0, "cloud_cover_mean": 42.0, "wet": 0.0}]
    )
    client = F1Client(cache_dir=tmp_path, session=_dummy_session())
    out = build_dataset(client, [2024], cache_path=tmp_path / "f.parquet", weather=wx)
    assert (out["temperature_max"] == 24.5).all()


def _dummy_session():
    class _S:
        headers: ClassVar[dict[str, str]] = {}

        def get(self, *a, **k):
            raise AssertionError("no network in tests")

    return _S()


def test_weather_is_strictly_pre_race_no_forward_leak():
    """Weather for round 2 must never touch round 1 rows (exact (season, round) join)."""
    from features.build import add_features, merge_weather

    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "round": [1, 1, 2, 2],
            "position": [1, 2, 1, 2],
            "grid": [1, 2, 1, 2],
            "points": [25.0, 18.0, 25.0, 18.0],
            "status": ["Finished"] * 4,
            "driver_id": ["a", "b", "a", "b"],
            "constructor_id": ["t1", "t1", "t1", "t1"],
            "date": ["2024-08-25"] * 4,
            "circuit_id": ["c1", "c1", "c2", "c2"],
            "race_name": ["R1", "R1", "R2", "R2"],
            "is_sprint_round": [False] * 4,
        }
    )
    out = add_features(df)
    wx = weather_frame(
        [{"season": 2024, "round": 2, "temperature_max": 30.0,
          "temperature_min": 20.0, "precipitation_sum": 0.0, "wind_max": 20.0,
          "humidity_mean": 50.0, "cloud_cover_mean": 10.0, "wet": 0.0}]
    )
    merged = merge_weather(out, wx)
    round1 = merged[merged["round"] == 1]
    assert round1["temperature_max"].isna().all()  # R2 weather absent from R1
    round2 = merged[merged["round"] == 2]
    assert (round2["temperature_max"] == 30.0).all()




def test_apply_target_weather_uses_forecast_for_upcoming(monkeypatch):
    """An upcoming (synthetic) race requests the FORECAST and merges it in."""
    import predict as predict_module

    calls = {}

    def fake_load(cache_dir, season, round_, date, lat, long, forecast=False):
        calls["forecast"] = forecast
        return {
            "season": season, "round": round_, "temperature_max": 25.0,
            "temperature_min": 15.0, "precipitation_sum": 0.0, "wind_max": 20.0,
            "humidity_mean": 60.0, "cloud_cover_mean": 30.0, "wet": 0.0,
        }

    import f1weather
    monkeypatch.setattr(f1weather, "load_race_weather", fake_load)
    monkeypatch.setattr(
        predict_module, "_target_calendar_row",
        lambda *a, **k: {"round": 12, "date": "2026-08-23",
                         "circuit_lat": "52.07", "circuit_long": "4.31"},
    )

    df = pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": [12, 12],
            "position": [np.nan, np.nan],
            "grid": [1.0, 2.0],
            "points": [np.nan, np.nan],
            "status": ["", ""],
            "driver_id": ["a", "b"],
            "constructor_id": ["t1", "t1"],
            "date": ["2026-08-23", "2026-08-23"],
            "circuit_id": ["zandvoort", "zandvoort"],
            "race_name": ["Dutch GP", "Dutch GP"],
            "is_sprint_round": [False, False],
        }
    )
    from features.build import add_features

    featured = add_features(df)
    out = predict_module._apply_target_weather(
        featured, None, [], [2010, 2025], 2026, 12, synthetic=True, cache_dir="x"
    )
    assert calls["forecast"] is True  # upcoming race -> live forecast
    assert (out["temperature_max"] == 25.0).all()  # merged onto the race's rows


def test_apply_target_weather_uses_archive_for_past(monkeypatch):
    """A past race loads the cached ERA5 actuals (forecast=False)."""
    import predict as predict_module

    calls = {}

    def fake_load(cache_dir, season, round_, date, lat, long, forecast=False):
        calls["forecast"] = forecast
        return None  # no cached actuals in this test

    import f1weather
    monkeypatch.setattr(f1weather, "load_race_weather", fake_load)
    monkeypatch.setattr(
        predict_module, "_target_calendar_row",
        lambda *a, **k: {"round": 22, "date": "2024-11-23",
                         "circuit_lat": "36.11", "circuit_long": "-115.16"},
    )
    out = predict_module._apply_target_weather(
        pd.DataFrame(), None, [], [2010, 2025], 2024, 22, synthetic=False, cache_dir="x"
    )
    assert calls["forecast"] is False  # past race -> archive actuals
    assert out.empty


def test_merge_weather_accepts_partial_frame():
    """A weather frame without weather columns (e.g. API had no value for the
    date) must not crash; the dataset keeps NaN for those columns."""
    from features.build import WEATHER_FEATURES, add_features, merge_weather

    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "round": [1, 1],
            "position": [1, 2],
            "grid": [1, 2],
            "points": [25.0, 18.0],
            "status": ["Finished", "Finished"],
            "driver_id": ["a", "b"],
            "constructor_id": ["t1", "t1"],
            "date": ["2024-08-25", "2024-08-25"],
            "circuit_id": ["zandvoort", "zandvoort"],
            "race_name": ["Dutch GP", "Dutch GP"],
            "is_sprint_round": [False, False],
        }
    )
    out = add_features(df)
    partial = weather_frame([{"season": 2024, "round": 1}])  # no weather columns
    merged = merge_weather(out, partial)
    assert (merged["round"] == [1, 1]).all()
    for col in WEATHER_FEATURES:
        assert merged[col].isna().all()
