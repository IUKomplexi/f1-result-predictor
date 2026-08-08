"""Race-level weather helpers: (season, round) -> normalized feature row."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .client import WeatherClient

# Feature columns produced for every race (all floats; missing = NaN).
WEATHER_COLUMNS = [
    "temperature_max",
    "temperature_min",
    "precipitation_sum",
    "wind_max",
    "humidity_mean",
    "cloud_cover_mean",
    "wet",
]

WET_PRECIP_MM = 1.0  # precipitation above this counts as a wet race


def _daily_row(daily: dict[str, list[Any]], date: str) -> dict[str, Any] | None:
    """Map Open-Meteo ``daily`` arrays for one date to normalized columns."""
    times = daily.get("time", [])
    if date not in times:
        return None
    i = times.index(date)

    def value(key: str) -> Any:
        arr = daily.get(key)
        return arr[i] if arr else None

    precip = value("precipitation_sum")
    return {
        "temperature_max": value("temperature_2m_max"),
        "temperature_min": value("temperature_2m_min"),
        "precipitation_sum": precip,
        "wind_max": value("wind_speed_10m_max"),
        "humidity_mean": value("relative_humidity_2m_mean"),
        "cloud_cover_mean": value("cloud_cover_mean"),
        "wet": float(precip > WET_PRECIP_MM) if precip is not None else None,
    }


def fetch_race_weather(
    client: WeatherClient,
    season: int,
    round_: int,
    date: str,
    latitude: float,
    longitude: float,
    forecast: bool = False,
) -> dict[str, Any]:
    """Weather for one race as a normalized dict (NaN values when unavailable).

    ``forecast=False`` reads the ERA5 archive (past races); ``forecast=True``
    reads the live forecast (the upcoming race). Results are cached by the
    client, so repeat calls are offline.
    """
    daily = client.daily(latitude, longitude, date, date, forecast=forecast)
    row = _daily_row(daily, date) or {}
    return {"season": season, "round": round_, **row}


def weather_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalized (season, round, weather...) rows as a DataFrame."""
    return pd.DataFrame(rows)


def build_weather_frame(
    client: WeatherClient,
    calendar_rows: list[dict[str, Any]],
    forecast: bool = False,
) -> pd.DataFrame:
    """Weather for every calendar row (skipping rows without coordinates).

    Used by ``scripts/fetch_weather.py`` to populate the cache for all
    historical races at once.
    """
    rows = []
    for row in calendar_rows:
        lat, long = row.get("circuit_lat"), row.get("circuit_long")
        if not lat or not long or not row.get("date"):
            continue
        try:
            rows.append(
                fetch_race_weather(
                    client,
                    int(row["season"]),
                    int(row["round"]),
                    str(row["date"]),
                    float(lat),
                    float(long),
                    forecast=forecast,
                )
            )
        except Exception:  # noqa: BLE001 - one bad race must not abort the sweep
            continue
    return weather_frame(rows)


def load_race_weather(
    cache_dir: str | Path,
    season: int,
    round_: int,
    date: str,
    latitude: float,
    longitude: float,
    forecast: bool = False,
) -> dict[str, Any] | None:
    """Best-effort weather for a race, offline when already cached.

    Returns None (callers fall back to NaN features) on any failure, so a
    missing forecast or an empty cache never breaks prediction.
    """
    client = WeatherClient(cache_dir=cache_dir, sleep_seconds=0)
    try:
        return fetch_race_weather(
            client, season, round_, date, latitude, longitude, forecast=forecast
        )
    except Exception:  # noqa: BLE001 - prediction must survive weather failures
        return None
