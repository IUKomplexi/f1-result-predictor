"""Cached HTTP client for the Open-Meteo daily weather API.

Two endpoints share one client:

- ``archive-api.open-meteo.com/v1/archive`` — ERA5 reanalysis for past dates
  (training data uses these actuals).
- ``api.open-meteo.com/v1/forecast`` — up to 16 days ahead (the next-race
  prediction uses this forecast).

The discipline mirrors ``f1data``: a descriptive ``User-Agent``, polite rate
limiting, retry/backoff on 429/5xx, and transparent on-disk caching of raw
JSON under ``data/weather/`` keyed by (url, params). No API key is required.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from httpclient import RETRYABLE_STATUS, CachedHTTPClient

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_USER_AGENT = (
    "f1-result-predictor/0.1.0 "
    "(https://github.com/example/f1-result-predictor; contact: dev@example.com)"
)

# Daily variables used as race features. Names are Open-Meteo's.
DAILY_VARIABLES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "cloud_cover_mean",
)


class WeatherClient(CachedHTTPClient):
    """A cached HTTP client for Open-Meteo daily weather."""

    def __init__(
        self,
        cache_dir: str | Path = "data/weather",
        refresh: bool = False,
        sleep_seconds: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(
            user_agent=DEFAULT_USER_AGENT,
            cache_dir=cache_dir,
            refresh=refresh,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
        )

    # -- request -------------------------------------------------------------

    def get_json(self, url: str, params: dict, cache: bool = True) -> dict:
        """GET a JSON document, using cache and retry/backoff as needed.

        ``cache=False`` skips both the read and the write — used for live
        forecasts, which change over time and must never be served stale.
        """
        params = dict(params)
        if cache:
            cached = self._read_cache(url, params)
            if cached is not None:
                return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(self._backoff(attempt - 1))
            try:
                resp = self._get(url, params)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                continue
            if resp.status_code == 200:
                payload: dict = resp.json()
                if cache:
                    self._write_cache(url, params, payload)
                return payload
            last_error = requests.HTTPError(
                f"{resp.status_code} for {url} {params}", response=resp
            )
            if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                continue
            break
        raise requests.HTTPError(
            f"weather request failed: {last_error!r}", response=None
        ) from last_error

    def daily(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        forecast: bool = False,
    ) -> dict[str, list[Any]]:
        """Return the Open-Meteo ``daily`` block for a date range.

        ``forecast=False`` uses the ERA5 archive (past dates); ``forecast=True``
        uses the live forecast (future dates). ``timezone=auto`` aligns the
        daily buckets with the circuit's local day, matching the calendar's
        local race date.
        """
        url = FORECAST_URL if forecast else ARCHIVE_URL
        payload = self.get_json(
            url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": "auto",
            },
            # Forecasts change over time; only immutable archive actuals
            # belong in the cache.
            cache=not forecast,
        )
        daily = payload.get("daily") or {}
        return {
            var: daily.get(var, []) for var in ("time", *DAILY_VARIABLES)
        }
