"""Fetch and cache race-day weather (Open-Meteo) for the cached seasons.

Historical mode (default) fetches ERA5 archive actuals for every race in the
cached raw data — one polite request per race, results cached under
``data/weather/`` so later runs are offline::

    python scripts/fetch_weather.py [--start 2010] [--end 2025]

Forecast mode pre-warms the forecast for one (usually the next) race::

    python scripts/fetch_weather.py --forecast --season 2026 --round 12

``predict.py`` also fetches the forecast on demand when predicting an
upcoming race, so the script is optional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from f1data import F1Client, fetch_season
from f1weather import (
    WeatherClient,
    build_weather_frame,
    fetch_race_weather,
)


def _calendar_rows(client: F1Client, seasons: list[int]) -> list[dict]:
    rows: list[dict] = []
    for s in seasons:
        rows.extend(fetch_season(client, s)["calendar"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=None, help="first season (default: config)")
    parser.add_argument("--end", type=int, default=None, help="last season (default: config)")
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="fetch the forecast for one race instead of historical actuals",
    )
    parser.add_argument("--season", type=int, help="season for --forecast")
    parser.add_argument("--round", type=int, help="round for --forecast")
    parser.add_argument("--cache-dir", default=None, help="weather cache dir (default: config)")
    parser.add_argument("--sleep", type=float, default=None, help="seconds between requests")
    args = parser.parse_args()

    cfg = load_config()
    start = args.start if args.start is not None else cfg["data"]["start_season"]
    end = args.end if args.end is not None else cfg["data"]["end_season"]
    cache_dir = args.cache_dir or cfg["weather"]["cache_dir"]

    f1 = F1Client(
        base_url=cfg["api"]["base_url"],
        user_agent=cfg["api"]["user_agent"],
        cache_dir=cfg["data"]["cache_dir"],
        sleep_seconds=cfg["api"]["sleep_seconds"],
    )
    wx = WeatherClient(cache_dir=cache_dir, sleep_seconds=args.sleep or 0.2)

    if args.forecast:
        if args.season is None or args.round is None:
            raise SystemExit("--forecast requires --season and --round")
        rows = [
            r
            for r in _calendar_rows(f1, [args.season])
            if int(r["round"]) == args.round
        ]
        if not rows:
            raise SystemExit(f"no calendar row for {args.season} R{args.round}")
        row = rows[0]
        if not row.get("circuit_lat") or not row.get("circuit_long"):
            raise SystemExit(f"no circuit coordinates for {args.season} R{args.round}")
        res = fetch_race_weather(
            wx,
            args.season,
            args.round,
            row["date"],
            float(row["circuit_lat"]),
            float(row["circuit_long"]),
            forecast=True,
        )
        print(f"forecast {args.season} R{args.round}: {res}")
        return 0

    seasons = list(range(start, end + 1))
    frame = build_weather_frame(wx, _calendar_rows(f1, seasons))
    print(f"cached weather for {len(frame)} races ({len(frame.columns)} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
