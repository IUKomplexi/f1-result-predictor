"""Weather data layer: cached Open-Meteo client + race-level helpers."""

from .client import ARCHIVE_URL, DAILY_VARIABLES, FORECAST_URL, WeatherClient
from .fetch import (
    WEATHER_COLUMNS,
    build_weather_frame,
    fetch_race_weather,
    load_race_weather,
    weather_frame,
)

__all__ = [
    "ARCHIVE_URL",
    "DAILY_VARIABLES",
    "FORECAST_URL",
    "WEATHER_COLUMNS",
    "WeatherClient",
    "build_weather_frame",
    "fetch_race_weather",
    "load_race_weather",
    "weather_frame",
]
