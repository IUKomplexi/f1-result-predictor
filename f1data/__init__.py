"""F1 data layer: a polite, cached client for the Jolpica F1 API (Ergast-compatible)."""

from .client import DEFAULT_BASE_URL, DEFAULT_USER_AGENT, F1APIError, F1Client
from .fetchers import (
    fetch_calendar,
    fetch_constructor_standings,
    fetch_driver_standings,
    fetch_qualifying,
    fetch_results,
    fetch_season,
    fetch_season_qualifying,
    fetch_season_results,
    fetch_sprint,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_USER_AGENT",
    "F1APIError",
    "F1Client",
    "fetch_calendar",
    "fetch_constructor_standings",
    "fetch_driver_standings",
    "fetch_qualifying",
    "fetch_results",
    "fetch_season",
    "fetch_season_qualifying",
    "fetch_season_results",
    "fetch_sprint",
]
