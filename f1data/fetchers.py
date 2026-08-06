"""High-level fetchers returning normalized records from the Jolpica F1 API.

Every fetcher takes an :class:`f1data.client.F1Client` and returns plain
``dict`` records (or lists of them). The raw MRData envelope is parsed here,
so downstream code never sees API-specific JSON shapes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .client import MAX_PAGE_SIZE, F1Client


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_classified(status: str) -> bool:
    """True if the driver was a classified finisher (Finished / +N Laps)."""
    if not status:
        return False
    return status == "Finished" or status.startswith("+")


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

def fetch_calendar(client: F1Client, season: int) -> List[Dict[str, Any]]:
    """Race calendar for a season (round, name, date, circuit, sprint flag)."""
    data = client.get_paged(f"/{season}.json")
    races = data["MRData"]["RaceTable"]["Races"]
    out: List[Dict[str, Any]] = []
    for race in races:
        circuit = race.get("Circuit", {})
        out.append(
            {
                "season": _to_int(race.get("season"), season),
                "round": _to_int(race.get("round")),
                "race_name": race.get("raceName"),
                "date": race.get("date"),
                "time": race.get("time"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "country": circuit.get("Location", {}).get("country"),
                "is_sprint_round": "Sprint" in race,
            }
        )
    return out


# --------------------------------------------------------------------------
# Per-round session results
# --------------------------------------------------------------------------

def _race_rows(race: Dict[str, Any], results_key: str, season: int) -> List[Dict[str, Any]]:
    """Parse one race's result array (Results/QualifyingResults/SprintResults)."""
    rows: List[Dict[str, Any]] = []
    for entry in race.get(results_key, []):
        rows.append(
            {
                "season": _to_int(race.get("season"), season),
                "round": _to_int(race.get("round")),
                "position": _to_int(entry.get("position")),
                "grid": _to_int(entry.get("grid")),
                "points": float(entry.get("points") or 0.0),
                "laps": _to_int(entry.get("laps")),
                "status": entry.get("status", ""),
                "driver_id": entry.get("Driver", {}).get("driverId"),
                "constructor_id": entry.get("Constructor", {}).get("constructorId"),
            }
        )
    return rows


def _session_rows(client: F1Client, season: int, round_: int, endpoint: str,
                  results_key: str) -> List[Dict[str, Any]]:
    """Generic parser for per-round result arrays (Results/QualifyingResults/SprintResults)."""
    data = client.get_paged(f"/{season}/{round_}/{endpoint}.json")
    race = data["MRData"]["RaceTable"]["Races"][0]
    return _race_rows(race, results_key, season)


def _season_session_rows(client: F1Client, season: int, endpoint: str,
                         results_key: str) -> Dict[int, List[Dict[str, Any]]]:
    """Fetch one endpoint for the whole season and group rows by round.

    Jolpica paginates season-level results by *result entries* (``total``
    counts entries, and a single race can span two pages), so we page over
    entries and concatenate each round's rows.
    """
    by_round: Dict[int, List[Dict[str, Any]]] = {}
    offset = 0
    while True:
        data = client.get_json(
            f"/{season}/{endpoint}.json",
            {"limit": MAX_PAGE_SIZE, "offset": offset},
        )
        mdata = data["MRData"]
        races = mdata["RaceTable"]["Races"]
        page_entries = 0
        for race in races:
            round_ = _to_int(race.get("round"))
            rows = _race_rows(race, results_key, season)
            by_round.setdefault(round_, []).extend(rows)
            page_entries += len(rows)
        total = int(mdata.get("total", offset))
        offset += page_entries
        if page_entries == 0 or offset >= total:
            break
    return by_round


def fetch_results(client: F1Client, season: int, round_: int) -> List[Dict[str, Any]]:
    """Race results for one round (position, grid, points, status per driver)."""
    return _session_rows(client, season, round_, "results", "Results")


def fetch_season_results(client: F1Client, season: int) -> Dict[int, List[Dict[str, Any]]]:
    """Race results for every round of a season, keyed by round."""
    return _season_session_rows(client, season, "results", "Results")


def fetch_qualifying(client: F1Client, season: int, round_: int) -> List[Dict[str, Any]]:
    """Qualifying results for one round (position + driver)."""
    return _session_rows(client, season, round_, "qualifying", "QualifyingResults")


def fetch_season_qualifying(client: F1Client, season: int) -> Dict[int, List[Dict[str, Any]]]:
    """Qualifying results for every round of a season, keyed by round."""
    return _season_session_rows(client, season, "qualifying", "QualifyingResults")


def fetch_sprint(client: F1Client, season: int, round_: int) -> List[Dict[str, Any]]:
    """Sprint results for one round, or [] when the round has no sprint.

    Sprint points are *not* part of the main-race points target; they are
    returned so callers can use them as features and to compute championship
    points entering a race.
    """
    return _session_rows(client, season, round_, "sprint", "SprintResults")


# --------------------------------------------------------------------------
# Standings
# --------------------------------------------------------------------------

def _standings_rows(client: F1Client, season: int, round_: int | None,
                    endpoint: str, rows_key: str) -> List[Dict[str, Any]]:
    """Parse standings at a round (default: latest available).

    Jolpica paginates standings by *driver/constructor entries* inside the
    latest round's list (``total`` counts entries, not rounds), so we page
    over the entries of the most recent ``StandingsList`` ourselves rather
    than using :meth:`F1Client.get_paged` (which paginates round items).
    """
    if round_ is not None:
        url = f"/{season}/{round_}/{endpoint}.json"
    else:
        url = f"/{season}/{endpoint}.json"

    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        data = client.get_json(url, {"limit": MAX_PAGE_SIZE, "offset": offset})
        lists = data["MRData"]["StandingsTable"].get("StandingsLists", [])
        if not lists:
            break
        latest = lists[-1]
        entries = latest.get(rows_key, [])
        for entry in entries:
            row: Dict[str, Any] = {
                "season": _to_int(latest.get("season"), season),
                "round": _to_int(latest.get("round"), round_ or 0),
                "position": _to_int(entry.get("position")),
                "points": float(entry.get("points") or 0.0),
                "wins": _to_int(entry.get("wins")),
            }
            if rows_key == "DriverStandings":
                row["driver_id"] = entry.get("Driver", {}).get("driverId")
                constructors = entry.get("Constructors") or entry.get("Constructor")
                if isinstance(constructors, list):
                    row["constructor_id"] = (
                        constructors[0].get("constructorId") if constructors else None
                    )
                elif constructors is not None:
                    row["constructor_id"] = constructors.get("constructorId")
                else:
                    row["constructor_id"] = None
            else:
                row["constructor_id"] = entry.get("Constructor", {}).get("constructorId")
            rows.append(row)

        offset += len(entries)
        total = int(data["MRData"].get("total", offset))
        if not entries or offset >= total:
            break
    return rows


def fetch_driver_standings(client: F1Client, season: int,
                           round_: int | None = None) -> List[Dict[str, Any]]:
    """Driver championship standings at a round (default: latest available)."""
    return _standings_rows(client, season, round_, "driverstandings", "DriverStandings")


def fetch_constructor_standings(client: F1Client, season: int,
                                round_: int | None = None) -> List[Dict[str, Any]]:
    """Constructor championship standings at a round (default: latest available)."""
    return _standings_rows(client, season, round_, "constructorstandings", "ConstructorStandings")


# --------------------------------------------------------------------------
# Whole season
# --------------------------------------------------------------------------

def fetch_season(client: F1Client, season: int) -> Dict[str, Any]:
    """Everything needed to build training features for one season.

    Returns ``{"calendar": [...], "results": {round: [...]},
    "qualifying": {round: [...]}, "sprints": {round: [...]}}``.

    Results and qualifying are fetched season-wide (a few requests per
    season); only sprint results are fetched per sprint round.
    """
    calendar = fetch_calendar(client, season)
    results = fetch_season_results(client, season)
    qualifying = fetch_season_qualifying(client, season)
    sprints: Dict[int, List[Dict[str, Any]]] = {}
    for race in calendar:
        if race["is_sprint_round"]:
            sprints[race["round"]] = fetch_sprint(client, season, race["round"])
    return {
        "calendar": calendar,
        "results": results,
        "qualifying": qualifying,
        "sprints": sprints,
    }
