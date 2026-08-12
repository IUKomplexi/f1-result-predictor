"""Predict expected race points for an F1 grid using the trained hurdle model.

Usage::

    f1 predict                               # next race (latest season's next round)
    f1 predict --season 2024 --round 22      # any race; past races are verified vs actuals
    f1 predict --grid qual.csv               # supply a qualifying grid
                                             # (driver_id,grid) for an upcoming race

Requires a trained checkpoint (``f1 train``) and cached raw data
(``python scripts/fetch_all.py``). Output: the ranked grid with expected
points and win/podium probabilities, saved to ``reports/prediction.md``.

For an upcoming race the grid/qualifying result is not known yet: grid and
qual_pos default to missing (the gradient-boosted models handle missing
values natively) unless supplied via ``--grid``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1core.config import load_config
from f1core.reporting import rank_by, to_md
from f1data import F1APIError, F1Client, fetch_calendar, fetch_season
from features.build import add_features, assemble
from features.registry import enabled_features, feature_fingerprint
from model.calibrate import apply_calibration, load_calibrators
from model.evaluate import race_metrics
from model.train import load_checkpoint, model_params, prepare, quantize_points

# --------------------------------------------------------------------------
# Target race discovery
# --------------------------------------------------------------------------

def _next_round_in_calendar(calendar: list[dict], completed: set) -> int | None:
    """First calendar round without cached results, or None."""
    for race in sorted(calendar, key=lambda r: r["round"]):
        if race["round"] not in completed:
            return race["round"]
    return None


def _latest_completed_round(client: F1Client, season: int) -> int:
    """Round of the most recent completed race via the API (0 when none)."""
    try:
        data = client.get_json(f"/{season}/last/results.json")
        return int(data["MRData"]["RaceTable"]["Races"][0]["round"])
    except (F1APIError, KeyError, IndexError, ValueError):
        return 0


def find_next_race(client: F1Client, df: pd.DataFrame, seasons: Sequence[int]) -> tuple[int, int]:
    """Locate the next race without results: the first unfinished round of
    the *newest* cached season, else the first unfinished round of the next
    season (discovered on demand from the API).

    Only the newest cached season is eligible — a gap in an older season
    (cancelled round, partial cache) is historical, not "the next race".
    """
    newest = max(seasons)
    completed = set(df.loc[df["season"] == newest, "round"])
    calendar = fetch_calendar(client, newest)
    round_ = _next_round_in_calendar(calendar, completed)
    if round_ is not None:
        return newest, round_

    nxt = newest + 1
    try:
        calendar = fetch_calendar(client, nxt)
    except F1APIError as exc:
        raise SystemExit(
            f"no upcoming race: seasons {min(seasons)}-{max(seasons)} are complete "
            f"and season {nxt} is not available ({exc}). Pass --season/--round explicitly."
        ) from None
    if not calendar:
        raise SystemExit(f"season {nxt} has an empty calendar; pass --season/--round explicitly.")
    latest = _latest_completed_round(client, nxt)
    # Every round up to and including the latest completed one has results.
    completed = {r["round"] for r in calendar if r["round"] <= latest}
    round_ = _next_round_in_calendar(calendar, completed)
    if round_ is None:
        raise SystemExit(f"season {nxt} is fully raced; extend config [data] end_season.")
    return nxt, round_


# --------------------------------------------------------------------------
# Entry list for an upcoming race
# --------------------------------------------------------------------------

def _apply_target_weather(
    df: pd.DataFrame,
    client: F1Client,
    season_datas: Sequence[dict],
    seasons: list[int],
    target_season: int,
    target_round: int,
    synthetic: bool,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Merge the target race's weather onto its rows (best-effort).

    Past races load the cached ERA5 actuals; an upcoming race (``synthetic``)
    requests the live forecast. Any failure keeps the weather columns NaN, so
    prediction never breaks on missing weather.

    The weather imports are lazy: this helper is plumbing for a future
    adoption (the shipped model does not use weather), so `predict` must not
    depend on the ``f1weather`` package at import time.
    """
    from f1weather import load_race_weather, weather_frame  # lazy: plumbing only
    from features.build import merge_weather

    row = _target_calendar_row(client, season_datas, seasons, target_season, target_round)
    weather = None
    if row and row.get("circuit_lat") and row.get("circuit_long"):
        weather = load_race_weather(
            cache_dir,
            target_season,
            target_round,
            str(row["date"]),
            float(row["circuit_lat"]),
            float(row["circuit_long"]),
            forecast=synthetic,
        )
    if weather:
        df = merge_weather(df, weather_frame([weather]))
    return df


def _target_calendar_row(
    client: F1Client,
    season_datas: Sequence[dict],
    seasons: list[int],
    target_season: int,
    target_round: int,
) -> dict | None:
    """Calendar row for (target_season, target_round), for weather lookups.

    Uses the already-fetched season data when the season is in range (offline),
    otherwise fetches the season's calendar (needed for an upcoming season).
    """
    if seasons[0] <= target_season <= seasons[-1]:
        calendar = season_datas[target_season - seasons[0]].get("calendar", [])
    else:
        calendar = fetch_calendar(client, target_season)
    return next(
        (r for r in calendar if int(r["round"]) == target_round), None
    )


def _latest_teams_from_df(df: pd.DataFrame) -> dict[str, str]:
    """driver -> constructor from each driver's most recent cached race."""
    latest = df.sort_values("date").drop_duplicates("driver_id", keep="last")
    return dict(zip(latest["driver_id"], latest["constructor_id"], strict=True))


def _entry_list(client: F1Client, season: int, df: pd.DataFrame) -> list[tuple[str, str | None]]:
    """(driver_id, constructor_id) for the upcoming race's grid.

    The grid of the season's most recent completed race is the base (real
    entrants, real teams). It is unioned with season entrants that hold a
    known team from the cached history, so a driver who missed that race
    (injury return, one-off substitute) is not dropped. For a season with no
    completed race yet (an opener), all season entrants are returned with
    teams from the cached history where known.
    """
    try:
        drivers = [
            d["driverId"]
            for d in client.get_json(f"/{season}/drivers.json")["MRData"]["DriverTable"]["Drivers"]
        ]
    except (F1APIError, KeyError, TypeError):
        drivers = []  # fall back to grid-only below
    completed = _latest_completed_round(client, season)
    if not completed:
        team_of = _latest_teams_from_df(df)
        if not drivers:
            raise SystemExit(
                f"could not determine the entry list for season {season} "
                "(drivers endpoint unavailable and no completed race)"
            )
        return [(d, team_of.get(d)) for d in drivers]

    team_of = _latest_teams_from_df(df)
    grid: list[tuple[str, str | None]] = []
    try:
        data = client.get_json(f"/{season}/{completed}/results.json")
        results = data["MRData"]["RaceTable"]["Races"][0]["Results"]
        grid = [(e["Driver"]["driverId"], e["Constructor"]["constructorId"]) for e in results]
        # Last race wins; None teams keep the driver's prior entry.
        team_of.update({k: v for k, v in grid if v is not None})
    except (F1APIError, KeyError, IndexError):
        grid = []

    grid_ids = {d for d, _ in grid}
    extras = [(d, team_of[d]) for d in drivers if d not in grid_ids and d in team_of]
    return grid + extras


def _synthetic_rows(
    calendar: list[dict],
    target_round: int,
    entries: Sequence[tuple[str, str | None]],
    grid_map: dict[str, int] | None,
) -> list[dict]:
    """Result-style rows for an upcoming race, with unknown outcomes (NaN).

    Sprint points are not set: they are unknown at prediction time (the
    sprint runs Saturday), so 0 is the honest pre-weekend default.
    """
    race = next((r for r in calendar if r["round"] == target_round), None)
    if race is None:
        raise SystemExit(
            f"round {target_round} is not in the season's calendar "
            "(cancelled round or invalid --round?)"
        )
    rows = []
    for driver, constructor in entries:
        grid = grid_map.get(driver) if grid_map else np.nan
        rows.append(
            {
                "season": race["season"],
                "round": race["round"],
                "position": np.nan,
                "grid": grid,
                "points": np.nan,
                "status": "",  # marker: assemble keeps it, add_features sees "not classified"
                "driver_id": driver,
                "constructor_id": constructor,
            }
        )
    return rows


def _target_frame(
    client: F1Client,
    base_df: pd.DataFrame,
    season_datas: list[dict],
    seasons: list[int],
    target_season: int,
    target_round: int,
    grid_csv: str | None,
    quiet: bool,
) -> tuple[pd.DataFrame, bool]:
    """(feature frame, is_synthetic) for the target race.

    A round with no cached results is an upcoming race: synthetic entry rows
    are injected so pre-race features are derived exactly as they would be
    before the race.
    """
    synthetic = base_df[(base_df["season"] == target_season) &
                        (base_df["round"] == target_round)].empty
    if not synthetic:
        return base_df, False
    if not quiet:
        print(
            f"Note: no cached results for {target_season} R{target_round} - "
            "prediction uses synthetic entry rows and is unverified.",
            file=sys.stderr,
        )
    calendar = fetch_calendar(client, target_season)
    grid_map = read_grid_csv(grid_csv) if grid_csv else None
    rows = _synthetic_rows(
        calendar, target_round, _entry_list(client, target_season, base_df), grid_map
    )
    if seasons[0] <= target_season <= seasons[-1]:
        season_datas[target_season - seasons[0]]["results"][target_round] = rows
    else:
        season_datas.append(
            {"calendar": calendar, "results": {target_round: rows},
             "qualifying": {}, "sprints": {}}
        )
    return add_features(assemble(season_datas)), True


def read_grid_csv(path: str | Path) -> dict[str, int]:
    """Load a qualifying grid override (CSV with ``driver_id,grid`` columns)."""
    table = pd.read_csv(path)
    for col in ("driver_id", "grid"):
        if col not in table.columns:
            raise SystemExit(f"--grid file {path} must have columns 'driver_id' and 'grid'")
    return dict(zip(table["driver_id"], table["grid"].astype(int), strict=True))


# --------------------------------------------------------------------------
# JSON-safe payload + disk-backed prediction cache
# --------------------------------------------------------------------------

# Default location for the disk-backed prediction cache (gitignored, like the
# other data/ artifacts). Overridable per-call via ``cache_dir``.
DEFAULT_PREDICTION_CACHE = "data/predictions"


def _json_safe(value: Any) -> Any:
    """Recursively make a value JSON-serializable (pandas/numpy aware).

    The prediction ``meta`` carries a ``date`` that may be a ``pd.Timestamp``
    (and rows may hold numpy scalars). These serialize fine through FastAPI's
    ``jsonable_encoder`` but not through raw ``json.dumps`` (the disk cache),
    so normalize them here.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, (np.generic, np.ndarray)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return float(value)
    return value


def prediction_payload(pred: dict) -> dict:
    """JSON-safe representation of a prediction dict (drivers as records).

    Mirrors the web dashboard's ``_payload`` shape so a cached entry can be
    returned directly from ``/api/prediction`` and the season endpoint without
    re-serializing the underlying DataFrame.
    """
    rows = json.loads(pred["result"].to_json(orient="records"))
    return {
        "season": pred["season"],
        "round": pred["round"],
        "race": _json_safe(pred["meta"]),
        "synthetic": pred["synthetic"],
        "verified": pred["verified"],
        "calibrated": pred["calibrated"],
        "checkpoint": pred["checkpoint"],
        "features": pred.get("features"),
        "drivers": rows,
    }


def _params_hash(cfg: dict) -> str:
    """Stable short hash of the effective ``[model.params]`` (cache key part)."""
    payload = json.dumps(model_params(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def prediction_cache_key(
    season: int, round_: int, feats_fingerprint: str, params_hash: str
) -> str:
    """Cache filename key for ``(season, round)`` given the feature fingerprint.

    A feature-selection or ``[model.params]`` change alters the fingerprint /
    params hash, so stale entries are naturally bypassed (never read, never
    written over) rather than invalidated eagerly.
    """
    payload = f"{season}|{round_}|{feats_fingerprint}|{params_hash}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_cached_prediction(cache_dir: str | Path, key: str) -> dict | None:
    """Read a cached JSON payload for ``key``, or None when missing/corrupt."""
    path = Path(cache_dir) / f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_prediction(cache_dir: str | Path, key: str, payload: dict) -> None:
    """Atomically write a JSON payload for ``key`` under ``cache_dir``."""
    path = Path(cache_dir) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _pred_from_payload(payload: dict) -> dict:
    """Rebuild a prediction dict (with a DataFrame ``result``) from a cached payload."""
    result = pd.DataFrame(payload["drivers"])[
        [
            "pred_rank", "driver_id", "constructor_id", "grid", "expected_points",
            "p_scored", "p_top3", "p_win", "actual_points", "actual_position",
        ]
    ]
    return {
        "result": result,
        "meta": payload["race"],
        "season": payload["season"],
        "round": payload["round"],
        "synthetic": payload["synthetic"],
        "verified": payload["verified"],
        "calibrated": payload["calibrated"],
        "checkpoint": payload["checkpoint"],
        "features": payload.get("features"),
    }


# --------------------------------------------------------------------------
# Shared prediction context (built once, reused across target races)
# --------------------------------------------------------------------------

def _make_client(cfg: dict, refresh: bool, client: F1Client | None) -> F1Client:
    """The configured API client, or an injected one (tests)."""
    if client is not None:
        return client
    return F1Client(
        base_url=cfg["api"]["base_url"],
        user_agent=cfg["api"]["user_agent"],
        cache_dir=cfg["data"]["cache_dir"],
        refresh=refresh,
        sleep_seconds=cfg["api"]["sleep_seconds"],
        timeout=cfg["api"]["timeout"],
        max_retries=cfg["api"]["max_retries"],
    )


def _featured_frame(client: F1Client, seasons: Sequence[int]) -> tuple[list[dict], pd.DataFrame]:
    """Fetch + assemble + feature every cached season exactly once.

    This is the expensive part of a prediction (one dataset pass); scoring many
    rounds from the returned frame avoids repeating it per race.
    """
    season_datas = [fetch_season(client, s) for s in seasons]
    return season_datas, add_features(assemble(season_datas))


def _load_models(cfg: dict, model_path: str | None, feats: Sequence[str]) -> tuple:
    """(checkpoint, model, calibrators) for the configured model + feature set."""
    checkpoint = model_path or cfg["model"]["checkpoint"]
    model = load_checkpoint(checkpoint, expected=feats)
    calibrators = load_calibrators(cfg["model"]["calibrators"], expected=list(feats))
    return checkpoint, model, calibrators


def _race_meta(df: pd.DataFrame, season: int, round_: int) -> dict:
    """race_name/circuit_id/date for a round, tolerant of a missing round."""
    target_rows = df[(df["season"] == season) & (df["round"] == round_)]
    return {k: (target_rows[k].iloc[0] if k in target_rows else None)  # type: ignore[reportAttributeAccessIssue]  # boolean-mask slice is Unknown
            for k in ("race_name", "circuit_id", "date")}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _rank_expected(out: pd.DataFrame) -> pd.DataFrame:
    """Sort by expected points desc; ties broken by grid (pit-lane starts last).

    Uses the shared :func:`reporting.rank_by` so quantized-point ties rank
    identically to the backtest's ``race_metrics``.
    """
    out = out.assign(_rank=rank_by(out, "expected_points", "grid"))
    out = out.sort_values("_rank").reset_index(drop=True)
    out.insert(0, "pred_rank", range(1, len(out) + 1))
    return out.drop(columns="_rank")


def predict_race(
    df: pd.DataFrame,
    model,
    season: int,
    round_: int,
    calibrators: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Score every row of ``(season, round_)`` and rank by expected points.

    Returns driver/constructor/grid, expected points, P(scored)/P(top-3)/P(win)
    and ``pred_rank``; actual points/position are included when the race has
    results (``actual_points`` is NaN for upcoming races). When ``calibrators``
    are given, the probability columns are isotonic-calibrated. ``features``
    selects the model columns (must match the checkpoint's training set;
    default: the full set).
    """
    X, _ = prepare(df, features)
    target = df[(df["season"] == season) & (df["round"] == round_)]
    if target.empty:
        raise ValueError(f"no rows for season {season} round {round_}")

    X_target = X.loc[target.index]
    # Deployed output is quantized to the points table (adopted because it
    # improves walk-forward MAE/top-3/Spearman); ranking uses a grid tiebreak
    # for equal quantized values, matching the backtest.
    expected = quantize_points(model.predict_expected_points(X_target))
    probs = model.predict_probs(X_target)
    if calibrators:
        probs = apply_calibration(probs, calibrators)

    out = pd.DataFrame(
        {
            "driver_id": target["driver_id"].to_numpy(),  # type: ignore[reportAttributeAccessIssue]  # target is a boolean-mask slice (Unknown)
            "constructor_id": target["constructor_id"].to_numpy(),  # type: ignore[reportAttributeAccessIssue]
            "grid": target["grid"].to_numpy(),  # type: ignore[reportAttributeAccessIssue]
            "expected_points": expected,
            "p_scored": probs["p_scored"],
            "p_top3": probs["p_top3"],
            "p_win": probs["p_win"],
            "actual_points": target["points"].to_numpy(dtype=float),  # type: ignore[reportAttributeAccessIssue]
            "actual_position": target["position"].to_numpy(),  # type: ignore[reportAttributeAccessIssue]
        }
    )
    out = _rank_expected(out)
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_report(
    result: pd.DataFrame,
    season: int,
    round_: int,
    meta: dict,
    verified: bool,
    checkpoint: str,
    calibrated: bool = False,
) -> str:
    lines = [
        f"# Prediction: {meta.get('race_name', f'Round {round_}')} ({season} Round {round_})",
        "",
        f"- Circuit: {meta.get('circuit_id', '?')} · Date: {meta.get('date', '?')}",
        f"- Model checkpoint: `{checkpoint}`",
    ]
    if calibrated:
        lines.append(
            "- Probability columns are isotonic-calibrated model scores where "
            "calibration improved hold-out Brier (others stay raw); ranking is "
            "the primary output."
        )
    else:
        lines.append(
            "- Probabilities are raw model scores (P top-10 / top-3 / win); "
            "ranking is the primary output."
        )
    lines.append(
        "- Expected points are quantized to the points table; ties are broken "
        "by grid (pit-lane starts last). For an upcoming race the grid is "
        "unknown, so order within a tied bucket is entry-list order."
    )
    if not verified:
        lines.append(
            "- Unverified: no cached results for this race; entry list from "
            "the latest completed round."
        )
    lines += ["", "## Predicted grid (ranked by expected points)", ""]
    table = result[
        [
            "pred_rank", "driver_id", "constructor_id", "grid",
            "expected_points", "p_scored", "p_top3", "p_win",
        ]
    ].copy()
    table["expected_points"] = table["expected_points"].round(2)
    table["p_scored"] = (table["p_scored"] * 100).round(1).astype(str) + "%"
    table["p_top3"] = (table["p_top3"] * 100).round(1).astype(str) + "%"
    table["p_win"] = (table["p_win"] * 100).round(1).astype(str) + "%"
    lines.append(to_md(table))  # type: ignore[reportArgumentType]  # table is a column slice (Unknown)
    lines.append("")

    if verified:
        vdf = result.rename(
            columns={"actual_points": "points", "actual_position": "position"}
        ).copy()
        vdf["pred_points"] = vdf["expected_points"]
        m = race_metrics(vdf)
        actual = result[["pred_rank", "driver_id", "actual_points", "actual_position"]].copy()
        actual["actual_points"] = actual["actual_points"].astype(float)
        lines.append("## Actual results")
        lines.append("")
        lines.append(to_md(actual))  # type: ignore[reportArgumentType]  # column slice (Unknown)
        lines.append("")
        lines.append("## Verification vs actuals")
        lines.append("")
        lines.append(
            f"- winner_hit: {m['winner_hit']:.2f} · top3_overlap: {m['top3_overlap']:.2f} "
            f"· spearman: {m['spearman']:.2f} · MAE (points): {m['mae']:.2f}"
        )
        lines.append("")
        lines.append(
            "Note: this uses the final model (trained on all seasons, including "
            "this race). Honest out-of-sample numbers are in `reports/backtest.md`."
        )
        lines.append("")
    return "\n".join(lines)


def format_console(
    result: pd.DataFrame,
    meta: dict,
    season: int,
    round_: int,
) -> str:
    """Compact ranked-grid console output for the `f1 predict` CLI."""
    console = result[
        [
            "pred_rank", "driver_id", "constructor_id", "expected_points",
            "p_scored", "p_top3", "p_win",
        ]
    ].copy()
    console["expected_points"] = console["expected_points"].round(2)
    console["p_scored"] = (console["p_scored"] * 100).round(1)
    console["p_top3"] = (console["p_top3"] * 100).round(1)
    console["p_win"] = (console["p_win"] * 100).round(1)
    console = console.rename(
        columns={"p_scored": "p_scored%", "p_top3": "p_top3%", "p_win": "p_win%"}
    )
    header = (
        f"Prediction: {meta.get('race_name', f'Round {round_}')} "
        f"({season} R{round_}) - {meta.get('circuit_id', '?')} "
        f"{meta.get('date', '')}".rstrip()
    )
    return f"{header}\n{console.to_string(index=False)}"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def get_prediction(
    season: int | None = None,
    round_: int | None = None,
    grid_csv: str | None = None,
    refresh: bool = False,
    cfg: dict | None = None,
    model_path: str | None = None,
    quiet: bool = False,
    client: F1Client | None = None,
    enable_features: Sequence[str] | None = None,
    disable_features: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    """Compute a prediction, returning results + metadata (no I/O side effects).

    ``season``/``round_`` target a specific race; both None selects the next
    race. ``grid_csv`` supplies a qualifying grid for an upcoming race.
    ``client`` injects an F1Client (tests); the default is built from ``cfg``.
    ``enable_features``/``disable_features`` override the config feature
    selection. When ``cache_dir`` is given, the JSON payload is cached on disk
    keyed by ``(season, round, feature-fingerprint, params-hash)`` so repeat
    calls (and the web dashboard) are instant; a feature-selection or
    ``[model.params]`` change bypasses stale entries automatically. Returns a
    dict with: ``result`` (DataFrame from :func:`predict_race`), ``meta``
    (race_name/circuit_id/date), ``season``, ``round``, ``synthetic``,
    ``verified``, ``calibrated``, ``checkpoint``, ``features``.
    """
    cfg = cfg or load_config()
    if round_ is not None and season is None:
        raise ValueError("season is required when round is given")
    seasons = list(range(cfg["data"]["start_season"], cfg["data"]["end_season"] + 1))
    feats = enabled_features(cfg, enable=enable_features or [], disable=disable_features or [])
    client = _make_client(cfg, refresh, client)

    # An explicit target is known up-front, so a cache hit skips the expensive
    # dataset assembly entirely. The "next race" default needs the assembled
    # frame to discover the target first.
    if season is not None:
        if round_ is None:
            raise ValueError("round_ is required when season is given")
        target_season, target_round = season, round_
        needs_frame = True
    else:
        season_datas, base_df = _featured_frame(client, seasons)
        target_season, target_round = find_next_race(client, base_df, seasons)
        needs_frame = False

    if cache_dir is not None:
        key = prediction_cache_key(
            target_season, target_round, feature_fingerprint(feats), _params_hash(cfg)
        )
        cached = load_cached_prediction(cache_dir, key)
        if cached is not None:
            return _pred_from_payload(cached)

    if needs_frame:
        season_datas, base_df = _featured_frame(client, seasons)
    df, synthetic = _target_frame(
        client, base_df, season_datas, seasons, target_season, target_round, grid_csv, quiet
    )

    # Weather is not adopted (see reports/weather.md): the shipped model does
    # not use weather features, so no forecast is fetched at predict time.
    # The plumbing (_apply_target_weather / merge_weather) stays available
    # for a future re-evaluation.

    checkpoint, model, calibrators = _load_models(cfg, model_path, feats)
    result = predict_race(df, model, target_season, target_round, calibrators, feats)

    meta = _race_meta(df, target_season, target_round)
    pred = {
        "result": result,
        "meta": meta,
        "season": target_season,
        "round": target_round,
        "synthetic": synthetic,
        "verified": not synthetic,
        "calibrated": bool(calibrators),
        "checkpoint": checkpoint,
        "features": feats,
    }
    if cache_dir is not None:
        save_cached_prediction(cache_dir, key, prediction_payload(pred))
    return pred


def predict_season(
    season: int,
    cfg: dict | None = None,
    model_path: str | None = None,
    quiet: bool = True,
    client: F1Client | None = None,
    enable_features: Sequence[str] | None = None,
    disable_features: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    """Score every completed round of ``season`` in one dataset pass.

    Assembles the featured frame once (instead of once per race, the old
    Race History cost) and scores each completed round from it. Each prediction
    is shaped like :func:`get_prediction`'s return; ``synthetic`` is always
    False here because only cached (completed) rounds are scored. When
    ``cache_dir`` is given, per-round payloads are also written to the shared
    disk cache so single-race fetches are instant too.
    """
    cfg = cfg or load_config()
    seasons = list(range(cfg["data"]["start_season"], cfg["data"]["end_season"] + 1))
    client = _make_client(cfg, False, client)
    _, base_df = _featured_frame(client, seasons)
    completed = sorted(
        base_df.loc[base_df["season"] == season, "round"].astype(int).unique()
    )
    feats = enabled_features(cfg, enable=enable_features or [], disable=disable_features or [])
    checkpoint, model, calibrators = _load_models(cfg, model_path, feats)

    preds: list[dict] = []
    for round_ in completed:
        result = predict_race(base_df, model, season, round_, calibrators, feats)
        pred = {
            "result": result,
            "meta": _race_meta(base_df, season, round_),
            "season": season,
            "round": int(round_),
            "synthetic": False,
            "verified": True,
            "calibrated": bool(calibrators),
            "checkpoint": checkpoint,
            "features": feats,
        }
        if cache_dir is not None:
            key = prediction_cache_key(
                season, int(round_), feature_fingerprint(feats), _params_hash(cfg)
            )
            save_cached_prediction(cache_dir, key, prediction_payload(pred))
        preds.append(pred)
    return preds
