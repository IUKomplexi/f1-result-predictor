"""Predict expected race points for an F1 grid using the trained hurdle model.

Usage::

    python predict.py                            # next race (latest season's next round)
    python predict.py --season 2024 --round 22   # any race; past races are verified vs actuals
    python predict.py --grid qual.csv            # supply a qualifying grid
                                                # (driver_id,grid) for an upcoming race

Requires a trained checkpoint (``python model/train.py``) and cached raw data
(``python scripts/fetch_all.py``). Output: the ranked grid with expected
points and win/podium probabilities, saved to ``reports/prediction.md``.

For an upcoming race the grid/qualifying result is not known yet: grid and
qual_pos default to missing (the gradient-boosted models handle missing
values natively) unless supplied via ``--grid``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from f1data import F1APIError, F1Client, fetch_calendar, fetch_season
from features.build import add_features, assemble
from model.calibrate import apply_calibration, load_calibrators
from model.evaluate import race_metrics
from model.train import load_checkpoint, prepare, quantize_points
from reporting import rank_by, to_md

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
) -> pd.DataFrame:
    """Score every row of ``(season, round_)`` and rank by expected points.

    Returns driver/constructor/grid, expected points, P(scored)/P(top-3)/P(win)
    and ``pred_rank``; actual points/position are included when the race has
    results (``actual_points`` is NaN for upcoming races). When ``calibrators``
    are given, the probability columns are isotonic-calibrated.
    """
    X, _ = prepare(df)
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
) -> dict:
    """Compute a prediction, returning results + metadata (no I/O side effects).

    ``season``/``round_`` target a specific race; both None selects the next
    race. ``grid_csv`` supplies a qualifying grid for an upcoming race.
    ``client`` injects an F1Client (tests); the default is built from ``cfg``.
    Returns a dict with: ``result`` (DataFrame from :func:`predict_race`),
    ``meta`` (race_name/circuit_id/date), ``season``, ``round``,
    ``synthetic``, ``verified``, ``calibrated``, ``checkpoint``.
    """
    cfg = cfg or load_config()
    if round_ is not None and season is None:
        raise ValueError("season is required when round is given")
    seasons = list(range(cfg["data"]["start_season"], cfg["data"]["end_season"] + 1))
    if client is None:
        client = F1Client(
            base_url=cfg["api"]["base_url"],
            user_agent=cfg["api"]["user_agent"],
            cache_dir=cfg["data"]["cache_dir"],
            refresh=refresh,
            sleep_seconds=cfg["api"]["sleep_seconds"],
            timeout=cfg["api"]["timeout"],
            max_retries=cfg["api"]["max_retries"],
        )

    # Assemble every cached season once, then (re)compute features. For an
    # upcoming race we inject synthetic rows so the pre-race features are
    # derived exactly as they would be before the race.
    season_datas = [fetch_season(client, s) for s in seasons]
    base_df = add_features(assemble(season_datas))

    if season is not None:
        if round_ is None:
            raise ValueError("round_ is required when season is given")
        target_season, target_round = season, round_
    else:
        target_season, target_round = find_next_race(client, base_df, seasons)

    df, synthetic = _target_frame(
        client, base_df, season_datas, seasons, target_season, target_round, grid_csv, quiet
    )

    # Weather is not adopted (see reports/weather.md): the shipped model does
    # not use weather features, so no forecast is fetched at predict time.
    # The plumbing (_apply_target_weather / merge_weather) stays available
    # for a future re-evaluation.

    checkpoint = model_path or cfg["model"]["checkpoint"]
    model = load_checkpoint(checkpoint)
    calibrators = load_calibrators(cfg["model"]["calibrators"])
    result = predict_race(df, model, target_season, target_round, calibrators)

    target_rows = df[(df["season"] == target_season) & (df["round"] == target_round)]
    meta = {k: (target_rows[k].iloc[0] if k in target_rows else None)  # type: ignore[reportAttributeAccessIssue]  # target_rows is a boolean-mask slice (Unknown)
            for k in ("race_name", "circuit_id", "date")}
    return {
        "result": result,
        "meta": meta,
        "season": target_season,
        "round": target_round,
        "synthetic": synthetic,
        "verified": not synthetic,
        "calibrated": bool(calibrators),
        "checkpoint": checkpoint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="target season (default: next race)")
    parser.add_argument("--round", type=int, help="target round (required with --season)")
    parser.add_argument("--grid", help="CSV with 'driver_id,grid' for an upcoming race")
    parser.add_argument(
        "--model", help="model checkpoint path (default: config [model] checkpoint)"
    )
    parser.add_argument("--out", help="report path (default: reports/prediction.md)")
    parser.add_argument("--refresh", action="store_true", help="ignore the raw-data cache")
    args = parser.parse_args()

    cfg = load_config()
    try:
        pred = get_prediction(
            season=args.season,
            round_=args.round,
            grid_csv=args.grid,
            refresh=args.refresh,
            cfg=cfg,
            model_path=args.model,
        )
    except ValueError as exc:
        # get_prediction raises ValueError for bad arguments (e.g. --season
        # without --round); keep the CLI error clean, without a traceback.
        raise SystemExit(str(exc)) from None
    result, meta = pred["result"], pred["meta"]
    target_season, target_round = pred["season"], pred["round"]

    report = format_report(result, target_season, target_round, meta,
                           verified=pred["verified"], checkpoint=pred["checkpoint"],
                           calibrated=pred["calibrated"])
    out = Path(args.out or cfg["report"]["prediction"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    # Console: compact ranked grid.
    console = result[["pred_rank", "driver_id", "constructor_id", "expected_points",
                      "p_scored", "p_top3", "p_win"]].copy()
    console["expected_points"] = console["expected_points"].round(2)
    console["p_scored"] = (console["p_scored"] * 100).round(1)
    console["p_top3"] = (console["p_top3"] * 100).round(1)
    console["p_win"] = (console["p_win"] * 100).round(1)
    console = console.rename(
        columns={"p_scored": "p_scored%", "p_top3": "p_top3%", "p_win": "p_win%"}
    )
    print(f"Prediction: {meta.get('race_name', f'Round {target_round}')} "
          f"({target_season} R{target_round}) - {meta.get('circuit_id', '?')} "
          f"{meta.get('date', '')}".rstrip())
    print(console.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
