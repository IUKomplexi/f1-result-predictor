"""Feature engineering for the F1 result predictor.

Every row in the dataset is one (race, driver) start. All features are
computed from information available *before* the race starts (grid,
qualifying, prior form, championship position entering the race), and the
target is the points the driver actually scored in that race.

Leakage safety: every rolling / cumulative feature uses ``shift(1)`` so it
only ever sees races strictly before the target race. The unit test
``test_no_leakage`` verifies this invariant against a small dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from f1data import F1Client, fetch_season
from f1data.fetchers import is_classified

logger = logging.getLogger(__name__)

FORM_WINDOW = 5          # rolling window (races) for driver/constructor form
CIRCUIT_WINDOW = 3       # prior visits at the same circuit to consider

NUMERIC_FEATURES = [
    "grid",
    "qual_pos",
    "grid_qual_gap",
    "season",
    "round",
    "is_sprint_round",
    "n_prior_races",
    "team_tenure",
    "team_switch",
    "driver_prev_finish_mean",
    "driver_prev_points_mean",
    "driver_prev_points_sum",
    "last_race_points",
    "driver_prev_dnf_rate",
    "driver_wins_prior",
    "team_prev_points_mean",
    "team_prev_pos_mean",
    "team_wins_prior",
    "circuit_prev_finish_mean",
    "circuit_prev_points_mean",
    "finish_gap_vs_teammate",
    "qual_gap_vs_teammate",
    "champ_points_entering",
    "champ_pos_entering",
    "constructor_champ_pos_entering",
    "season_driver_pts_per_race",
    "season_team_pts_per_race",
]

CATEGORICAL_FEATURES = ["driver_id", "constructor_id", "circuit_id", "points_era"]

META_COLUMNS = [
    "season",
    "round",
    "race_name",
    "date",
    "driver_id",
    "constructor_id",
    "grid",
    "position",
]


# --------------------------------------------------------------------------
# Rolling helpers (strictly backward-looking)
# --------------------------------------------------------------------------

def _rolling_mean(s: pd.Series, window: int = FORM_WINDOW) -> pd.Series:
    # pandas rolling chains resolve to Unknown without stubs; runtime is a Series.
    return s.shift(1).rolling(window, min_periods=1).mean()  # type: ignore[reportReturnType]


def _rolling_sum(s: pd.Series, window: int = FORM_WINDOW) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).sum()  # type: ignore[reportReturnType]


def _career_wins(s: pd.Series) -> pd.Series:
    """Cumulative wins strictly before the current race."""
    return (s == 1).astype(float).shift(1).cumsum().fillna(0.0)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def assemble(season_datas: Sequence[dict]) -> pd.DataFrame:
    """Flatten ``fetch_season`` output into a tidy per-start DataFrame.

    Adds ``date``/``circuit_id``/``race_name``/``is_sprint_round`` from the
    calendar, ``qual_pos`` from qualifying, and ``sprint_points`` from the
    sprint results (0 when the round has no sprint).
    """
    frames: list[pd.DataFrame] = []
    for data in season_datas:
        calendar = pd.DataFrame(data["calendar"])[
            ["round", "date", "circuit_id", "race_name", "is_sprint_round"]
        ]

        rows: list[dict] = []
        for round_, results in data["results"].items():
            meta = calendar[calendar["round"] == round_]
            if meta.empty:  # type: ignore[reportAttributeAccessIssue]  # boolean-mask slice is Unknown
                logger.warning("round %s missing from calendar", round_)
                continue
            m = meta.iloc[0]  # type: ignore[reportAttributeAccessIssue]  # same Unknown slice
            for r in results:
                rows.append(
                    {
                        "season": r["season"],
                        "round": r["round"],
                        "position": r["position"],
                        "grid": r["grid"],
                        "points": float(r["points"]),
                        "status": r["status"],
                        "driver_id": r["driver_id"],
                        "constructor_id": r["constructor_id"],
                        "date": m["date"],
                        "circuit_id": m["circuit_id"],
                        "race_name": m["race_name"],
                        "is_sprint_round": bool(m["is_sprint_round"]),
                    }
                )
        results_df = pd.DataFrame(rows)

        # Sprint points (count toward championship entering the main race).
        sprint_rows: list[dict] = []
        for _, sprints in data["sprints"].items():
            for r in sprints:
                sprint_rows.append(
                    {"season": r["season"], "round": r["round"],
                     "driver_id": r["driver_id"], "sprint_points": r["points"]}
                )
        if sprint_rows:
            sprints_df = pd.DataFrame(sprint_rows)
            results_df = results_df.merge(
                sprints_df, on=["season", "round", "driver_id"], how="left"
            )
            results_df["sprint_points"] = results_df["sprint_points"].fillna(0.0)
        else:
            results_df["sprint_points"] = 0.0

        # Qualifying position (fall back to grid when missing).
        qual_rows: list[dict] = []
        for _, quals in data["qualifying"].items():
            for r in quals:
                qual_rows.append(
                    {"season": r["season"], "round": r["round"],
                     "driver_id": r["driver_id"], "qual_pos": r["position"]}
                )
        if qual_rows:
            quals_df = pd.DataFrame(qual_rows)
            results_df = results_df.merge(
                quals_df, on=["season", "round", "driver_id"], how="left"
            )
        else:
            results_df["qual_pos"] = np.nan
        results_df["qual_pos"] = results_df["qual_pos"].fillna(results_df["grid"])

        frames.append(results_df)

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-race features and target columns. Never touches future data."""
    out = df.copy().sort_values(["date", "round"]).reset_index(drop=True)
    out = _add_baseline(out)
    out = _add_driver_history(out)
    out = _add_constructor_history(out)
    out = _add_circuit_history(out)
    out = _add_teammate_gaps(out)
    out = _add_championship(out)
    out = _add_era_and_targets(out)
    return out


def _add_baseline(out: pd.DataFrame) -> pd.DataFrame:
    """Normalize input columns (points, sprint, qualifying, status flags)."""
    out["points"] = out["points"].astype(float)
    if "sprint_points" in out.columns:
        out["sprint_points"] = out["sprint_points"].fillna(0.0).astype(float)
    else:
        out["sprint_points"] = 0.0
    out["race_points"] = out["points"] + out["sprint_points"]
    if "qual_pos" not in out.columns:
        out["qual_pos"] = out["grid"]
    out["grid_qual_gap"] = out["grid"] - out["qual_pos"]
    if "is_sprint_round" not in out.columns:
        out["is_sprint_round"] = 0  # data assembled without a calendar
    out["is_sprint_round"] = out["is_sprint_round"].astype(float)
    out["is_dnf"] = ~out["status"].map(is_classified).fillna(False)
    # finish_pos: numeric position, NaN when unknown (e.g. missing).
    out["finish_pos"] = out["position"].where(out["position"].fillna(0) > 0)
    return out


def _add_driver_history(out: pd.DataFrame) -> pd.DataFrame:
    """Driver career history (across all seasons, strictly prior)."""
    g = out.groupby("driver_id", sort=False)
    out["n_prior_races"] = g.cumcount()
    out["team_tenure"] = out.groupby(["driver_id", "constructor_id"], sort=False).cumcount()
    prev_team = g["constructor_id"].shift(1)
    out["team_switch"] = ((out["constructor_id"] != prev_team) & prev_team.notna()).astype(float)
    out["last_race_points"] = g["points"].shift(1)
    out["driver_prev_finish_mean"] = g["finish_pos"].transform(_rolling_mean)
    out["driver_prev_points_mean"] = g["points"].transform(_rolling_mean)
    out["driver_prev_points_sum"] = g["points"].transform(_rolling_sum)
    out["driver_prev_dnf_rate"] = g["is_dnf"].transform(
        lambda s: s.astype(float).shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )
    out["driver_wins_prior"] = g["position"].transform(_career_wins)
    return out


def _add_constructor_history(out: pd.DataFrame) -> pd.DataFrame:
    """Constructor history aggregated per race (both cars).

    One driver's same-race result never leaks into the other driver's
    features.
    """
    team_race = out.groupby(["constructor_id", "season", "round"], sort=False).agg(
        team_points=("points", "sum"),
        team_sprint_points=("sprint_points", "sum"),
        team_finish_pos=("finish_pos", "mean"),
        team_wins=("position", lambda s: int((s == 1).sum())),
    ).reset_index()
    team_race["team_race_points"] = team_race["team_points"] + team_race["team_sprint_points"]
    team_race = team_race.sort_values(["season", "round"]).reset_index(drop=True)

    g = team_race.groupby("constructor_id", sort=False)
    team_race["team_prev_points_mean"] = g["team_points"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )
    team_race["team_prev_pos_mean"] = g["team_finish_pos"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )
    team_race["team_wins_prior"] = g["team_wins"].transform(
        lambda s: s.shift(1).cumsum().fillna(0.0)
    )

    # Constructor championship position entering the race.
    gs = team_race.groupby(["season", "constructor_id"], sort=False)
    team_race["season_team_n_prior"] = gs.cumcount()
    team_race["season_team_pts_entering"] = (
        gs["team_race_points"].transform(lambda s: s.shift(1).cumsum().fillna(0.0))
        + team_race["team_sprint_points"]
    )
    team_race["season_team_pts_per_race"] = np.where(
        team_race["season_team_n_prior"] > 0,
        team_race["season_team_pts_entering"] / team_race["season_team_n_prior"].clip(lower=1),
        np.nan,
    )
    # Constructor championship rank entering the race (per round, best first).
    team_race["constructor_champ_pos_entering"] = (
        team_race.groupby(["season", "round"])["season_team_pts_entering"]
        .rank(ascending=False, method="min")
    )

    out = out.merge(
        team_race[
            [
                "constructor_id", "season", "round",
                "team_prev_points_mean", "team_prev_pos_mean", "team_wins_prior",
                "season_team_pts_entering", "season_team_pts_per_race",
                "constructor_champ_pos_entering",
            ]
        ],
        on=["constructor_id", "season", "round"],
        how="left",
    )
    return out


def _add_circuit_history(out: pd.DataFrame) -> pd.DataFrame:
    """Prior visits at the same circuit per driver."""
    g = out.groupby(["driver_id", "circuit_id"], sort=False)
    out["circuit_prev_finish_mean"] = g["finish_pos"].transform(
        lambda s: s.shift(1).rolling(CIRCUIT_WINDOW, min_periods=1).mean()
    )
    out["circuit_prev_points_mean"] = g["points"].transform(
        lambda s: s.shift(1).rolling(CIRCUIT_WINDOW, min_periods=1).mean()
    )
    out["circuit_n_prior"] = g.cumcount()
    return out


def _teammate_gap(out: pd.DataFrame, value_col: str, out_col: str) -> None:
    """Rolling gap to the teammate for ``value_col`` (strictly prior).

    The raw gap is computed within the same race (own value minus the
    teammates' mean), then rolled over *prior* races only via shift(1), so
    the current race's teammate result never leaks into this race's feature.
    """
    grp = out.groupby(["season", "round", "constructor_id"], sort=False)[value_col]
    others_sum = grp.transform("sum") - out[value_col]
    others_n = grp.transform("size") - 1
    out["_raw_teammate_gap"] = (
        (out[value_col] - others_sum / others_n.replace(0, np.nan))
        .where(others_n > 0)
    )
    gd = out.groupby("driver_id", sort=False)
    out[out_col] = gd["_raw_teammate_gap"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )
    del out["_raw_teammate_gap"]


def _add_teammate_gaps(out: pd.DataFrame) -> pd.DataFrame:
    """Teammate-relative performance (strictly prior)."""
    _teammate_gap(out, "finish_pos", "finish_gap_vs_teammate")
    _teammate_gap(out, "qual_pos", "qual_gap_vs_teammate")
    return out


def _add_championship(out: pd.DataFrame) -> pd.DataFrame:
    """Championship position entering the race.

    Includes the current round's sprint points (known before the main race),
    plus all main-race + sprint points from earlier rounds of the season.
    """
    g = out.groupby(["season", "driver_id"], sort=False)
    out["champ_points_entering"] = (
        g["race_points"].transform(lambda s: s.shift(1).cumsum().fillna(0.0))
        + out["sprint_points"]
    )
    out["season_n_prior"] = g.cumcount()
    out["season_driver_pts_per_race"] = np.where(
        out["season_n_prior"] > 0,
        out["champ_points_entering"] / out["season_n_prior"].clip(lower=1),
        np.nan,
    )
    out["champ_pos_entering"] = out.groupby(["season", "round"])[
        "champ_points_entering"
    ].rank(ascending=False, method="min")
    return out


def _add_era_and_targets(out: pd.DataFrame) -> pd.DataFrame:
    """Points era and target columns."""
    out["points_era"] = np.where(out["season"] >= 2019, "post2019", "pre2019")
    out["scored"] = out["points"] > 0
    out["top3"] = out["position"].between(1, 3)
    out["win"] = out["position"].eq(1)
    return out


# --------------------------------------------------------------------------
# Dataset builder + coverage report
# --------------------------------------------------------------------------

def _dataset_fingerprint() -> str:
    """Fingerprint of the *computed* feature set, stored in the parquet cache.

    ``add_features`` always computes every registry feature, so the cache
    content only depends on the full computed set — not on the enabled
    selection (that lives in the model-checkpoint fingerprint). Lazy import:
    ``features/registry.py`` imports this module.
    """
    from features.registry import all_feature_ids, feature_fingerprint

    return feature_fingerprint(all_feature_ids())


def _read_cache_fingerprint(path: Path) -> str | None:
    """Feature fingerprint embedded in a cached parquet (None when absent)."""
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(path)
        fp = (schema.metadata or {}).get(b"feature_fingerprint")
        return fp.decode("utf-8") if fp else None
    except Exception:  # noqa: BLE001 - any read failure ⇒ "no fingerprint" ⇒ rebuild
        return None


def _write_parquet(df: pd.DataFrame, path: Path, fingerprint: str) -> None:
    """Write the parquet cache with the feature fingerprint in its metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"feature_fingerprint"] = fingerprint.encode("utf-8")
    pq.write_table(table.replace_schema_metadata(meta), path)


def _raw_cache_newer_than(client: F1Client, cache: Path) -> bool:
    """True when any cached raw response is newer than the parquet cache.

    ``f1 fetch`` after a race weekend adds raw JSON without changing the
    feature fingerprint or the season coverage, so the parquet would otherwise
    look valid and be silently stale. Comparing file mtimes closes that gap:
    new raw data forces a dataset rebuild.
    """
    cache_dir = getattr(client, "cache_dir", None)
    if not cache_dir:
        return False
    raw = Path(cache_dir)
    try:
        if not raw.is_dir():
            return False
        newest = max((p.stat().st_mtime for p in raw.rglob("*.json")), default=0.0)
        return cache.exists() and newest > cache.stat().st_mtime
    except OSError:
        return False


def build_dataset(
    client: F1Client,
    seasons: Iterable[int],
    cache_path: str | Path = "data/features.parquet",
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch (or load cached) features for a range of seasons.

    The assembled+featured dataset is cached as parquet at ``cache_path``;
    pass ``refresh=True`` to rebuild from the API.
    """
    seasons = list(seasons)
    cache = Path(cache_path)
    # Corrupt parquet raises pyarrow.ArrowInvalid, which is not a ValueError
    # in every version; treat any cache-read failure as "rebuild".
    try:
        from pyarrow.lib import ArrowInvalid
    except ImportError:  # pragma: no cover - pyarrow is a hard dependency
        ArrowInvalid = ValueError
    # Feature-version validation: if the cached parquet lacks any currently
    # defined feature column (or was built from a different feature set, per
    # the embedded fingerprint), it is stale and must be rebuilt (silently
    # using a stale feature set would invalidate every downstream result).
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    fingerprint = _dataset_fingerprint()
    if not refresh and cache.exists():
        try:
            cached = pd.read_parquet(cache)
            cached_seasons = set(cached["season"]) if "season" in cached else set()
            if (not _raw_cache_newer_than(client, cache)
                    and set(required) <= set(cached.columns)
                    and cached_seasons >= set(seasons)
                    and _read_cache_fingerprint(cache) == fingerprint):
                logger.info("Loading cached dataset from %s", cache)
                df = cached
            else:
                logger.warning(
                    "Cached dataset %s is stale (missing features, seasons, "
                    "newer raw data, or feature fingerprint); rebuilding",
                    cache,
                )
                df = _build_fresh(client, seasons, cache, fingerprint)
        except (OSError, ValueError, ImportError, ArrowInvalid):
            logger.warning("Unreadable cached dataset %s; rebuilding", cache)
            df = _build_fresh(client, seasons, cache, fingerprint)
    else:
        df = _build_fresh(client, seasons, cache, fingerprint)
    return df


def _build_fresh(
    client: F1Client, seasons: list[int], cache: Path, fingerprint: str
) -> pd.DataFrame:
    datas = [fetch_season(client, s) for s in seasons]
    df = add_features(assemble(datas))
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, cache, fingerprint)
        logger.info("Wrote dataset (%d rows) to %s", len(df), cache)
    except OSError:
        logger.warning("Could not write dataset cache %s", cache)
    return df
