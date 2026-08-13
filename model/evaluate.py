"""Evaluation of the hurdle model vs. baselines via walk-forward backtest."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1core.config import load_config
from f1core.reporting import rank_by, to_md
from f1data import F1Client
from features.build import build_dataset
from features.registry import enabled_features, feature_fingerprint
from model.train import (
    HurdleModels,
    model_params,
    points_for_position,
    prepare,
    quantize_points,
    walk_forward_seasons,
)

# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def baseline_grid_points(df: pd.DataFrame) -> pd.Series:
    """Predict the points table value of the grid slot."""
    # pandas .map on an untyped index is Unknown; runtime is always a Series.
    return df["grid"].map(points_for_position)  # type: ignore[reportReturnType]


def baseline_champ_points(df: pd.DataFrame) -> pd.Series:
    """Predict the points table value of the championship position entering."""
    return df["champ_pos_entering"].map(points_for_position)  # type: ignore[reportReturnType]


def baseline_zero_points(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


# --------------------------------------------------------------------------
# Race-level metrics
# --------------------------------------------------------------------------

def race_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """One race's metrics: winner hit, top-3 overlap, spearman, MAE."""
    actual_points = df["points"].to_numpy(dtype=float)
    pred_points = df["pred_points"].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return {}

    # Keep these as Series: boolean indexing a DataFrame with a Series aligns
    # by index, whereas a numpy array would select rows positionally.
    actual_rank = rank_by(df, "points", "position")
    pred_rank = rank_by(df, "pred_points", "grid")

    actual_winner_rows = df.loc[df["position"].eq(1), "driver_id"]
    pred_winner_rows = df.loc[pred_rank.eq(1), "driver_id"]
    actual_winner = actual_winner_rows.iloc[0] if not actual_winner_rows.empty else None
    pred_winner = pred_winner_rows.iloc[0] if not pred_winner_rows.empty else None

    top3_actual = set(df.loc[df["position"].between(1, 3), "driver_id"])
    top3_pred = set(df.loc[pred_rank.le(3), "driver_id"])

    corr = spearmanr(pred_rank, actual_rank).statistic  # type: ignore[reportAttributeAccessIssue]  # scipy stubs type the result as `_`
    if np.isnan(corr):
        corr = 0.0

    return {
        "winner_hit": float(
            pred_winner == actual_winner
            if pred_winner is not None and actual_winner is not None
            else 0.0
        ),
        "top3_overlap": len(top3_actual & top3_pred) / 3.0,
        "spearman": float(corr),
        "mae": float(np.mean(np.abs(pred_points - actual_points))),
    }


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

def _collect_metric_rows(df: pd.DataFrame) -> list[dict]:
    """Per-race metrics for every ``(season, round)`` in ``df``.

    ``df`` must already carry the ``pred_model``/``pred_grid``/
    ``pred_champ``/``pred_zero`` columns (see :func:`run_backtest`).
    """
    rows: list[dict] = []
    for (season, round_), race in df.groupby(["season", "round"]):  # type: ignore[reportGeneralTypeIssues]  # keys are Hashable; runtime is a 2-tuple
        for name, col in (
            ("model", "pred_model"),
            ("grid", "pred_grid"),
            ("championship", "pred_champ"),
            ("zero", "pred_zero"),
        ):
            sub = race.copy()
            sub["pred_points"] = sub[col]
            m = race_metrics(sub)
            m.update({"season": int(season), "round": int(round_), "baseline": name})
            rows.append(m)
    return rows


def run_backtest(
    df: pd.DataFrame,
    quantize: bool = True,
    features: list[str] | None = None,
    model: Any = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest; returns (overall table, per-season tables).

    The default is a walk-forward backtest: the model is re-trained for every
    test season (train = all strictly earlier seasons) on ``features``
    (default: the full feature set). Pass ``model`` (a pre-trained
    :class:`~model.train.HurdleModels` checkpoint) to instead score every
    season with that fixed model — the "how good is *this* model on a season
    range" mode (not out-of-sample w.r.t. its own training data). Metrics are
    computed per race and then averaged per season and overall. Expected
    points are quantized to the points table by default (the deployed
    output); pass ``quantize=False`` to compare the raw continuous
    expectations.
    """
    df = df.copy()
    df["pred_model"] = np.nan
    df["pred_grid"] = baseline_grid_points(df)
    df["pred_champ"] = baseline_champ_points(df)
    df["pred_zero"] = baseline_zero_points(df)

    if model is not None:
        # Deployed-checkpoint mode: score every row with the fixed model.
        X_all, _ = prepare(df, features)
        pred = model.predict_expected_points(X_all)
        df["pred_model"] = quantize_points(pred) if quantize else pred
        season_rows = _collect_metric_rows(df)
    else:
        season_rows: list[dict] = []
        params = model_params()  # read [model.params] once, outside the loop
        for train, test, _season in walk_forward_seasons(df):
            X_train, y_train = prepare(train, features)
            fitted = HurdleModels(params=params).fit(X_train, y_train)
            X_test, _ = prepare(test, features)
            test = test.copy()
            pred = fitted.predict_expected_points(X_test)
            test["pred_model"] = quantize_points(pred) if quantize else pred
            df.loc[test.index, "pred_model"] = test["pred_model"]
            season_rows.extend(_collect_metric_rows(test))

    results = pd.DataFrame(season_rows)
    overall = (
        results.groupby("baseline")[
            ["winner_hit", "top3_overlap", "spearman", "mae"]
        ]
        .mean()
        .reindex(["model", "grid", "championship", "zero"])  # type: ignore[reportAttributeAccessIssue]  # groupby() is Unknown without pandas stubs
        .round(4)
    )

    by_season = {
        name: g.drop(columns=["baseline", "round"])
        .groupby("season")
        .mean()
        .round(4)  # type: ignore[reportAttributeAccessIssue]  # same Unknown chain as reindex above
        for name, g in results.groupby("baseline")
    }
    return overall, by_season  # type: ignore[reportReturnType]  # Unknown unions from the groupby chain


def _metric_row(row: pd.Series) -> dict[str, float]:
    """One metric row as JSON-safe floats (pandas values are often np scalars)."""
    return {str(metric): float(value) for metric, value in row.items()}


def backtest_snapshot(
    overall: pd.DataFrame, by_season: dict[str, pd.DataFrame]
) -> dict:
    """JSON-safe snapshot of the backtest for the web dashboard.

    ``overall`` is mean metrics per baseline; ``by_season`` is the same per
    (baseline, season). Keys are strings (JSON object keys must be).
    """
    return {
        "overall": {str(name): _metric_row(overall.loc[name]) for name in overall.index},
        "by_season": {
            str(name): {
                str(season): _metric_row(row) for season, row in table.iterrows()
            }
            for name, table in by_season.items()
        },
    }


def format_tables(overall: pd.DataFrame, by_season: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Backtest (walk-forward)",
        "",
        "Train on all seasons strictly before the test season; evaluate one season at a time.",
        "",
        "## Overall (mean per race across all test seasons)",
        "",
        to_md(overall),
        "",
    ]
    for name in ("model", "grid", "championship", "zero"):
        table = by_season.get(name)
        if table is None:
            continue
        lines.append(f"## Per season — {name}")
        lines.append("")
        lines.append(to_md(table))
        lines.append("")
    return "\n".join(lines)


def _fixed_model_backtest(
    df: pd.DataFrame, checkpoint: str, quantize: bool, log
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score every season with a saved checkpoint, using its own features.

    Raises :class:`ValueError` when the checkpoint carries no feature list
    (retrain with the current ``model/train.py`` first).
    """
    from model.train import checkpoint_meta, load_checkpoint

    meta = checkpoint_meta(checkpoint)
    if not meta or "features" not in meta:
        raise ValueError(
            f"checkpoint {checkpoint} carries no feature list; retrain it "
            "with the current model/train.py before backtesting"
        )
    feats = list(meta["features"])
    model = load_checkpoint(checkpoint, expected=feats)
    log(
        f"Scoring with model {checkpoint} ({len(feats)} features, "
        f"fp {feature_fingerprint(feats)})"
    )
    return run_backtest(df, quantize=quantize, features=feats, model=model)


def run(
    *,
    start: int = 2010,
    end: int = 2026,
    refresh: bool = False,
    cache_dir: str = "data/raw",
    dataset: str = "data/features.parquet",
    out: str = "reports/backtest.md",
    out_json: str = "reports/backtest.json",
    quantize: bool = True,
    use_checkpoint: bool = False,
    model_path: str | None = None,
    model_paths: Sequence[str] = (),
    enable_features: Sequence[str] = (),
    disable_features: Sequence[str] = (),
    cfg: dict | None = None,
    log=None,
) -> dict:
    """Run the backtest end-to-end, write report + JSON snapshot, return a summary.

    ``log`` is an optional progress callback (web job runner); all arguments are
    keyword-only so the CLI and the dashboard share one code path. The returned
    dict is JSON-safe and carries the full ``backtest_snapshot`` so the web
    runner can refresh the dashboard views without re-reading the file.

    ``model_path`` scores every season in the range with that saved checkpoint,
    using the feature set the checkpoint was trained on (the feature toggles
    are ignored — a model is tested with its own features). ``use_checkpoint``
    is the legacy alias for the *deployed* checkpoint (``[model] checkpoint``);
    ``model_path`` takes precedence when both are given.

    ``model_paths`` compares several saved checkpoints on one shared dataset:
    each is scored with its own feature set, the snapshot gains a
    ``"models"`` key (``{checkpoint stem: {overall, by_season}}``), and the
    primary ``overall``/``by_season`` tables come from the first checkpoint.
    """
    log = log or (lambda msg: print(msg, flush=True))
    cfg = cfg or load_config()
    client = F1Client(cache_dir=cache_dir, refresh=refresh)
    log(f"Building dataset {start}-{end} ...")
    df = build_dataset(client, range(start, end + 1), cache_path=dataset)
    feats = enabled_features(
        cfg, enable=list(enable_features), disable=list(disable_features)
    )
    checkpoint: str | None = None
    compared: dict[str, dict] = {}
    if model_paths:
        from model.train import checkpoint_meta

        paths = [str(p) for p in model_paths]
        overall: pd.DataFrame | None = None
        by_season: dict[str, pd.DataFrame] | None = None
        for path in paths:
            overall_i, by_season_i = _fixed_model_backtest(df, path, quantize, log)
            compared[Path(path).stem] = backtest_snapshot(overall_i, by_season_i)
            if overall is None:
                # The primary tables show the first compared model vs the baselines.
                overall, by_season = overall_i, by_season_i
        checkpoint = paths[0]
        feats = list(checkpoint_meta(checkpoint)["features"])
        assert overall is not None and by_season is not None
    elif model_path:
        from model.train import checkpoint_meta

        checkpoint = model_path
        overall, by_season = _fixed_model_backtest(df, checkpoint, quantize, log)
        feats = list(checkpoint_meta(checkpoint)["features"])
    elif use_checkpoint:
        from model.train import load_checkpoint

        checkpoint = cfg["model"]["checkpoint"]
        model = load_checkpoint(checkpoint, expected=feats)
        log(
            f"Scoring with deployed checkpoint {checkpoint} "
            f"({len(feats)} features, fp {feature_fingerprint(feats)})"
        )
        overall, by_season = run_backtest(
            df, quantize=quantize, features=feats, model=model
        )
    else:
        log(
            f"Running walk-forward backtest with {len(feats)} features "
            f"(fp {feature_fingerprint(feats)})"
        )
        overall, by_season = run_backtest(df, quantize=quantize, features=feats)

    report = format_tables(overall, by_season)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(report, encoding="utf-8")
    log(f"Wrote {out_p}")

    snapshot = backtest_snapshot(overall, by_season)
    if compared:
        snapshot["models"] = compared
    json_out = Path(out_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    log(f"Wrote {json_out}")

    overall_rows = {
        str(name): _metric_row(overall.loc[name]) for name in overall.index
    }
    return {
        "overall": overall_rows,
        "features": feats,
        "n_features": len(feats),
        "fingerprint": feature_fingerprint(feats),
        "checkpoint": checkpoint,
        "quantize": quantize,
        "report": out,
        "snapshot": snapshot,
        "models": sorted(compared),
    }


