"""Evaluation of the hurdle model vs. baselines via walk-forward backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1data import F1Client
from features.build import build_dataset
from model.train import (
    HurdleModels,
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
    return df["grid"].map(points_for_position)


def baseline_champ_points(df: pd.DataFrame) -> pd.Series:
    """Predict the points table value of the championship position entering."""
    return df["champ_pos_entering"].map(points_for_position)


def baseline_zero_points(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


# --------------------------------------------------------------------------
# Race-level metrics
# --------------------------------------------------------------------------

def _rank_by(df: pd.DataFrame, score_col: str, tiebreak_col: str) -> pd.Series:
    """Rank drivers within a race by descending score, ascending tiebreak.

    Non-positive tiebreak values (e.g. ``grid=0`` for pit-lane starts, or a
    missing position) are treated as the worst — they sort last, never first.

    The returned Series is aligned to ``df``'s index (rank 1 = best), so it
    can be boolean-indexed against ``df`` and compared across drivers.
    """
    tiebreak = (
        df[tiebreak_col]
        .replace(0, np.inf)
        .fillna(np.inf)
    )
    ranked = (
        df.assign(_tiebreak=tiebreak)
        .sort_values([score_col, "_tiebreak"], ascending=[False, True])
    )
    ranks = pd.Series(range(1, len(df) + 1), index=ranked.index)
    return ranks.reindex(df.index)


def race_metrics(df: pd.DataFrame) -> dict[str, float]:
    """One race's metrics: winner hit, top-3 overlap, spearman, MAE."""
    actual_points = df["points"].to_numpy(dtype=float)
    pred_points = df["pred_points"].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return {}

    # Keep these as Series: boolean indexing a DataFrame with a Series aligns
    # by index, whereas a numpy array would select rows positionally.
    actual_rank = _rank_by(df, "points", "position")
    pred_rank = _rank_by(df, "pred_points", "grid")

    actual_winner_rows = df.loc[df["position"].eq(1), "driver_id"]
    pred_winner_rows = df.loc[pred_rank.eq(1), "driver_id"]
    actual_winner = actual_winner_rows.iloc[0] if not actual_winner_rows.empty else None
    pred_winner = pred_winner_rows.iloc[0] if not pred_winner_rows.empty else None

    top3_actual = set(df.loc[df["position"].between(1, 3), "driver_id"])
    top3_pred = set(df.loc[pred_rank.le(3), "driver_id"])

    corr = spearmanr(pred_rank, actual_rank).statistic
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

def run_backtest(df: pd.DataFrame, quantize: bool = True) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Walk-forward backtest; returns (overall table, per-season tables).

    The model is re-trained for every test season (train = all strictly
    earlier seasons). Metrics are computed per race and then averaged per
    season and overall. Expected points are quantized to the points table by
    default (the deployed output); pass ``quantize=False`` to compare the
    raw continuous expectations.
    """
    df = df.copy()
    df["pred_model"] = np.nan
    df["pred_grid"] = baseline_grid_points(df)
    df["pred_champ"] = baseline_champ_points(df)
    df["pred_zero"] = baseline_zero_points(df)

    season_rows: list[dict] = []
    model_by_season: dict[int, HurdleModels] = {}
    for train, test, season in walk_forward_seasons(df):
        X_train, y_train = prepare(train)
        model = HurdleModels().fit(X_train, y_train)
        model_by_season[season] = model
        X_test, _ = prepare(test)
        test = test.copy()
        pred = model.predict_expected_points(X_test)
        test["pred_model"] = quantize_points(pred) if quantize else pred
        df.loc[test.index, "pred_model"] = test["pred_model"]

        # Metrics are defined per race: rank drivers within one race only.
        for (_, round_), race in test.groupby(["season", "round"]):
            for name, col in (
                ("model", "pred_model"),
                ("grid", "pred_grid"),
                ("championship", "pred_champ"),
                ("zero", "pred_zero"),
            ):
                sub = race.copy()
                sub["pred_points"] = sub[col]
                m = race_metrics(sub)
                m.update({"season": season, "round": round_, "baseline": name})
                season_rows.append(m)

    results = pd.DataFrame(season_rows)
    overall = (
        results.groupby("baseline")[
            ["winner_hit", "top3_overlap", "spearman", "mae"]
        ]
        .mean()
        .reindex(["model", "grid", "championship", "zero"])
        .round(4)
    )

    by_season = {
        name: g.drop(columns=["baseline", "round"])
        .groupby("season")
        .mean()
        .round(4)
        for name, g in results.groupby("baseline")
    }
    return overall, by_season


def _to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without the `tabulate` package."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(row[c]) for c in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep, *body])


def format_tables(overall: pd.DataFrame, by_season: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Backtest (walk-forward)",
        "",
        "Train on all seasons strictly before the test season; evaluate one season at a time.",
        "",
        "## Overall (mean per race across all test seasons)",
        "",
        _to_md(overall),
        "",
    ]
    for name in ("model", "grid", "championship", "zero"):
        table = by_season.get(name)
        if table is None:
            continue
        lines.append(f"## Per season — {name}")
        lines.append("")
        lines.append(_to_md(table))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--dataset", default="data/features.parquet")
    parser.add_argument("--out", default="reports/backtest.md")
    parser.add_argument("--no-quantize", action="store_true",
                        help="keep continuous expected points (deployed output is quantized)")
    args = parser.parse_args()

    client = F1Client(cache_dir=args.cache_dir, refresh=args.refresh)
    df = build_dataset(client, range(args.start, args.end + 1), cache_path=args.dataset)
    overall, by_season = run_backtest(df, quantize=not args.no_quantize)
    print(overall.to_string())
    report = format_tables(overall, by_season)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
