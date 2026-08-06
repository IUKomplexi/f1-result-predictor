"""Probability calibration for the hurdle model's companion classifiers.

The gradient-boosted classifiers' raw probabilities are overconfident (a
common trait of gradient boosting). This module fits *isotonic* calibrators
on genuinely out-of-sample predictions from the walk-forward backtest, so the
reported P(top-10) / P(top-3) / P(win) become trustworthy probabilities.

Usage::

    python model/calibrate.py            # fit + save data/model/calibrators.joblib

The saved calibrators are consumed by ``predict.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.build import build_dataset  # noqa: E402
from f1data import F1Client  # noqa: E402
from model.train import HurdleModels, prepare, walk_forward_seasons  # noqa: E402

TARGETS = {"scored": "p_scored", "top3": "p_top3", "win": "p_win"}


# --------------------------------------------------------------------------
# Out-of-sample score collection
# --------------------------------------------------------------------------

def collect_oos_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Raw model probabilities on held-out races via walk-forward.

    Each test-season row carries the raw scores of a model trained only on
    strictly earlier seasons, plus the actual outcome labels.
    """
    chunks = []
    for train, test, _ in walk_forward_seasons(df):
        X_train, y_train = prepare(train)
        model = HurdleModels().fit(X_train, y_train)
        X_test, _ = prepare(test)
        probs = model.predict_probs(X_test)
        chunks.append(
            pd.DataFrame(
                {
                    "p_scored": probs["p_scored"],
                    "p_top3": probs["p_top3"],
                    "p_win": probs["p_win"],
                    "scored": test["scored"].to_numpy(dtype=int),
                    "top3": test["top3"].to_numpy(dtype=int),
                    "win": test["win"].to_numpy(dtype=int),
                    "season": test["season"].to_numpy(),
                },
                index=test.index,
            )
        )
    if not chunks:
        raise ValueError("no walk-forward splits produced (too few seasons?)")
    return pd.concat(chunks)


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------

def fit_calibrators(oos: pd.DataFrame) -> Dict[str, IsotonicRegression]:
    """Fit an isotonic regressor per target on out-of-sample scores."""
    calibrators: Dict[str, IsotonicRegression] = {}
    for target, score in TARGETS.items():
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oos[score].to_numpy(dtype=float), oos[target].to_numpy(dtype=float))
        calibrators[target] = iso
    return calibrators


def apply_calibration(probs: Dict[str, np.ndarray],
                      calibrators: Dict[str, IsotonicRegression]) -> Dict[str, np.ndarray]:
    """Map raw score arrays to calibrated probabilities (keys: p_scored/...)."""
    out = dict(probs)
    for target, score in TARGETS.items():
        if target in calibrators:
            out[score] = calibrators[target].predict(np.asarray(probs[score], dtype=float))
    return out


def load_calibrators(path: str | Path) -> Optional[Dict[str, IsotonicRegression]]:
    """Load calibrators, or None when the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    payload = joblib.load(p)
    if not isinstance(payload, dict) or "calibrators" not in payload:
        raise ValueError(f"{p} is not a calibrator file (missing 'calibrators' key)")
    return payload["calibrators"]


def save_calibrators(calibrators: Dict[str, IsotonicRegression], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrators": calibrators}, path)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def brier(y_true, y_prob) -> float:
    """Brier score: mean squared error of a probability forecast."""
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_prob, dtype=float)) ** 2))


def reliability_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Binned mean prediction vs observed frequency (reliability diagram)."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                       "p": np.asarray(y_prob, dtype=float)})
    df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    table = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return table.round(3)


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


def summarize(oos: pd.DataFrame, calibrators: Dict[str, IsotonicRegression]) -> str:
    """Raw-vs-calibrated Brier scores and reliability tables, as text."""
    lines = [
        "# Calibration (isotonic, fit on walk-forward out-of-sample scores)",
        "",
        "| target | brier_raw | brier_calibrated | delta |",
        "| --- | --- | --- | --- |",
    ]
    for target, score in TARGETS.items():
        y = oos[target].to_numpy(dtype=float)
        raw = oos[score].to_numpy(dtype=float)
        cal = calibrators[target].predict(raw)
        lines.append(
            f"| {target} | {brier(y, raw):.4f} | {brier(y, cal):.4f} | "
            f"{brier(y, cal) - brier(y, raw):+.4f} |"
        )
    lines.append("")
    lines.append("## Reliability - calibrated")
    lines.append("")
    for target, score in TARGETS.items():
        cal = calibrators[target].predict(oos[score].to_numpy(dtype=float))
        lines.append(f"### {target}")
        lines.append("")
        lines.append(_to_md(reliability_table(oos[target], cal)))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--dataset", default="data/features.parquet")
    parser.add_argument("--out", default="data/model/calibrators.joblib")
    args = parser.parse_args()

    client = F1Client(cache_dir=args.cache_dir, refresh=args.refresh)
    df = build_dataset(client, range(args.start, args.end + 1), cache_path=args.dataset)

    oos = collect_oos_scores(df)
    calibrators = fit_calibrators(oos)
    save_calibrators(calibrators, args.out)
    print(summarize(oos, calibrators))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
