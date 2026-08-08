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

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.build import build_dataset  # noqa: E402
from f1data import F1Client  # noqa: E402
from model.train import (  # noqa: E402
    HurdleModels,
    _joblib_dump,
    _joblib_load,
    prepare,
    walk_forward_seasons,
)

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
    payload = _joblib_load(p)
    if not isinstance(payload, dict) or "calibrators" not in payload:
        raise ValueError(f"{p} is not a calibrator file (missing 'calibrators' key)")
    return payload["calibrators"]


def save_calibrators(calibrators: Dict[str, IsotonicRegression], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _joblib_dump({"calibrators": calibrators}, path)


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
    n_unique = df["p"].nunique()
    if n_unique < 2:
        return pd.DataFrame(
            {"mean_pred": [df["p"].mean()], "observed": [df["y"].mean()],
             "n": [len(df)]}
        ).round(3)
    df["bin"] = pd.cut(df["p"], bins=min(bins, n_unique), include_lowest=True)
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


def summarize(oos: pd.DataFrame, calibrators: Dict[str, IsotonicRegression],
              context: str = "", keep: Optional[set] = None) -> str:
    """Raw-vs-calibrated Brier scores and reliability tables, as text.

    ``context`` is a one-line note describing how ``calibrators`` were fit
    relative to ``oos`` (e.g. a chronological hold-out), so the numbers are
    never mistaken for out-of-calibration-sample evidence. ``keep`` lists the
    targets whose calibrators are actually deployed.
    """
    lines = [
        "# Calibration (isotonic, fit on walk-forward out-of-sample scores)",
        "",
        f"- Evaluation context: {context or 'in-sample (calibrators fit and evaluated on the same rows)'}",
        f"- Deployed: {', '.join(sorted(keep)) if keep else 'none (raw probabilities kept)'}",
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
    # Deployment calibrators: fit on ALL out-of-sample scores (most data).
    calibrators = fit_calibrators(oos)
    save_calibrators(calibrators, args.out)

    # Honest evaluation: fit calibrators on the earlier OOS seasons only and
    # evaluate on the later ones, so the reported Brier deltas are not
    # in-sample for the calibration step.
    seasons = sorted(oos["season"].unique())
    split_at = seasons[len(seasons) * 2 // 3]
    fit_oos = oos[oos["season"] < split_at]
    eval_oos = oos[oos["season"] >= split_at]
    if len(eval_oos) < 200:
        eval_oos = oos
        context = f"in-sample (too few hold-out rows; fit+eval on the same {len(oos)} OOS rows)"
        eval_cal = calibrators
    else:
        eval_cal = fit_calibrators(fit_oos)
        context = (
            f"chronological hold-out: calibrators fit on OOS seasons "
            f"{min(fit_oos['season'])}-{max(fit_oos['season'])}, evaluated on "
            f"seasons {min(eval_oos['season'])}-{max(eval_oos['season'])}"
        )

    # Deploy a calibrator only where it improves hold-out Brier; other targets
    # keep their raw scores.
    keep = {
        target
        for target, score in TARGETS.items()
        if brier(eval_oos[target], eval_cal[target].predict(eval_oos[score]))
        < brier(eval_oos[target], eval_oos[score])
    }
    deployment = {t: calibrators[t] for t in keep}
    save_calibrators(deployment, args.out)
    print(summarize(eval_oos, eval_cal, context=context, keep=keep))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
