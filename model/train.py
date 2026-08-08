"""Hurdle models for predicting points scored per race, with walk-forward training."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_dataset  # noqa: E402
from f1data import F1Client  # noqa: E402

logger = logging.getLogger(__name__)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Classic top-10 points table (fastest-lap point is part of the target data
# and is not needed for the baselines).
POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
# All values the points target can take (incl. the fastest-lap point).
QUANTIZED_POINTS = np.array([0, 1, 2, 4, 6, 8, 10, 12, 15, 18, 25, 26], dtype=float)


def points_for_position(position) -> float:
    try:
        return float(POINTS_TABLE.get(int(position), 0.0))
    except (TypeError, ValueError):
        return 0.0


def quantize_points(values) -> np.ndarray:
    """Round continuous expected points to the nearest points-table value.

    The grid baseline wins on MAE partly because it predicts discrete table
    values while the model predicts smoothed continuous expectations. This
    post-processing is the deployed default (see model/evaluate.py --no-quantize
    to compare). NaN inputs map to 0.0.
    """
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    idx = np.argmin(np.abs(QUANTIZED_POINTS[None, :] - values[:, None]), axis=1)
    return QUANTIZED_POINTS[idx]


# Default gradient-boosting hyperparameters (see model/search.py to tune).
# Chosen by walk-forward-validated search (model/search.py, test seasons
# <= 2019) with the full feature set incl. teammate-relative features.
DEFAULT_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "l2_regularization": 1.0,
    "min_samples_leaf": 20,
}


def _clf(seed: int, params: Optional[Dict[str, Any]] = None) -> HistGradientBoostingClassifier:
    p = {**DEFAULT_PARAMS, **(params or {})}
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=seed,
        **p,
    )


def _reg(seed: int, params: Optional[Dict[str, Any]] = None) -> HistGradientBoostingRegressor:
    p = {**DEFAULT_PARAMS, **(params or {})}
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        random_state=seed,
        **p,
    )


def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a featured dataset into (X, y).

    Categorical features are converted to pandas ``category`` dtype so the
    gradient-boosted models treat them as categorical (unknown categories at
    prediction time become missing values, which HGB handles natively).
    """
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = pd.DataFrame(
        {
            "points": df["points"].astype(float),
            "scored": (df["points"] > 0).astype(int),
            "top3": df["top3"].astype(int),
            "win": df["win"].astype(int),
        },
        index=df.index,
    )
    return X, y


class _ConstantProb:
    """Binary classifier stand-in for a constant training target.

    When every row (or none) has the positive class, gradient boosting has
    no signal to learn; newer sklearn versions can return degenerate
    probabilities. This wrapper returns the training base rate instead.
    """

    def __init__(self, prob: float) -> None:
        self.prob = float(prob)

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


class _ConstantPoints:
    """Regressor stand-in when the scored subset is empty (degenerate)."""

    def __init__(self, log_points: float = 0.0) -> None:
        self.log_points = log_points

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.log_points)


def _fit_binary(clf, X, y) -> Any:
    """Fit a binary classifier, falling back to a constant when y is flat."""
    if pd.Series(y).nunique() < 2:
        return _ConstantProb(float(pd.Series(y).mean()))
    clf.fit(X, y)
    return clf


class HurdleModels:
    """Zero-inflated points model:

    ``E[points] = P(scored) * E[points | scored]``
    plus companion classifiers for P(top-3) and P(win).
    """

    def __init__(self, seed: int = 42, params: Optional[Dict[str, Any]] = None) -> None:
        self.seed = seed
        self.params = dict(params or {})
        self.scored: Any = _clf(seed, self.params)
        self.top3: Any = _clf(seed + 1, self.params)
        self.win: Any = _clf(seed + 2, self.params)
        self.points_if_scored: Any = _reg(seed + 3, self.params)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "HurdleModels":
        self.scored = _fit_binary(self.scored, X, y["scored"])
        self.top3 = _fit_binary(self.top3, X, y["top3"])
        self.win = _fit_binary(self.win, X, y["win"])
        mask = y["scored"].to_numpy() == 1
        if mask.sum() == 0:
            self.points_if_scored = _ConstantPoints()
        else:
            self.points_if_scored.fit(X[mask], np.log1p(y["points"].to_numpy()[mask]))
        return self

    def predict_expected_points(self, X: pd.DataFrame) -> np.ndarray:
        """E[points] = P(scored) * E[points | scored]."""
        p_scored = self.scored.predict_proba(X)[:, 1]
        exp_points = np.expm1(self.points_if_scored.predict(X))
        return p_scored * np.clip(exp_points, 0.0, None)

    def predict_probs(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "p_scored": self.scored.predict_proba(X)[:, 1],
            "p_top3": self.top3.predict_proba(X)[:, 1],
            "p_win": self.win.predict_proba(X)[:, 1],
        }


def walk_forward_seasons(df: pd.DataFrame, min_train_seasons: int = 3):
    """Yield (train_df, test_df, test_season) in chronological order.

    Train = every season strictly before the test season, so no test-season
    information (including circuit history) is available during training.
    """
    seasons = sorted(df["season"].unique())
    for test_season in seasons:
        train = df[df["season"] < test_season]
        if train["season"].nunique() < min_train_seasons:
            continue
        yield train, df[df["season"] == test_season], test_season


def train_final_model(df: pd.DataFrame) -> HurdleModels:
    """Train on every season (used for live prediction)."""
    X, y = prepare(df)
    return HurdleModels().fit(X, y)


def _joblib_dump(obj: Any, path: Path) -> None:
    """``joblib.dump`` with a targeted suppression of a numpy 2.5 deprecation.

    joblib's ``numpy_pickle`` still restores array shape by assignment
    (``array.shape = ...``), which numpy 2.5 deprecated; the upstream fix is
    unreleased as of joblib 1.5.3. Suppress only DeprecationWarnings raised
    inside joblib, so checkpoint I/O stays clean under ``-W error``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="joblib"
        )
        joblib.dump(obj, path)


def _joblib_load(path: Path) -> Any:
    """``joblib.load`` with the same joblib/numpy 2.5 suppression as dump."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="joblib"
        )
        return joblib.load(path)


def save_checkpoint(models: HurdleModels, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # When train.py is executed as a script, classes are defined in
    # ``__main__``; a pickled reference to ``__main__.HurdleModels`` cannot be
    # resolved by other processes (e.g. predict.py). Alias the running module
    # under the canonical name and repoint the classes so the checkpoint
    # loads anywhere.
    if __name__ == "__main__":
        sys.modules["model.train"] = sys.modules["__main__"]
    for cls in (HurdleModels, _ConstantProb, _ConstantPoints):
        cls.__module__ = "model.train"
    _joblib_dump({"models": models, "features": FEATURES}, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path: str | Path) -> HurdleModels:
    payload = _joblib_load(Path(path))
    stored = list(payload.get("features", []))
    if stored != list(FEATURES):
        raise ValueError(
            "checkpoint feature set does not match the current feature set; "
            "retrain with model/train.py"
        )
    return payload["models"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--dataset", default="data/features.parquet")
    parser.add_argument("--out", default="data/model/hurdle.joblib")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = F1Client(cache_dir=args.cache_dir, refresh=args.refresh)
    df = build_dataset(client, range(args.start, args.end + 1), cache_path=args.dataset)
    logger.info("Training final model on %d rows (%d seasons)", len(df), df["season"].nunique())
    models = train_final_model(df)
    save_checkpoint(models, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
