"""Hurdle models for predicting points scored per race, with walk-forward training."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from f1core.config import load_config
from f1data import F1Client
from features.build import CATEGORICAL_FEATURES, build_dataset
from features.registry import all_feature_ids, enabled_features, feature_fingerprint

logger = logging.getLogger(__name__)

# Canonical full feature set = the registry (27 numeric + 4 categorical), in
# registry order. Feature selection (config + CLI) narrows this at prepare time.
FEATURES = all_feature_ids()

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


# Default gradient-boosting hyperparameters live in ``[model.params]``
# (f1core/config.py DEFAULTS) — the single source of truth. They were chosen
# by walk-forward-validated search (model/search.py, test seasons <= 2019)
# with the full feature set incl. teammate-relative features. The dashboard
# tunes them by writing ``[model.params]``; see :func:`model_params`.


def model_params(cfg: dict | None = None) -> dict[str, Any]:
    """Effective HGB hyperparameters: ``[model.params]`` from config."""
    cfg = cfg or load_config()
    return dict((cfg.get("model") or {}).get("params") or {})


def _clf(seed: int, params: dict[str, Any] | None = None) -> HistGradientBoostingClassifier:
    p = dict(params or {})
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=seed,
        **p,
    )


def _reg(seed: int, params: dict[str, Any] | None = None) -> HistGradientBoostingRegressor:
    p = dict(params or {})
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        random_state=seed,
        **p,
    )


def prepare(
    df: pd.DataFrame, features: Sequence[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a featured dataset into (X, y).

    ``features`` selects the model columns (default: the full
    ``FEATURES`` set); categorical features among them are converted to
    pandas ``category`` dtype so the gradient-boosted models treat them as
    categorical (unknown categories at prediction time become missing
    values, which HGB handles natively).
    """
    feats = list(features) if features is not None else FEATURES
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(
            f"features not present in the dataset: {missing}; "
            "the dataset may be stale (rebuild with features/build.py)"
        )
    X = df[feats].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
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
    return X, y  # type: ignore[reportReturnType]  # df[FEATURES] is Unknown; runtime is (DataFrame, DataFrame)


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

    def __init__(self, seed: int = 42, params: dict[str, Any] | None = None) -> None:
        self.seed = seed
        self.params = dict(params or {})
        self.scored: Any = _clf(seed, self.params)
        self.top3: Any = _clf(seed + 1, self.params)
        self.win: Any = _clf(seed + 2, self.params)
        self.points_if_scored: Any = _reg(seed + 3, self.params)
        # Numeric columns that were constant in the training set are dropped
        # at fit time and remembered for prediction, so train and predict see
        # the same columns (sklearn validates feature names against the fit).
        self.column_drop_: list[str] = []

    def _drop_constant_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Numeric columns with at most one distinct value are dropped.

        They carry no signal, and ``HistGradientBoosting``'s binning crashes
        on them (it needs at least two distinct values) — which a pre-sprint
        season range (constant ``is_sprint_round``) would otherwise trigger.
        """
        n_distinct = X.select_dtypes(include="number").apply(
            lambda s: s.dropna().nunique()
        )
        self.column_drop_ = list(
            n_distinct[n_distinct <= 1].index  # type: ignore[reportAttributeAccessIssue]  # apply() is Unknown
        )
        return X.drop(columns=self.column_drop_)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> HurdleModels:
        X = self._drop_constant_columns(X)
        self.scored = _fit_binary(self.scored, X, y["scored"])
        self.top3 = _fit_binary(self.top3, X, y["top3"])
        self.win = _fit_binary(self.win, X, y["win"])
        mask = y["scored"].to_numpy() == 1
        if mask.sum() == 0:
            self.points_if_scored = _ConstantPoints()
        else:
            self.points_if_scored.fit(X[mask], np.log1p(y["points"].to_numpy()[mask]))
        return self

    def _apply_drop(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fit-time column drop to a prediction-time frame."""
        # getattr: checkpoints trained before the drop feature have no
        # column_drop_ attribute; treat them as dropping nothing.
        drop = [c for c in getattr(self, "column_drop_", []) if c in X.columns]
        return X.drop(columns=drop) if drop else X

    def predict_expected_points(self, X: pd.DataFrame) -> np.ndarray:
        """E[points] = P(scored) * E[points | scored]."""
        X = self._apply_drop(X)
        p_scored = self.scored.predict_proba(X)[:, 1]
        exp_points = np.expm1(self.points_if_scored.predict(X))
        return p_scored * np.clip(exp_points, 0.0, None)

    def predict_probs(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self._apply_drop(X)
        return {
            "p_scored": self.scored.predict_proba(X)[:, 1],
            "p_top3": self.top3.predict_proba(X)[:, 1],
            "p_win": self.win.predict_proba(X)[:, 1],
        }


def walk_forward_seasons(
    df: pd.DataFrame, min_train_seasons: int = 3
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Yield (train_df, test_df, test_season) in chronological order.

    Train = every season strictly before the test season, so no test-season
    information (including circuit history) is available during training.
    """
    seasons = sorted(df["season"].unique())
    for test_season in seasons:
        train = df[df["season"] < test_season]
        if train["season"].nunique() < min_train_seasons:  # type: ignore[reportAttributeAccessIssue]  # boolean-mask slice is Unknown
            continue
        yield train, df[df["season"] == test_season], test_season  # type: ignore[reportReturnType]  # df[...] is Unknown; declared type is authoritative for callers


def train_final_model(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    params: dict[str, Any] | None = None,
) -> HurdleModels:
    """Train on every season (used for live prediction).

    ``features`` selects the model columns (default: the full set). ``params``
    are the HGB hyperparameters (default: ``model_params()`` from config).
    """
    X, y = prepare(df, features)
    return HurdleModels(params=params or model_params()).fit(X, y)


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


def save_checkpoint(
    models: HurdleModels,
    path: str | Path,
    features: Sequence[str] | None = None,
    season_range: tuple[int, int] | None = None,
) -> None:
    """Persist the model plus the feature set (and its fingerprint) it was trained on.

    ``features`` defaults to the full ``FEATURES`` set. The stored fingerprint
    lets :func:`load_checkpoint` reject a checkpoint trained on a different
    feature selection instead of silently reusing it. ``season_range`` records
    the training window (used by calibration to know which seasons are truly
    out-of-sample for this model).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = list(features) if features is not None else FEATURES
    # When train.py is executed as a script, classes are defined in
    # ``__main__``; a pickled reference to ``__main__.HurdleModels`` cannot be
    # resolved by other processes (e.g. predict.py). Alias the running module
    # under the canonical name and repoint the classes so the checkpoint
    # loads anywhere.
    if __name__ == "__main__":
        sys.modules["model.train"] = sys.modules["__main__"]
    for cls in (HurdleModels, _ConstantProb, _ConstantPoints):
        cls.__module__ = "model.train"
    payload: dict[str, Any] = {
        "models": models,
        "features": feats,
        "fingerprint": feature_fingerprint(feats),
    }
    if season_range is not None:
        payload["season_range"] = [int(season_range[0]), int(season_range[1])]
    _joblib_dump(payload, path)
    logger.info("Saved checkpoint (%d features, fp %s) to %s",
                len(feats), feature_fingerprint(feats), path)


def checkpoint_meta(path: str | Path) -> dict[str, Any] | None:
    """Best-effort metadata from a checkpoint file (features/fingerprint/season_range).

    Returns None for a missing file; missing keys are simply absent from the
    dict so legacy checkpoints (no ``season_range``) degrade gracefully.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        payload = _joblib_load(p)
    except (OSError, ValueError, KeyError):
        return None
    if not isinstance(payload, dict):
        return None
    meta: dict[str, Any] = {}
    for key in ("features", "fingerprint", "season_range"):
        if key in payload:
            meta[key] = payload[key]
    return meta or None


def update_model_index(path: str | Path, meta: dict) -> Path:
    """Record a trained checkpoint in ``index.json`` next to it (name -> metadata).

    The index is the source of truth for the dashboard's model selector
    (``GET /api/models``): each entry maps the checkpoint's stem (e.g.
    ``hurdle-2022-2026-1a2b3c4d``) to its metadata. Re-training a name
    overwrites its entry.
    """
    path = Path(path)
    index_path = path.parent / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index: dict = {}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {}
    index[path.stem] = meta
    tmp = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
    os.replace(tmp, index_path)
    return index_path


def load_checkpoint(
    path: str | Path, expected: Sequence[str] | None = None
) -> HurdleModels:
    """Load a checkpoint, rejecting one trained on a different feature set.

    ``expected`` is the feature selection the caller will predict with
    (default: the full ``FEATURES`` set). Checkpoints store both the feature
    list and its fingerprint; legacy checkpoints without a fingerprint are
    validated against their stored list.
    """
    payload = _joblib_load(Path(path))
    stored = list(payload.get("features", []))
    if "fingerprint" in payload:
        stored_fp = payload["fingerprint"]
    else:
        stored_fp = feature_fingerprint(stored)
    expect = list(expected) if expected is not None else FEATURES
    if stored_fp != feature_fingerprint(expect):
        raise ValueError(
            "checkpoint feature set does not match the requested feature set "
            f"(stored fp {stored_fp} vs {feature_fingerprint(expect)}); "
            "retrain with model/train.py"
        )
    return payload["models"]


def run(
    *,
    start: int | None = None,
    end: int | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
    dataset: str | None = None,
    out: str | None = None,
    enable_features: Sequence[str] = (),
    disable_features: Sequence[str] = (),
    cfg: dict | None = None,
    log=None,
) -> dict:
    """Run the training step end-to-end and return a JSON-safe summary.

    ``log`` is an optional callable receiving progress lines (used by the web
    job runner to stream output); it defaults to module logging. All arguments
    are keyword-only so the web job runner and the CLI share one code path.

    Every path/season argument defaults to ``None`` and resolves from the
    config (``[data] start_season/end_season/cache_dir/dataset``,
    ``[model] checkpoint``) so ``config.toml`` is the single source of truth.
    """
    log = log or (lambda msg: logger.info(msg))
    cfg = cfg or load_config()
    # Config values are validated as the cast types (see validate_config); the
    # casts keep the declared types after the None-guards so downstream calls
    # don't carry Optional[None].
    if start is None:
        start = cast(int, cfg["data"]["start_season"])
    if end is None:
        end = cast(int, cfg["data"]["end_season"])
    if cache_dir is None:
        cache_dir = cast(str, cfg["data"]["cache_dir"])
    if dataset is None:
        dataset = cast(str, cfg["data"]["dataset"])
    if out is None:
        out = cast(str, cfg["model"]["checkpoint"])
    client = F1Client(cache_dir=cache_dir, refresh=refresh)
    log(f"Building dataset {start}-{end} ...")
    df = build_dataset(client, range(start, end + 1), cache_path=dataset)
    feats = enabled_features(
        cfg, enable=list(enable_features), disable=list(disable_features)
    )
    log(
        f"Training final model on {len(df)} rows ({df['season'].nunique()} "
        f"seasons), {len(feats)} features (fp {feature_fingerprint(feats)})"
    )
    models = train_final_model(df, feats, params=model_params(cfg))
    # Record the ACTUAL data window the model saw (the requested start..end may
    # include seasons with no fetched data).
    actual_range = (int(df["season"].min()), int(df["season"].max()))  # type: ignore[reportAttributeAccessIssue]
    save_checkpoint(models, out, features=feats, season_range=actual_range)
    index_path = update_model_index(out, {
        "checkpoint": out,
        "params": model_params(cfg),
        "features": feats,
        "fingerprint": feature_fingerprint(feats),
        "season_range": list(actual_range),
        "rows": len(df),
        "seasons": int(df["season"].nunique()),  # type: ignore[reportArgumentType]  # nunique on an untyped Series is Unknown
        "trained_at": time.time(),
    })
    log(f"Saved checkpoint to {out}")
    log(f"Recorded model index {index_path}")
    return {
        "rows": len(df),
        "seasons": int(df["season"].nunique()),  # type: ignore[reportArgumentType]  # nunique on an untyped Series is Unknown
        "n_features": len(feats),
        "features": feats,
        "fingerprint": feature_fingerprint(feats),
        "checkpoint": out,
        "params": model_params(cfg),
    }


