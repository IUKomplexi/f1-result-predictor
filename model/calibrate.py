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

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from f1core.config import load_config
from f1core.reporting import to_md
from f1data import F1Client
from features.build import build_dataset
from features.registry import enabled_features, feature_fingerprint
from model.train import (
    HurdleModels,
    _joblib_dump,
    _joblib_load,
    model_params,
    prepare,
    walk_forward_seasons,
)

TARGETS = {"scored": "p_scored", "top3": "p_top3", "win": "p_win"}


# --------------------------------------------------------------------------
# Out-of-sample score collection
# --------------------------------------------------------------------------

def collect_oos_scores(
    df: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Raw model probabilities on held-out races via walk-forward.

    Each test-season row carries the raw scores of a model trained only on
    strictly earlier seasons, plus the actual outcome labels. ``features``
    selects the model columns (default: the full set).
    """
    chunks = []
    params = model_params()  # read [model.params] once, outside the loop
    for train, test, _ in walk_forward_seasons(df):
        X_train, y_train = prepare(train, features)
        model = HurdleModels(params=params).fit(X_train, y_train)
        X_test, _ = prepare(test, features)
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


def collect_oos_scores_fixed(
    df: pd.DataFrame, model, features: list[str] | None = None
) -> pd.DataFrame:
    """Raw probabilities of one *fixed* model on every row of ``df``.

    Used to calibrate a saved checkpoint: the caller decides which seasons are
    genuinely out-of-sample (after the model's training window) and filters the
    result. Same output columns as :func:`collect_oos_scores`.
    """
    X, _ = prepare(df, features)
    probs = model.predict_probs(X)
    return pd.DataFrame(
        {
            "p_scored": probs["p_scored"],
            "p_top3": probs["p_top3"],
            "p_win": probs["p_win"],
            "scored": df["scored"].to_numpy(dtype=int),  # type: ignore[reportAttributeAccessIssue]
            "top3": df["top3"].to_numpy(dtype=int),  # type: ignore[reportAttributeAccessIssue]
            "win": df["win"].to_numpy(dtype=int),  # type: ignore[reportAttributeAccessIssue]
            "season": df["season"].to_numpy(),  # type: ignore[reportAttributeAccessIssue]
        },
        index=df.index,
    )


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------

def fit_calibrators(oos: pd.DataFrame) -> dict[str, IsotonicRegression]:
    """Fit an isotonic regressor per target on out-of-sample scores."""
    calibrators: dict[str, IsotonicRegression] = {}
    for target, score in TARGETS.items():
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oos[score].to_numpy(dtype=float), oos[target].to_numpy(dtype=float))
        calibrators[target] = iso
    return calibrators


def apply_calibration(probs: dict[str, np.ndarray],
                      calibrators: dict[str, IsotonicRegression]) -> dict[str, np.ndarray]:
    """Map raw score arrays to calibrated probabilities (keys: p_scored/...)."""
    out = dict(probs)
    for target, score in TARGETS.items():
        if target in calibrators:
            out[score] = calibrators[target].predict(np.asarray(probs[score], dtype=float))
    return out


def load_calibrators(
    path: str | Path, expected: list[str] | None = None
) -> dict[str, IsotonicRegression] | None:
    """Load calibrators, or None when the file does not exist.

    ``expected`` is the feature selection the caller will predict with;
    when the file carries a feature fingerprint it must match (a calibrator
    file fit on a different enabled subset would silently corrupt the
    probabilities). Legacy files without a fingerprint load unchecked.
    """
    p = Path(path)
    if not p.exists():
        return None
    payload = _joblib_load(p)
    if not isinstance(payload, dict) or "calibrators" not in payload:
        raise ValueError(f"{p} is not a calibrator file (missing 'calibrators' key)")
    if expected is not None and payload.get("fingerprint") not in (
        None, feature_fingerprint(expected)
    ):
        raise ValueError(
            "calibrator feature set does not match the requested feature set "
            f"(stored fp {payload['fingerprint']} vs {feature_fingerprint(expected)}); "
            "re-run f1 calibrate after f1 train"
        )
    return payload["calibrators"]


def save_calibrators(
    calibrators: dict[str, IsotonicRegression],
    path: str | Path,
    features: list[str] | None = None,
) -> None:
    """Persist calibrators plus the feature-set fingerprint they were fit on.

    ``features`` defaults to None (no fingerprint written); pass the enabled
    feature set so :func:`load_calibrators` can reject a mismatch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"calibrators": calibrators}
    if features is not None:
        payload["fingerprint"] = feature_fingerprint(features)
    _joblib_dump(payload, path)


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
    return table.round(3)  # type: ignore[reportReturnType]  # groupby().agg() is Unknown without stubs


def summarize(oos: pd.DataFrame, calibrators: dict[str, IsotonicRegression],
              context: str = "", keep: set | None = None) -> str:
    """Raw-vs-calibrated Brier scores and reliability tables, as text.

    ``context`` is a one-line note describing how ``calibrators`` were fit
    relative to ``oos`` (e.g. a chronological hold-out), so the numbers are
    never mistaken for out-of-calibration-sample evidence. ``keep`` lists the
    targets whose calibrators are actually deployed.
    """
    context_label = context or "in-sample (calibrators fit and evaluated on the same rows)"
    lines = [
        "# Calibration (isotonic, fit on walk-forward out-of-sample scores)",
        "",
        f"- Evaluation context: {context_label}",
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
        lines.append(to_md(reliability_table(oos[target], cal)))
        lines.append("")
    return "\n".join(lines)


def calibration_snapshot(
    oos: pd.DataFrame,
    calibrators: dict[str, IsotonicRegression],
    keep: set,
    context: str = "",
) -> dict:
    """JSON-safe calibration evaluation for the web dashboard.

    Per target (scored/top3/win): raw and calibrated Brier, the deployment
    decision, and the binned reliability table for a reliability curve.
    """
    targets: dict[str, dict] = {}
    for target, score in TARGETS.items():
        y = oos[target].to_numpy(dtype=float)
        raw = oos[score].to_numpy(dtype=float)
        cal = calibrators[target].predict(raw)
        rel = reliability_table(oos[target], cal)
        targets[target] = {
            "brier_raw": brier(y, raw),
            "brier_calibrated": brier(y, cal),
            "delta": brier(y, cal) - brier(y, raw),
            "deployed": target in keep,
            "reliability": [
                {
                    "mean_pred": float(mp),
                    "observed": float(ob),
                    "n": int(n),
                }
                for mp, ob, n in zip(
                    rel["mean_pred"], rel["observed"], rel["n"], strict=True
                )
            ],
        }
    return {"context": context, "deployed": sorted(keep), "targets": targets}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _holdout_split(
    oos: pd.DataFrame,
    fit_through_season: int | None = None,
    eval_from_season: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fit, eval) OOS frames for the calibration evaluation split.

    With both ``fit_through_season`` and ``eval_from_season`` given, the split
    is explicit: fit on OOS seasons <= fit-through and evaluate on seasons >=
    eval-from. Otherwise a chronological two-thirds split is used.
    """
    if fit_through_season is not None and eval_from_season is not None:
        return (
            oos[oos["season"] <= fit_through_season],  # type: ignore[reportAttributeAccessIssue]  # boolean-mask slice is Unknown
            oos[oos["season"] >= eval_from_season],  # type: ignore[reportAttributeAccessIssue]
        )
    seasons = sorted(oos["season"].unique())
    split_at = seasons[len(seasons) * 2 // 3]
    return (
        oos[oos["season"] < split_at],  # type: ignore[reportAttributeAccessIssue]
        oos[oos["season"] >= split_at],  # type: ignore[reportAttributeAccessIssue]
    )


def run(
    *,
    start: int = 2010,
    end: int = 2026,
    refresh: bool = False,
    cache_dir: str = "data/raw",
    dataset: str = "data/features.parquet",
    out: str = "data/model/calibrators.joblib",
    out_json: str = "reports/calibration.json",
    enable_features: Sequence[str] = (),
    disable_features: Sequence[str] = (),
    cfg: dict | None = None,
    log=None,
    fit_through_season: int | None = None,
    eval_from_season: int | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit + deploy calibrators end-to-end and return a JSON-safe summary.

    ``log`` is an optional progress callback (web job runner). Writes the
    calibrator checkpoint and the ``reports/calibration.json`` dashboard
    snapshot; the returned dict carries the full ``calibration_snapshot`` so
    the web runner can refresh the dashboard view without re-reading the file.

    ``model_path`` calibrates a *saved checkpoint*: the checkpoint's own
    feature set is used (the feature toggles are ignored), out-of-sample
    scores come from that fixed model on seasons after its training window,
    and the deployment decision is evaluated on the newest such season only.
    Calibrators are written next to the model
    (``data/model/<stem>.calibrators.joblib``) so each model keeps its own.

    Without ``model_path`` the legacy walk-forward behaviour applies:
    ``fit_through_season`` / ``eval_from_season`` optionally override the
    hold-out split used for the *evaluation* Brier deltas (which calibrators
    are deployed): with both set, calibrators for evaluation are fit on OOS
    seasons <= ``fit_through_season`` and evaluated on seasons >=
    ``eval_from_season``. Deployment calibrators are always fit on all OOS
    scores. When omitted, the default chronological two-thirds split is used.
    This is a configuration choice, not new model behavior.
    """
    log = log or (lambda msg: print(msg, flush=True))
    cfg = cfg or load_config()
    client = F1Client(cache_dir=cache_dir, refresh=refresh)
    log(f"Building dataset {start}-{end} ...")
    df = build_dataset(client, range(start, end + 1), cache_path=dataset)
    feats = enabled_features(
        cfg, enable=list(enable_features), disable=list(disable_features)
    )

    if model_path:
        from model.train import checkpoint_meta, load_checkpoint

        meta = checkpoint_meta(model_path)
        if not meta or "features" not in meta:
            raise ValueError(
                f"checkpoint {model_path} carries no feature list; retrain it "
                "with the current model/train.py before calibrating"
            )
        feats = list(meta["features"])
        train_range = meta.get("season_range")
        if not train_range:
            raise ValueError(
                f"checkpoint {model_path} has no recorded training window; "
                "retrain it with the current model/train.py before calibrating"
            )
        model = load_checkpoint(model_path, expected=feats)
        log(
            f"Scoring with model {model_path} ({len(feats)} features, "
            f"trained {train_range[0]}-{train_range[1]})"
        )
        scores = collect_oos_scores_fixed(df, model, feats)
        train_end = int(train_range[1])
        oos = scores[scores["season"] > train_end]  # type: ignore[reportAttributeAccessIssue]
        if oos.empty:
            raise ValueError(
                f"model {model_path} was trained through {train_end} — no "
                f"out-of-sample seasons remain in {start}-{end}. Retrain the "
                "model with an earlier end season (leave the newest season "
                "out), or fetch newer data first."
            )
        oos_seasons = sorted(int(s) for s in oos["season"].unique())
        log(f"Out-of-sample seasons for this model: {oos_seasons}")
        # The model is always judged on the newest season: fit calibrators on
        # every earlier OOS season, evaluate the deployment decision on the
        # newest one only.
        fit_through_season = oos_seasons[-1] - 1
        eval_from_season = oos_seasons[-1]
        cal_out = str(Path(model_path).with_name(
            f"{Path(model_path).stem}.calibrators.joblib"
        ))
        log(f"Calibrators for this model will be written to {cal_out}")
    else:
        cal_out = out
        log(
            f"Collecting out-of-sample scores "
            f"({len(feats)} features, fp {feature_fingerprint(feats)})"
        )
        oos = collect_oos_scores(df, feats)
    # Deployment calibrators: fit on ALL out-of-sample scores (most data).
    calibrators = fit_calibrators(oos)
    save_calibrators(calibrators, cal_out, features=feats)

    # Honest evaluation: fit calibrators on the earlier OOS seasons only and
    # evaluate on the later ones, so the reported Brier deltas are not
    # in-sample for the calibration step. An explicit split (fit through /
    # evaluate from) overrides the default two-thirds chronological split.
    fit_oos, eval_oos = _holdout_split(oos, fit_through_season, eval_from_season)
    if len(eval_oos) < 200 or fit_oos.empty:
        eval_oos = oos
        context = f"in-sample (too few hold-out rows; fit+eval on the same {len(oos)} OOS rows)"
        eval_cal = calibrators
    else:
        eval_cal = fit_calibrators(fit_oos)  # type: ignore[reportArgumentType]  # boolean-mask slice is Unknown
        if model_path:
            context = (
                f"model {Path(model_path).name}: calibrators fit on OOS seasons "
                f"{min(fit_oos['season'])}-{max(fit_oos['season'])}, evaluated "
                f"on the newest season {max(eval_oos['season'])}"
            )
        else:
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
    save_calibrators(deployment, cal_out, features=feats)
    log(f"Deployed calibrators: {', '.join(sorted(keep)) if keep else 'none (raw kept)'}")
    log(f"Wrote {cal_out}")

    snapshot = calibration_snapshot(
        eval_oos, eval_cal, keep=keep, context=context  # type: ignore[reportArgumentType]  # eval_oos is a boolean-mask slice (Unknown)
    )
    json_out = Path(out_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    log(f"Wrote {json_out}")

    return {
        "deployed": sorted(keep),
        "n_features": len(feats),
        "fingerprint": feature_fingerprint(feats),
        "calibrators": cal_out,
        "checkpoint": model_path,
        "snapshot": snapshot,
        "summary": summarize(eval_oos, eval_cal, context=context, keep=keep),  # type: ignore[reportArgumentType]  # eval_oos is a boolean-mask slice (Unknown)
    }


