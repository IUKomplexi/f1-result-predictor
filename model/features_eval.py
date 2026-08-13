"""Feature ablation evaluation for the dashboard's Feature Lab tab.

Answers "what happens to walk-forward MAE/Spearman if I add or remove one
feature?" around a baseline feature set (the config selection, or a saved
checkpoint's own set). Each variant reuses the reduced-window evaluator from
``model/search.py`` so the numbers are comparable to the search page; the job
is compute-heavy (one walk-forward per candidate) and streams progress lines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from f1core.config import load_config
from f1data import F1Client
from features.build import build_dataset
from features.registry import all_feature_ids, enabled_features, feature_fingerprint
from model.search import evaluate_config
from model.train import model_params


def evaluate_feature_deltas(
    df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    max_test_season: int | None = None,
    log=None,
) -> dict:
    """Baseline walk-forward metrics + per-feature deltas for add/remove.

    ``features`` is the baseline set; every feature in the registry not in it
    and present in the dataset is tested as an addition, and every baseline
    feature as a removal. Deltas are variant-minus-baseline, so a negative
    ``delta_mae`` means the variant scored better (lower MAE). Results are
    sorted best-first (most helpful addition / most beneficial removal).
    """
    log = log or (lambda msg: print(msg, flush=True))
    baseline_mae, baseline_spearman = evaluate_config(
        df, params, max_test_season, features
    )
    log(
        f"Baseline ({len(features)} features): mae={baseline_mae:.4f} "
        f"spearman={baseline_spearman:.4f}"
    )

    def deltas(variant: list[str]) -> tuple[float, float]:
        return evaluate_config(df, params, max_test_season, variant)

    additions: list[dict[str, float | str]] = []
    for feat in all_feature_ids():
        if feat in features or feat not in df.columns:
            continue
        mae, spearman = deltas([*features, feat])
        additions.append(
            {
                "feature": feat,
                "mae": mae,
                "spearman": spearman,
                "delta_mae": mae - baseline_mae,
                "delta_spearman": spearman - baseline_spearman,
            }
        )
        log(
            f"add {feat}: mae {mae:.4f} "
            f"(delta {mae - baseline_mae:+.4f}), "
            f"spearman {spearman:.4f} "
            f"(delta {spearman - baseline_spearman:+.4f})"
        )

    removals: list[dict[str, float | str]] = []
    for feat in features:
        variant = [f for f in features if f != feat]
        if not variant or feat not in df.columns:
            continue
        mae, spearman = deltas(variant)
        removals.append(
            {
                "feature": feat,
                "mae": mae,
                "spearman": spearman,
                "delta_mae": mae - baseline_mae,
                "delta_spearman": spearman - baseline_spearman,
            }
        )
        log(
            f"remove {feat}: mae {mae:.4f} "
            f"(delta {mae - baseline_mae:+.4f}), "
            f"spearman {spearman:.4f} "
            f"(delta {spearman - baseline_spearman:+.4f})"
        )

    additions.sort(key=lambda row: float(row["delta_mae"]))  # type: ignore[arg-type]
    removals.sort(key=lambda row: float(row["delta_mae"]))  # type: ignore[arg-type]
    return {
        "baseline": {
            "mae": baseline_mae,
            "spearman": baseline_spearman,
            "n_features": len(features),
        },
        "additions": additions,
        "removals": removals,
    }


def run(
    *,
    start: int | None = None,
    end: int | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
    dataset: str | None = None,
    max_test_season: int | None = None,
    model_path: str | None = None,
    enable_features: Sequence[str] = (),
    disable_features: Sequence[str] = (),
    cfg: dict | None = None,
    log=None,
) -> dict:
    """Run the feature evaluation end-to-end and return JSON-safe deltas.

    ``log`` is an optional progress callback (web job runner). ``model_path``
    uses a saved checkpoint's feature set as the baseline (the checkpoint's
    stored features); otherwise the config feature selection (plus the
    ``enable_features``/``disable_features`` toggles) is the baseline. The
    hyperparameters come from ``[model.params]`` either way.

    Every path/season argument defaults to ``None`` and resolves from the
    config (``[data]`` seasons/cache/dataset).
    """
    log = log or (lambda msg: print(msg, flush=True))
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
    if end < start:
        raise ValueError(
            f"feature evaluation: end season {end} is before start season {start}"
        )
    if end - start + 1 < 4:
        raise ValueError(
            f"feature evaluation needs at least 4 seasons (3 to train on + 1 "
            f"to test); {start}-{end} is only {end - start + 1} seasons"
        )
    client = F1Client(cache_dir=cache_dir, refresh=refresh)
    log(f"Building dataset {start}-{end} ...")
    df = build_dataset(client, range(start, end + 1), cache_path=dataset)
    params = model_params(cfg)

    if model_path:
        from model.train import checkpoint_meta

        meta = checkpoint_meta(model_path)
        if not meta or "features" not in meta:
            raise ValueError(
                f"checkpoint {model_path} carries no feature list; retrain it "
                "with the current model/train.py before evaluating"
            )
        feats = list(meta["features"])
        log(
            f"Baseline feature set from model {model_path} "
            f"({len(feats)} features)"
        )
    else:
        feats = enabled_features(
            cfg, enable=list(enable_features), disable=list(disable_features)
        )
        log(f"Baseline feature set from config ({len(feats)} features)")

    result = evaluate_feature_deltas(df, feats, params, max_test_season, log)
    result.update(
        {
            "n_features": len(feats),
            "fingerprint": feature_fingerprint(feats),
            "params": params,
            "model_path": model_path,
            "max_test_season": max_test_season,
        }
    )
    log("Done.")
    return result
