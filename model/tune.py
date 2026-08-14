"""Hyperparameter search for the hurdle model via walk-forward backtest.

The objective is the walk-forward model MAE (mean absolute error of the
quantized expected points — the dashboard scoreboard headline), measured the
same way as ``f1 backtest``. Candidate 0 is always the effective
``[model.params]`` (the baseline), so a run that finds nothing better still
proves the current defaults hold up against the sampled grid. The sampling is
seeded from ``[model] seed``, so a fixed candidate count is reproducible.

Every candidate runs a full walk-forward backtest (the model is re-trained per
test season), so a 24-candidate run takes roughly 24x one backtest. The
harness mirrors ``model/evaluate.py::run``'s config resolution so the CLI and
the web job runner share one code path.
"""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from f1core.config import load_config
from f1data import F1Client
from features.build import build_dataset
from features.registry import enabled_features, feature_fingerprint
from model.evaluate import run_backtest
from model.train import model_params

# HGB hyperparameter grids (the cartesian product has 4*4*4*3*3 = 576 combos).
PARAM_GRIDS: dict[str, list[Any]] = {
    "max_iter": [200, 400, 600, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 5],
    "l2_regularization": [0.1, 1.0, 10.0],
    "min_samples_leaf": [5, 20, 50],
}

# Metrics where a lower value is better; all others: higher is better.
_BETTER_IS_LOWER = {"mae"}

# The five headline metrics, in display order.
_METRIC_KEYS = ("winner_hit", "top3_overlap", "top10_overlap", "spearman", "mae")

_N_CANDIDATE_MAX = 576


def candidate_params(cfg: dict, n: int) -> list[dict]:
    """Deterministic candidate list: current params first, then a random sample.

    Candidate 0 is ``model_params(cfg)`` (the effective ``[model.params]``);
    the remaining ``n-1`` are drawn without replacement from the cartesian
    product of :data:`PARAM_GRIDS`, sampled with
    ``random.Random(int(cfg["model"]["seed"]))`` so a fixed ``n`` always
    yields the same list. ``n`` is clamped to the product size (576).
    """
    n = max(1, min(int(n), _N_CANDIDATE_MAX))
    base = model_params(cfg)
    if n == 1:
        return [base]
    combos = [
        dict(zip(PARAM_GRIDS, values, strict=True))
        for values in itertools.product(*PARAM_GRIDS.values())
    ]
    rng = random.Random(int(cfg["model"]["seed"]))
    sampled = rng.sample(combos, min(n - 1, len(combos)))
    return [base, *sampled]


def evaluate_candidate(
    df: pd.DataFrame,
    params: dict,
    features: Sequence[str],
    quantize: bool = True,
) -> dict[str, float]:
    """Walk-forward backtest for one param set; the model row as floats.

    Returns the five headline metrics (``winner_hit``, ``top3_overlap``,
    ``top10_overlap``, ``spearman``, ``mae``) for the model row. Expected
    points are quantized by default (the deployed output).
    """
    overall, _ = run_backtest(df, quantize=quantize, features=features, params=params)
    row = overall.loc["model"]
    return {metric: float(row[metric]) for metric in _METRIC_KEYS}


def _format_report(
    baseline: dict, ranked: list[dict], metric: str, quantize: bool
) -> str:
    """Markdown tuning report: baseline first, then candidates best-last."""
    lines = [
        "# Hyperparameter tuning (walk-forward)",
        "",
        (
            f"Objective: **{metric}** (quantize={'on' if quantize else 'off'}). "
            "Each candidate is a full walk-forward backtest; the model is "
            "re-trained per test season."
        ),
        "",
        "## Candidates",
        "",
        "| rank | params | winner_hit | top3_overlap | top10_overlap | spearman | mae |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.append(
        "| baseline | " + _params_cell(baseline["params"]) + " |"
        + _metrics_cell(baseline["metrics"]) + " |"
    )
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | " + _params_cell(row["params"]) + " |"
            + _metrics_cell(row["metrics"]) + " |"
        )
    return "\n".join(lines) + "\n"


def _params_cell(params: dict) -> str:
    return f"`{json.dumps(params)}`"


def _metrics_cell(metrics: dict) -> str:
    return " |".join(f" {metrics[m]:.4f}" for m in _METRIC_KEYS)


def run(
    *,
    start: int | None = None,
    end: int | None = None,
    refresh: bool = False,
    cache_dir: str | None = None,
    dataset: str | None = None,
    candidates: int = 24,
    metric: str = "mae",
    quantize: bool = True,
    out: str | None = None,
    out_json: str | None = None,
    enable_features: Sequence[str] = (),
    disable_features: Sequence[str] = (),
    cfg: dict | None = None,
    log=None,
) -> dict:
    """Evaluate ``candidates`` param sets and write a tuning report + JSON.

    Mirrors ``model/evaluate.py::run``'s config resolution (``[data]``
    seasons/cache/dataset, feature toggles on top of the config). The dataset
    is built once and reused for every candidate. Candidates are ranked by
    ``metric`` (ascending for ``mae``, descending otherwise — best first);
    the report lists the baseline candidate first and the best candidate
    last. Returns a JSON-safe dict with the best and baseline params/metrics.
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
    if out is None:
        out = "reports/tuning.md"
    if out_json is None:
        out_json = str(Path(out).with_suffix(".json"))
    client = F1Client(cache_dir=cache_dir, refresh=refresh)
    log(f"Building dataset {start}-{end} ...")
    df = build_dataset(client, range(start, end + 1), cache_path=dataset)
    feats = enabled_features(
        cfg, enable=list(enable_features), disable=list(disable_features)
    )
    log(
        f"Evaluating {candidates} candidates on {len(feats)} features "
        f"(fp {feature_fingerprint(feats)}), metric={metric}, "
        f"quantize={quantize}"
    )

    all_params = candidate_params(cfg, candidates)
    results: list[dict] = []
    for i, params in enumerate(all_params, start=1):
        log(f"Candidate {i}/{len(all_params)}: {json.dumps(params)}")
        metrics = evaluate_candidate(df, params, feats, quantize=quantize)
        results.append({"params": params, "metrics": metrics})
        log(f"  {metrics}")

    ranked = sorted(
        results,
        key=lambda row: row["metrics"][metric],
        reverse=metric not in _BETTER_IS_LOWER,
    )
    baseline_row = next(
        row for row in results if row["params"] == all_params[0]
    )
    report = _format_report(baseline_row, ranked, metric, quantize)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(report, encoding="utf-8")
    log(f"Wrote {out_p}")

    json_out = Path(out_json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "metric": metric,
                "quantize": quantize,
                "n_candidates": len(ranked),
                "baseline": baseline_row,
                "best": ranked[0],
                "ranked": ranked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Wrote {json_out}")
    return {
        "metric": metric,
        "quantize": quantize,
        "n_candidates": len(ranked),
        "baseline": baseline_row,
        "best": ranked[0],
        "top": ranked[:5],
        "features": feats,
        "n_features": len(feats),
        "fingerprint": feature_fingerprint(feats),
        "report": out,
    }
