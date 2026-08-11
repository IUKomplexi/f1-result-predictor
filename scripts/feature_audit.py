"""Walk-forward permutation-importance audit of the model's features.

For every walk-forward window (train on strictly earlier seasons, validate on
one test season) this script computes *seeded permutation importance* for the
two hurdle components separately:

* **classifier** — P(top-10): ROC-AUC of the ``scored`` classifier
* **regressor** — E(points | top-10): negative MAE (points) of the
  ``points_if_scored`` regressor, evaluated on scored rows only

Per-feature importance is averaged across windows and reported as mean ± SE
(fold-to-fold), z = mean / SE, the one-sided p-value, a Benjamini-Hochberg
FDR q-value (q = 0.05, applied per component across features), and fold
sign-stability (features whose importance flips sign across windows are
flagged unreliable).

Correlation clusters (|r| >= 0.8, on the pooled numeric features) get
*grouped* permutation importance — all features of a cluster are shuffled
together — because per-feature permutation understates the members of a
collinear group. The same clusters are ablation candidates.

Usage::

    python scripts/feature_audit.py                      # audit -> JSON
    python scripts/feature_audit.py --ablate grid,round  # + drop-column ablation gate
    python scripts/feature_audit.py --ablate-noise       # ablate noise-flagged + clusters

The JSON (default ``reports/feature_audit.json``) is the machine-readable
input for the classification step recorded in ``reports/features.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

from f1core.config import load_config
from f1data import F1Client
from features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_dataset
from model.evaluate import race_metrics
from model.train import (
    HurdleModels,
    prepare,
    quantize_points,
    walk_forward_seasons,
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Primary headline metrics (the cut gate runs on these); MAE is secondary.
PRIMARY_METRICS = ["winner_hit", "top3_overlap", "spearman"]
HEADLINE_METRICS = [*PRIMARY_METRICS, "mae"]
CORR_THRESHOLD = 0.8
FDR_Q = 0.05
CLF_SCORER = get_scorer("roc_auc")


# --------------------------------------------------------------------------
# Scorers (higher is better, as sklearn's permutation_importance expects)
# --------------------------------------------------------------------------

def reg_scorer(estimator, X, y) -> float:
    """Negative MAE of the regressor on the *points* scale.

    The regressor predicts log1p(points); expm1 maps back so the importance
    reflects the headline metric rather than the log scale.
    """
    pred = np.expm1(np.asarray(estimator.predict(X), dtype=float))
    y = np.asarray(y, dtype=float)
    return -float(np.mean(np.abs(pred - y)))


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------

def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted q-values via the BH step-up procedure (NaN -> 1.0)."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    m = len(ranked)
    adjusted = np.full(m, np.nan)
    running = 1.0
    for k in range(m - 1, -1, -1):
        running = min(1.0, ranked[k] * m / (k + 1), running)
        adjusted[k] = running
    back = np.empty(m, dtype=int)
    back[order] = np.arange(m)
    q[valid] = adjusted[back]
    return q


def summarize_folds(fold_means: dict[int, float]) -> dict[str, Any]:
    """mean / SE / z / one-sided p / sign-stability across walk-forward folds."""
    values = np.array([v for _, v in sorted(fold_means.items())], dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n > 1:
        se = float(values.std(ddof=1) / np.sqrt(n))
    else:
        se = 0.0
    if se > 0:
        z = mean / se
        p = float(1.0 - norm.cdf(z))
    else:
        z = float("inf") if mean > 0 else 0.0
        p = 0.0 if mean > 0 else 1.0
    frac_pos = float((values > 0).mean())
    return {
        "mean": round(mean, 6),
        "se": round(se, 6),
        "z": round(z, 3) if np.isfinite(z) else None,
        "p": round(min(p, 1.0), 6),
        "frac_pos": round(frac_pos, 3),
        "unreliable": 0.0 < frac_pos < 1.0,
        "per_fold": {str(k): round(float(v), 6) for k, v in sorted(fold_means.items())},
    }


# --------------------------------------------------------------------------
# Correlation clusters + grouped permutation
# --------------------------------------------------------------------------

def correlation_clusters(
    df: pd.DataFrame, features: list[str], threshold: float = CORR_THRESHOLD
) -> list[dict[str, Any]]:
    """Union-find clusters of numeric features with pairwise |r| >= threshold.

    The correlation is computed on the pooled dataset (collinearity is a
    property of the feature definitions, not of one training window).
    """
    num = [f for f in features if f in NUMERIC_FEATURES]
    corr = pd.DataFrame(df[num]).corr().abs()
    parent = list(range(len(num)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            if corr.iloc[i, j] >= threshold:
                union(i, j)

    members: dict[int, list[str]] = {}
    for i, f in enumerate(num):
        members.setdefault(find(i), []).append(f)
    clusters = []
    for group in members.values():
        if len(group) <= 1:
            continue
        sub = corr.loc[group, group]
        max_r = float(sub.to_numpy()[np.triu_indices(len(group), k=1)].max())
        clusters.append({"features": sorted(group), "max_abs_r": round(max_r, 3)})
    return clusters


def _permute_column(X: pd.DataFrame, col: str, rng) -> None:
    """Shuffle one column in place (dtype-preserving, incl. categorical)."""
    X[col] = rng.permutation(X[col].to_numpy())


def group_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y,
    group: list[str],
    n_repeats: int,
    seed: int,
    scorer,
) -> np.ndarray:
    """Permutation importance of a feature *group* shuffled together.

    Mirrors ``sklearn.inspection.permutation_importance`` for a set of
    columns: each repeat shuffles every column of the group and records
    ``baseline_score - permuted_score`` (higher = the group matters).
    """
    rng = np.random.default_rng(seed)
    baseline = float(scorer(estimator, X, y))
    scores = np.empty(n_repeats)
    for i in range(n_repeats):
        X_perm = X.copy()
        for col in group:
            _permute_column(X_perm, col, rng)
        scores[i] = baseline - float(scorer(estimator, X_perm, y))
    return scores


# --------------------------------------------------------------------------
# Walk-forward audit
# --------------------------------------------------------------------------

def _component_importances(
    model: HurdleModels,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    clusters: list[dict[str, Any]],
    n_repeats: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Per-window mean importances for both components (and their groups).

    Returns (clf per-feature, reg per-feature, clf per-group, reg per-group)
    mean importances as {feature/group-key: mean importance}.
    """
    X = model._apply_drop(X_test)
    y_c = y_test["scored"].to_numpy()
    if len(np.unique(y_c)) < 2:
        clf_scorer: Any = get_scorer("accuracy")  # degenerate window: fall back
    else:
        clf_scorer = CLF_SCORER
    cols = list(X.columns)
    clf_per, reg_per, clf_grp, reg_grp = {}, {}, {}, {}
    imp = permutation_importance(
        model.scored, X, y_c, scoring=clf_scorer,
        n_repeats=n_repeats, random_state=seed,
    )
    # Constant columns are dropped at fit; map back by name (dropped features
    # get importance 0.0 — a constant column has no permutation effect).
    clf_per = {f: 0.0 for f in FEATURES}
    for f, m in zip(cols, imp.importances_mean, strict=True):  # type: ignore[reportAttributeAccessIssue]  # Bunch attribute
        clf_per[f] = float(m)

    mask = y_test["scored"].to_numpy() == 1
    if mask.sum() > 1:
        X_reg = pd.DataFrame(X[mask])
        y_reg = y_test["points"].to_numpy(dtype=float)[mask]
        imp = permutation_importance(
            model.points_if_scored, X_reg, y_reg, scoring=reg_scorer,
            n_repeats=n_repeats, random_state=seed,
        )
        reg_per = {f: 0.0 for f in FEATURES}
        for f, m in zip(cols, imp.importances_mean, strict=True):  # type: ignore[reportAttributeAccessIssue]  # Bunch attribute
            reg_per[f] = float(m)
    else:
        reg_per = {f: 0.0 for f in FEATURES}

    for idx, cluster in enumerate(clusters):
        key = f"group{idx}:{','.join(cluster['features'])}"
        group = cluster["features"]
        clf_grp[key] = float(
            group_permutation_importance(model.scored, X, y_c, group,
                                         n_repeats, seed, clf_scorer).mean()
        )
        if mask.sum() > 1:
            reg_grp[key] = float(
                group_permutation_importance(
                    model.points_if_scored, X_reg, y_reg, group,
                    n_repeats, seed, reg_scorer
                ).mean()
            )
        else:
            reg_grp[key] = 0.0
    return clf_per, reg_per, clf_grp, reg_grp


def run_audit(
    df: pd.DataFrame,
    features: list[str] | None = None,
    n_repeats: int = 25,
    seed: int = 0,
    max_test_season: int | None = None,
) -> dict[str, Any]:
    """Permutation-importance audit across walk-forward windows."""
    features = features or FEATURES
    clusters = correlation_clusters(df, features)
    clf_folds: dict[str, dict[int, float]] = {f: {} for f in features}
    reg_folds: dict[str, dict[int, float]] = {f: {} for f in features}
    clf_group_folds: dict[str, dict[int, float]] = {}
    reg_group_folds: dict[str, dict[int, float]] = {}
    windows: list[int] = []
    for train, test, season in walk_forward_seasons(df):
        if max_test_season is not None and season > max_test_season:
            break
        windows.append(int(season))
        X_train, y_train = prepare(train, features)
        model = HurdleModels(seed=seed).fit(X_train, y_train)
        X_test, y_test = prepare(test, features)
        clf_per, reg_per, clf_grp, reg_grp = _component_importances(
            model, X_test, y_test, clusters, n_repeats, seed + season
        )
        for f in features:
            clf_folds[f][season] = clf_per[f]
            reg_folds[f][season] = reg_per[f]
        for key in clf_grp:
            clf_group_folds.setdefault(key, {})[season] = clf_grp[key]
            reg_group_folds.setdefault(key, {})[season] = reg_grp[key]

    def _per_component(folds: dict[str, dict[int, float]]) -> dict[str, Any]:
        table = {f: summarize_folds(fold) for f, fold in folds.items()}
        pvals = np.array([table[f]["p"] for f in folds], dtype=float)
        qvals = benjamini_hochberg(pvals)
        for f, q in zip(folds, qvals, strict=True):
            table[f]["q"] = round(float(q), 6)
            table[f]["significant"] = bool(table[f]["p"] < 0.05 and q < FDR_Q)
            table[f]["noise"] = bool(table[f]["p"] >= 0.05)
        return table

    return {
        "method": {
            "n_repeats": n_repeats,
            "seed": seed,
            "corr_threshold": CORR_THRESHOLD,
            "fdr_q": FDR_Q,
            "windows": windows,
            "features": features,
        },
        "clusters": clusters,
        "classifier": _per_component(clf_folds),
        "regressor": _per_component(reg_folds),
        "classifier_groups": _per_component(clf_group_folds),
        "regressor_groups": _per_component(reg_group_folds),
    }


# --------------------------------------------------------------------------
# Drop-column ablation (the cut gate)
# --------------------------------------------------------------------------

def per_fold_metrics(
    df: pd.DataFrame,
    features: list[str],
    max_test_season: int | None = None,
) -> pd.DataFrame:
    """Model headline metrics per test season for a feature subset.

    Mirrors the walk-forward backtest (quantized expected points, per-race
    metrics averaged per season) but returns only the model column — the
    input the ±1 SE ablation gate compares fold to fold.
    """
    rows = []
    for train, test, season in walk_forward_seasons(df):
        if max_test_season is not None and season > max_test_season:
            break
        X_train, y_train = prepare(train, features)
        model = HurdleModels().fit(X_train, y_train)
        X_test, _ = prepare(test, features)
        test = test.copy()
        test["pred_points"] = quantize_points(model.predict_expected_points(X_test))
        for _, race in test.groupby(["season", "round"]):
            m = race_metrics(race)
            m["season"] = season
            rows.append(m)
    fold = (
        pd.DataFrame(rows)
        .groupby("season")[HEADLINE_METRICS]
        .mean()
        .reindex(sorted({r["season"] for r in rows}))  # type: ignore[reportAttributeAccessIssue]
    )
    return fold  # type: ignore[reportReturnType]  # groupby/reindex chain is Unknown without pandas stubs


def ablation_gate(
    df: pd.DataFrame,
    drop: list[str],
    features: list[str],
    max_test_season: int | None = None,
) -> dict[str, Any]:
    """Gate: does removing ``drop`` improve any headline metric by >= 1 SE?

    For each metric the per-fold delta (removal minus baseline) is compared
    to its fold-to-fold SE (std / sqrt(n_folds)). ``gate_pass`` means removal
    improves that metric by >= 1 SE; ``gate_fail`` means removal regresses it
    by >= 1 SE.
    """
    base = per_fold_metrics(df, features, max_test_season)
    alt = per_fold_metrics(df, [f for f in features if f not in drop], max_test_season)
    metrics: dict[str, Any] = {}
    for metric in HEADLINE_METRICS:
        d = alt[metric].to_numpy(dtype=float) - base[metric].to_numpy(dtype=float)
        n = len(d)
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        metrics[metric] = {
            "delta_mean": round(float(d.mean()), 5),
            "se": round(se, 5),
            "n_folds": int(n),
            "gate_pass": bool(d.mean() >= se),
            "gate_fail": bool(d.mean() <= -se),
        }
    primary_pass = [m for m in PRIMARY_METRICS if metrics[m]["gate_pass"]]
    primary_fail = [m for m in PRIMARY_METRICS if metrics[m]["gate_fail"]]
    return {
        "drop": drop,
        "metrics": metrics,
        "cut": bool(primary_pass and not primary_fail),
        "primary_improves": primary_pass,
        "primary_regresses": primary_fail,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-test-season", type=int, default=None,
                        help="latest walk-forward test season (speed)")
    parser.add_argument("--n-repeats", type=int, default=25,
                        help="permutation repeats per feature per window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablate", default="",
                        help="comma-separated features to run the ablation gate on")
    parser.add_argument("--ablate-noise", action="store_true",
                        help="run the ablation gate on noise-flagged features + clusters")
    parser.add_argument("--skip-audit", action="store_true",
                        help="load the existing --out-json instead of re-running "
                             "the permutation audit")
    parser.add_argument("--out-json", default="reports/feature_audit.json")
    args = parser.parse_args()

    cfg = load_config()
    start = args.start if args.start is not None else cfg["data"]["start_season"]
    end = args.end if args.end is not None else cfg["data"]["end_season"]
    client = F1Client(
        cache_dir=args.cache_dir or cfg["data"]["cache_dir"],
        refresh=args.refresh,
        base_url=cfg["api"]["base_url"],
        user_agent=cfg["api"]["user_agent"],
    )
    df = build_dataset(
        client, range(start, end + 1),
        cache_path=args.dataset or cfg["data"]["dataset"],
    )
    if args.skip_audit:
        out = Path(args.out_json)
        if not out.exists():
            raise SystemExit(f"--skip-audit but {out} does not exist")
        audit = json.loads(out.read_text(encoding="utf-8"))
        print(f"Loaded existing audit from {out}")
    else:
        print(f"Auditing {len(df)} rows, seasons {start}-{end} "
              f"(test windows {sorted(df['season'].unique())[3:]})...")
        audit = run_audit(
            df, n_repeats=args.n_repeats, seed=args.seed,
            max_test_season=args.max_test_season,
        )

    ablations = []
    if args.ablate:
        for feat in args.ablate.split(","):
            feat = feat.strip()
            if feat:
                ablations.append(
                    ablation_gate(df, [feat], FEATURES, args.max_test_season)
                )
    if args.ablate_noise:
        for component in ("classifier", "regressor"):
            for feat, row in audit[component].items():
                if row["noise"] and feat not in {a["drop"][0] for a in ablations}:
                    ablations.append(
                        ablation_gate(df, [feat], FEATURES, args.max_test_season)
                    )
        dropped_groups = {tuple(a["drop"]) for a in ablations}
        for cluster in audit["clusters"]:
            group = tuple(cluster["features"])
            if group not in dropped_groups:
                ablations.append(
                    ablation_gate(df, list(group), FEATURES, args.max_test_season)
                )
    audit["ablation"] = {"runs": ablations}

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    # Console summary: per-feature classifier/regressor verdicts.
    print("\nPer-feature noise/significance (q = 0.05):")
    print(f"{'feature':<28} {'clf p':>7} {'clf q':>7} {'reg p':>7} {'reg q':>7}")
    for feat in FEATURES:
        c, r = audit["classifier"][feat], audit["regressor"][feat]
        print(f"{feat:<28} {c['p']:>7.3f} {c['q']:>7.3f} {r['p']:>7.3f} {r['q']:>7.3f}")
    if ablations:
        print("\nAblation gate (removal improves a primary metric by >= 1 SE?):")
        for a in ablations:
            sig = "+".join(a["primary_improves"]) or "-"
            print(f"{','.join(a['drop']):<30} cut={a['cut']} primary_improves={sig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
