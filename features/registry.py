"""Declarative feature registry — the single source of truth for feature selection.

Every model feature (the 27 numeric + 4 categorical columns) has one
:class:`FeatureSpec` entry here, carrying its id, category, default state,
builder stage, rationale, and audit impact. The registry is authoritative:
``features/build.py`` computes every feature (all are always computed), and
the *enabled subset* — resolved by :func:`enabled_features` from the registry
defaults, ``config.toml`` ``[features] enabled``, and CLI overrides — is what
``model/train.prepare`` assembles into the training matrix.

Categories (assigned by the audit in ``reports/features.md``; all 31 features
were measured with seeded permutation importance per walk-forward window and
drop-column ablation, 2013-2025 test windows):

* ``core`` — high impact (survived BH-FDR at q=0.05 in at least one hurdle
  component); on by default.
* ``selectable`` — low impact (noise in both components); kept for experiments
  but off by default (the ``reports/weather.md`` "evaluated, not adopted"
  precedent).
* ``cut`` — removal significantly improved the backtest (the ±1 SE ablation
  gate passed); kept in the registry so it can be re-enabled.

Toggling the enabled set changes the model-checkpoint fingerprint (and thus
invalidates checkpoints), so stale artifacts are never silently reused.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES

Category = Literal["core", "selectable", "cut"]
CATEGORIES: tuple[Category, ...] = ("core", "selectable", "cut")


@dataclass(frozen=True)
class FeatureSpec:
    """One registered model feature."""

    id: str
    category: Category
    kind: Literal["numeric", "categorical"]
    builder: str
    rationale: str
    impact: str = ""
    default: bool = field(init=False)

    def __post_init__(self) -> None:
        # core = on by default; selectable/cut = kept but off by default.
        object.__setattr__(self, "default", self.category == "core")


# --------------------------------------------------------------------------
# The registry (27 numeric + 4 categorical).
# Categories: core / selectable / cut — see reports/features.md.
# --------------------------------------------------------------------------

_REGISTRY: list[FeatureSpec] = [
    # --- grid / qualifying (assemble + _add_baseline) ---
    FeatureSpec("grid", "core", "numeric", "assemble",
                "starting grid slot; the strongest single pre-race signal",
                "significant in both components (q=0.000); grouped with qual_pos (r=0.94)"),
    FeatureSpec("qual_pos", "core", "numeric", "assemble",
                "qualifying position (falls back to grid when missing)",
                "significant in both components (q=0.000); grouped with grid (r=0.94)"),
    FeatureSpec("grid_qual_gap", "core", "numeric", "_add_baseline",
                "grid minus qualifying position (penalties / setup shifts)",
                "significant in classifier (q=0.046); borderline in regressor (q=0.059)"),
    FeatureSpec("season", "selectable", "numeric", "assemble",
                "calendar season; captures regulation-era drift",
                "noise in both components (p=1.000); removal is neutral in the gate"),
    FeatureSpec("round", "selectable", "numeric", "assemble",
                "round number within the season (track-type order effect)",
                "noise in both components (clf p=0.49, reg p=0.09); removal neutral"),
    FeatureSpec("is_sprint_round", "selectable", "numeric", "assemble",
                "whether the weekend has a sprint race (session-shape effect)",
                "noise in both components; removal regresses spearman 1.4 SE at "
                "noise-level magnitude; constant in pre-2021 windows"),
    # --- driver history (_add_driver_history) ---
    FeatureSpec("n_prior_races", "selectable", "numeric", "_add_driver_history",
                "career starts so far (experience)",
                "noise in both components; removal regresses spearman 1.3 SE at "
                "noise-level magnitude"),
    FeatureSpec("team_tenure", "selectable", "numeric", "_add_driver_history",
                "races with the current team (settling-in effect)",
                "noise in both components; removal regresses winner_hit 1.1 SE at "
                "noise-level magnitude"),
    FeatureSpec("team_switch", "selectable", "numeric", "_add_driver_history",
                "1 when the driver changed teams since the last race",
                "noise in both components (p=1.000); zero ablation delta in every window"),
    FeatureSpec("driver_prev_finish_mean", "selectable", "numeric", "_add_driver_history",
                "rolling mean finishing position (window 5, strictly prior)",
                "noise in both components; removal regresses winner_hit 1.9 SE at "
                "noise-level magnitude; collinear with the form cluster (r=0.995)"),
    FeatureSpec("driver_prev_points_mean", "core", "numeric", "_add_driver_history",
                "rolling mean points per race (window 5, strictly prior)",
                "significant in both components (q=0.022 / 0.000)"),
    FeatureSpec("driver_prev_points_sum", "core", "numeric", "_add_driver_history",
                "rolling points sum (window 5, strictly prior)",
                "significant in both components (q=0.001 / 0.000)"),
    FeatureSpec("last_race_points", "cut", "numeric", "_add_driver_history",
                "points scored in the driver's previous race",
                "noise in both components; removal improves winner_hit by 2.5 SE (gate)"),
    FeatureSpec("driver_prev_dnf_rate", "cut", "numeric", "_add_driver_history",
                "rolling DNF rate (window 5, strictly prior)",
                "noise in both components; removal improves winner_hit by 1.8 SE (gate)"),
    FeatureSpec("driver_wins_prior", "cut", "numeric", "_add_driver_history",
                "career wins strictly before this race",
                "significant in classifier (q=0.046) but removal improves "
                "top3_overlap by 2.2 SE (gate)"),
    # --- constructor history (_add_constructor_history) ---
    FeatureSpec("team_prev_points_mean", "core", "numeric", "_add_constructor_history",
                "rolling team points per race, both cars (window 5, strictly prior)",
                "significant in both components (q=0.046 / 0.000)"),
    FeatureSpec("team_prev_pos_mean", "selectable", "numeric", "_add_constructor_history",
                "rolling team mean finishing position (window 5, strictly prior)",
                "noise in both components; removal regresses spearman 1.3 SE at "
                "noise-level magnitude; collinear with the form cluster (r=0.995)"),
    FeatureSpec("team_wins_prior", "core", "numeric", "_add_constructor_history",
                "cumulative team wins strictly before this race",
                "significant in classifier (q=0.018); noise in regressor"),
    # --- circuit history (_add_circuit_history) ---
    FeatureSpec("circuit_prev_finish_mean", "cut", "numeric", "_add_circuit_history",
                "driver's rolling mean finish at this circuit (window 3, strictly prior)",
                "noise in both components; removal improves top3_overlap by 1.1 SE (gate)"),
    FeatureSpec("circuit_prev_points_mean", "selectable", "numeric", "_add_circuit_history",
                "driver's rolling mean points at this circuit (window 3, strictly prior)",
                "noise in both components; removal improves top3_overlap 1.5 SE but "
                "regresses spearman 1.0 SE (mixed); grouped with finish variant (r=0.85)"),
    # --- teammate gaps (_add_teammate_gaps) ---
    FeatureSpec("finish_gap_vs_teammate", "selectable", "numeric", "_add_teammate_gaps",
                "rolling finish-position gap vs teammate (window 5, strictly prior)",
                "noise in both components; removal regresses top3_overlap 1.3 SE and "
                "spearman 1.4 SE at noise-level magnitude"),
    FeatureSpec("qual_gap_vs_teammate", "cut", "numeric", "_add_teammate_gaps",
                "rolling qualifying-position gap vs teammate (window 5, strictly prior)",
                "noise in both components; removal improves top3_overlap by 1.4 SE (gate)"),
    # --- championship (_add_championship) ---
    FeatureSpec("champ_points_entering", "core", "numeric", "_add_championship",
                "championship points entering the race (incl. current-round sprint)",
                "significant in regressor (q=0.026); near-significant in classifier (q=0.081)"),
    FeatureSpec("champ_pos_entering", "cut", "numeric", "_add_championship",
                "championship rank entering the race",
                "significant in classifier (q=0.002) but redundant with champ points; "
                "removal improves top3_overlap by 1.6 SE (gate)"),
    FeatureSpec("constructor_champ_pos_entering", "core", "numeric", "_add_constructor_history",
                "constructor championship rank entering the round",
                "significant in both components (q=0.000 / 0.026)"),
    FeatureSpec("season_driver_pts_per_race", "core", "numeric", "_add_championship",
                "driver points per race entering the round (season pace)",
                "significant in classifier (q=0.000); noise in regressor (p=0.08)"),
    FeatureSpec("season_team_pts_per_race", "core", "numeric", "_add_constructor_history",
                "team points per race entering the round (championship pace)",
                "significant in both components (q=0.002 / 0.000)"),
    # --- categorical ---
    FeatureSpec("driver_id", "core", "categorical", "assemble",
                "driver identity (talent-level signal)",
                "significant in both components (q=0.001 / 0.008)"),
    FeatureSpec("constructor_id", "core", "categorical", "assemble",
                "constructor identity (car-quality signal)",
                "significant in both components (q=0.031 / 0.009)"),
    FeatureSpec("circuit_id", "core", "categorical", "assemble",
                "circuit identity (track-characteristic signal)",
                "significant in regressor (q=0.029); ablation mixed (top3_overlap "
                "+1.2 SE vs spearman -1.3 SE, net-neutral)"),
    FeatureSpec("points_era", "selectable", "categorical", "_add_era_and_targets",
                "points-system era (pre/post 2019)",
                "noise in both components (p=1.000); zero ablation delta in every window"),
]

REGISTRY: tuple[FeatureSpec, ...] = tuple(_REGISTRY)
_REGISTRY_BY_ID = {f.id: f for f in REGISTRY}


def _validate_registry() -> None:
    """Fail fast if the registry drifts from the feature lists in build.py."""
    expected = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    actual = {f.id for f in REGISTRY}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(
            f"registry out of sync with features/build.py: "
            f"missing={missing} extra={extra}"
        )
    # Order must match NUMERIC_FEATURES + CATEGORICAL_FEATURES so the
    # "all features" fingerprint is identical from either source.
    if [f.id for f in REGISTRY] != NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        raise AssertionError(
            "registry order must match NUMERIC_FEATURES + CATEGORICAL_FEATURES "
            "(fingerprint stability)"
        )
    for f in REGISTRY:
        if f.category not in CATEGORIES:
            raise AssertionError(f"{f.id}: invalid category {f.category!r}")
        is_num = f.id in NUMERIC_FEATURES
        if (f.kind == "numeric") != is_num:
            raise AssertionError(
                f"{f.id}: kind {f.kind!r} does not match the feature lists"
            )


_validate_registry()


# --------------------------------------------------------------------------
# Selection helpers
# --------------------------------------------------------------------------

def all_feature_ids() -> list[str]:
    """Every registered feature id, in registry order."""
    return [f.id for f in REGISTRY]


def default_enabled() -> list[str]:
    """The registry-default enabled set (core features, in registry order)."""
    return [f.id for f in REGISTRY if f.default]


def feature_fingerprint(features: Sequence[str]) -> str:
    """Stable short hash of an ordered feature list (cache/checkpoint key)."""
    payload = "\n".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _check_known(features: Sequence[str], what: str) -> None:
    unknown = [f for f in features if f not in _REGISTRY_BY_ID]
    if unknown:
        raise ValueError(f"unknown {what} feature(s): {unknown}; see features/registry.py")


def enabled_features(
    cfg: dict | None = None,
    enable: Sequence[str] = (),
    disable: Sequence[str] = (),
) -> list[str]:
    """Resolve the effective enabled feature set (registry order).

    * ``cfg["features"]["enabled"]`` — the explicit list from ``config.toml``
      (``None``/absent ⇒ registry defaults);
    * ``enable`` / ``disable`` — CLI overrides applied on top.

    The result is validated and returned in registry order, so the model
    matrix and fingerprints are deterministic.
    """
    explicit = ((cfg or {}).get("features") or {}).get("enabled")
    if not explicit:  # None or an empty list ⇒ registry defaults
        enabled = set(default_enabled())
    else:
        _check_known(list(explicit), "enabled")
        enabled = set(explicit)
    _check_known(list(enable), "--enable-features")
    _check_known(list(disable), "--disable-features")
    enabled.update(enable)
    enabled.difference_update(disable)
    return [f.id for f in REGISTRY if f.id in enabled]
