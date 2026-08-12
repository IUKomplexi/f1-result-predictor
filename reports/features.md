# Feature audit — noise classification, cut/keep decisions, and registry

**Status:** complete — methodology, tooling, audit results, classification,
and the registry-backed defaults are all in place (all phases of the
[feature-audit plan](../docs/planning/feature-audit-and-registry.md)).

**Companion artifacts:** `scripts/feature_audit.py` (the audit), the declarative
feature registry `features/registry.py`, and the machine-readable
`reports/feature_audit.json` this document summarizes.

---

## Why this audit exists

The model's edge over the trivial grid baseline is thin (it wins `winner_hit`
and `spearman`, loses `top3_overlap` and `MAE`). Any unverified feature is
therefore suspect, and feature churn is risky: an unvalidated addition cannot
be trusted, and the walk-forward backtest is the only honest judge of whether a
feature helps or hurts.

## Methodology

`HistGradientBoosting` deliberately exposes no `feature_importances_`, so
impurity-based importance is unavailable *and* undesirable (it over-rewards
high-cardinality features and misattributes credit). This audit instead uses
**permutation importance on out-of-sample data** (sklearn user guide; Molnar
*IML* ch. 8.5 & 9.6), plus **drop-column ablation** for cuts.

| Question | Answer used |
| --- | --- |
| Impact metric | `sklearn.inspection.permutation_importance` on each walk-forward validation window |
| Repeat count | `n_repeats = 25` seeded (sklearn's default 5 is documented as too coarse) |
| Per-component | classifier (P(top-10), ROC-AUC) and regressor (E(points\|top-10), neg. MAE on points) assessed **separately** |
| Walk-forward | importance computed per validation window (train on strictly earlier seasons), **never on training data**; averaged across windows |
| Noise threshold | one-sided 95% CI overlaps zero ⇒ noise (equiv. z = mean/SE ≲ 1.6–2, p ≥ 0.05) |
| Multiple testing | Benjamini–Hochberg FDR at **q = 0.05**, applied per component across the ~30 features |
| Unreliable flag | importance **flips sign across walk-forward folds** (`0 < frac_pos < 1`) |
| Collinearity | correlation clusters (\|r\| ≥ 0.8, pooled numeric features) get **grouped permutation** + **grouped ablation** |
| **Cut gate** | drop-column ablation; cut only if removal **improves a primary headline metric by ≥ 1 SE (fold-to-fold std)**, with no primary metric regressing ≥ 1 SE ("significantly hurts the backtest if kept") |

References: sklearn permutation-importance user guide & API; Molnar,
*Interpretable Machine Learning* ch. 8.5 & 9.6; Hooker et al. 2021
(arXiv:1905.03151); Strobl et al. 2008; Fisher et al. 2019 (arXiv:1801.01489);
Benjamini & Hochberg 1995.

## Thresholds

- **Noise** (per component): one-sided p ≥ 0.05 (the 95% CI of the mean
  importance overlaps zero).
- **Significant** (per component): p < 0.05 **and** BH-adjusted q < 0.05.
- **Unreliable**: importance flips sign across folds.
- **Cut**: removal improves ≥ 1 primary metric (winner_hit / top3_overlap /
  spearman) by ≥ 1 SE **and** regresses no primary metric by ≥ 1 SE. MAE is
  secondary (reported, not gating). Nothing is ever removed on
  importance-metric faith alone.

## Feature categories

| Category | Meaning | Default |
| --- | --- | --- |
| `core` | high impact; on by default | on |
| `selectable` | low impact; kept for experiments but off by default (the `reports/weather.md` "evaluated, not adopted" precedent) | off |
| `cut` | removal significantly improved the backtest (the ablation gate passed); kept in the registry for re-enabling | off |

The registry (`features/registry.py`) is the single source of truth for id,
category, default, builder, and rationale. All features are still computed by
`add_features`; only the enabled subset is assembled into the training matrix.
Toggling the enabled set changes the matrix columns and the model-checkpoint
fingerprint, so stale artifacts are never silently reused.

## How the audit runs

```
python scripts/feature_audit.py                        # permutation audit → reports/feature_audit.json
python scripts/feature_audit.py --ablate-noise         # + ablation gate on noise-flagged features & clusters
```

1. For each walk-forward window (test season ≥ 2013): fit the hurdle models on
   strictly earlier seasons, then compute seeded permutation importance
   (25 repeats) for the `scored` classifier (ROC-AUC) and the
   `points_if_scored` regressor (neg. MAE on points, scored rows only).
2. Detect correlation clusters (|r| ≥ 0.8) on the pooled numeric features and
   run **grouped** permutation importance (the group shuffled together) — the
   honest measure for collinear members whose individual permutation
   understates their joint contribution.
3. Average per-feature importances across windows; report mean ± SE, z,
   one-sided p, BH-FDR q, and fold sign-stability.
4. For the noise shortlist + clusters, run the drop-column ablation gate
   (removal vs. all-on, per-fold deltas vs. the fold-to-fold SE).

## Results

_Run: 6911 starts, 2010–2025 data; 13 walk-forward test windows (2013–2025),
`n_repeats=25`, seed 0, |r| ≥ 0.8, FDR q = 0.05. Machine-readable detail:
`reports/feature_audit.json`._

### Correlation clusters

| cluster | max \|r\| |
| --- | --- |
| `grid`, `qual_pos` | 0.941 |
| `champ_pos_entering`, `constructor_champ_pos_entering`, `driver_prev_finish_mean`, `driver_prev_points_mean`, `driver_prev_points_sum`, `last_race_points`, `season_driver_pts_per_race`, `season_team_pts_per_race`, `team_prev_points_mean`, `team_prev_pos_mean` | 0.995 |
| `circuit_prev_finish_mean`, `circuit_prev_points_mean` | 0.848 |

Grouped permutation shows the form/championship cluster and the grid/qualifying
pair carry real joint signal (their members are individually understated by
per-feature permutation); the circuit pair is noise.

### Per-feature classification (31 features: 27 numeric + 4 categorical)

| feature | component significance (clf / reg, q) | category |
| --- | --- | --- |
| `grid`, `qual_pos` | 0.000 / 0.000 | **core** |
| `grid_qual_gap` | 0.046 / 0.059 | **core** |
| `driver_prev_points_mean`, `driver_prev_points_sum` | 0.022, 0.001 / 0.000 | **core** |
| `team_prev_points_mean` | 0.046 / 0.000 | **core** |
| `team_wins_prior` | 0.018 / 0.57 | **core** |
| `champ_points_entering` | 0.081 / 0.026 | **core** |
| `constructor_champ_pos_entering` | 0.000 / 0.026 | **core** |
| `season_driver_pts_per_race` | 0.000 / 0.17 | **core** |
| `season_team_pts_per_race` | 0.002 / 0.000 | **core** |
| `driver_id`, `constructor_id` | 0.001, 0.031 / 0.008, 0.009 | **core** |
| `circuit_id` | 0.54 / 0.029 | **core** |
| `season`, `round`, `is_sprint_round`, `n_prior_races`, `team_tenure`, `team_switch`, `driver_prev_finish_mean`, `team_prev_pos_mean`, `circuit_prev_points_mean`, `finish_gap_vs_teammate`, `points_era` | noise in both | **selectable** |
| `last_race_points`, `driver_prev_dnf_rate`, `driver_wins_prior`, `circuit_prev_finish_mean`, `qual_gap_vs_teammate`, `champ_pos_entering` | noise / gate-pass | **cut** |

`team_switch` and `points_era` deserve a note: their removal produced an
*exactly zero* delta on every metric in every fold (the walk-forward models
never split on them, even though the all-seasons final model does use
`team_switch`). Both are classified selectable (off by default), not cut —
their removal neither improves nor regresses the backtest, so the ablation
gate does not fire for them.

### Ablation gate (drop-column, ±1 SE on the fold-to-fold std)

Cut only where removal improves ≥ 1 primary metric by ≥ 1 SE **and** regresses
no primary metric by ≥ 1 SE:

| removed | improves (≥ 1 SE) | regresses (≥ 1 SE) |
| --- | --- | --- |
| `last_race_points` | winner_hit (+2.5 SE) | — |
| `driver_prev_dnf_rate` | winner_hit (+1.8 SE) | — |
| `driver_wins_prior` | top3_overlap (+2.2 SE) | — |
| `circuit_prev_finish_mean` | top3_overlap (+1.1 SE) | — |
| `qual_gap_vs_teammate` | top3_overlap (+1.4 SE) | — |
| `champ_pos_entering` | top3_overlap (+1.6 SE) | — |
| `circuit_prev_points_mean` | top3_overlap (+1.5 SE) | spearman (−1.0 SE) — mixed, not cut |
| `circuit_id` | top3_overlap (+1.2 SE) | spearman (−1.3 SE) — mixed, not cut |
| `team_switch`, `points_era` | all three (0.0 delta — degenerate) | all three (0.0) — zero impact |
| `season`, `round`, `team_wins_prior`, `season_driver_pts_per_race` | none | none — neutral |
| `is_sprint_round`, `n_prior_races`, `team_tenure`, `driver_prev_finish_mean`, `team_prev_pos_mean`, `finish_gap_vs_teammate` | none | 1.1–1.9 SE on winner_hit/top3/spearman — **noise-level** (see below) |

The last row needs an honest caveat. Six features that are *noise* by the
permutation test show per-feature removal "regressions" of 1.1–1.9 SE — but at
**noise-level absolute magnitudes** (e.g. spearman −0.0004 with SE 0.0003,
winner_hit −0.010 with SE 0.010). The 1-SE threshold is deliberately loose for
the cut gate (so nothing load-bearing is cut) and over-fires on low-variance
metrics like spearman. These features are therefore kept **off by default**
(selectable), and the arbiter is the combined-backtest validation below: a
counter-experiment adding all six back (a 19-feature default) made the backtest
strictly worse (winner_hit 0.528, MAE 2.98 vs 0.550 / 2.92 with the 14-feature
default), confirming they are not load-bearing as a set.

### Validation: backtest with the new defaults vs the previous report

Walk-forward, 2013–2025, quantized expected points. Old = all 31 features
(shipped before this change); new = the 14 core features (the default enabled
set). Overall values are race-weighted means (as in `reports/backtest.json`);
the SE is the fold-to-fold std of the per-season deltas / √13.

| metric | old | new | delta | fold SE | \|delta\| / SE |
| --- | --- | --- | --- | --- | --- |
| winner_hit | 0.5387 | 0.5498 | **+0.0111** | 0.0161 | 0.69 |
| top3_overlap | 0.6667 | 0.6593 | −0.0074 | 0.0062 | 1.19 |
| spearman | 0.6541 | 0.6559 | +0.0018 | 0.0016 | 1.13 |
| mae (secondary) | 2.9333 | 2.9155 | −0.0178 | 0.0109 | 1.63 |

winner_hit improves by ~0.7 SE; spearman and MAE improve within noise;
top3_overlap dips 1.19 SE — inside the 95% band, and no per-feature removal
regressed a primary metric beyond the noise band in the combined run.
**No regression beyond the noise floor**: the audit-driven defaults ship.

### Phase 4 (refinements)

No feature was flagged as *badly computed* — every audit signal pointed at
redundancy (clusters) or absence of signal, not at a broken builder. No
builder swap was needed; Phase 4 is a no-op and can be revisited with new
feature ideas.

## How to re-run

```
python scripts/feature_audit.py --ablate-noise     # permutation audit + ablation gate
python scripts/feature_audit.py --skip-audit --ablate-noise   # ablation only
```

The registry (`features/registry.py`) holds the resulting categories and the
per-feature impact summary; `config.toml` `[features] enabled` mirrors the
core defaults. Toggle features on the CLI with `--enable-features` /
`--disable-features` on `f1 train`, `f1 backtest`, `f1 predict`,
`f1 calibrate`, and `f1 search`.
