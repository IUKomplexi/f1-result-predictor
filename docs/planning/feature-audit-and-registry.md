# Plan — Feature audit, noise classification, and plug-in registry

Status: **approved** — planned, not yet implemented.

Owner-facing. The filesystem: decides what shapes feature work in the repo. Any
consequential change to this scope should be re-agreed with the user.

---

## The problem

The model's edge over the trivial grid baseline is thin (wins `winner_hit` and
`spearman`, loses `top3_overlap` and `MAE`). That makes any unverified feature
suspicious, and it makes feature **churn** risky: we cannot trust an unvalidated
addition, and the walk-forward gate is the only honest judge.

We want to be able to turn features on and off at will (web UI later, plumbing
now), including a **categorized registry** in which low-impact features are kept
but off by default (the `weather.md` "evaluated, not adopted" precedent), and
only features whose removal significantly improves the backtest are actually
cut.

## Methodology (research verdict)

`HistGradientBoosting` deliberately exposes **no `feature_importances_`**, so
impurity-based importance is unavailable *and* undesirable (over-rewards
high-cardinality features, misattributes credit). Standard practice instead uses
**permutation importance on out-of-sample data**, plus ablation for cuts.

| Question | Standard answer we will use |
|---|---|
| Impact metric | `sklearn.inspection.permutation_importance` on each walk-forward validation window |
| Repeat count | `n_repeats=25` seeded (sklearn default of 5 is documented as too coarse) |
| **Noise threshold** | importance's one-sided 95% CI overlaps zero ⇒ noise (equiv. z = mean/SE ≲ 1.6–2) |
| Multiple-testing on ~30 features | Benjamini–Hochberg FDR at **q=0.05** |
| Unreliable flag | importance **flips sign across walk-forward folds** |
| **Cut gate** | drop-column ablation; cut only if removal **improves a headline metric by ≥ 1 SE (fold-to-fold std)** (= "significantly hurts backtest if kept") |
| Collinearity trap | correlation clusters (\|r\| ≥ 0.8) get grouped permutation + grouped ablation |
| Per-component | assess classifier (P(top-10)) and regressor (E(points\|top-10)) **separately** |
| Walk-forward | importance computed per validation window, averaged across windows; never on training data |

References: sklearn permutation-importance user guide & API; Molnar, *Interpretable
Machine Learning* ch. 8.5 & 9.6; Hooker et al. 2021 (arXiv:1905.03151); Strobl et al.
2008; Fisher et al. 2019 (arXiv:1801.01489); Benjamini & Hochberg 1995.

## Output contract

- `reports/features.md` documents the method, the thresholds, per-feature
  classification, and the cut/keep decisions — the `reports/weather.md` precedent.
- A declarative **feature registry** (`features/registry.py`) is the single
  source of truth for feature id, category, default, builder, and rationale.
- Training, backtest, and predict consume the **enabled** subset. Toggling
  invalidates the dataset cache and the model fingerprint correctly.
- Defaults: **core** = high impact & on by default; **selectable** = low impact,
  kept but off by default; **cut** = removed only where the ablation gate passes.
- Only features whose removal significantly improves the backtest are cut. Nothing
  is ever removed on importance-metric faith alone.

## Out of scope (this plan)

- The web UI toggle itself (future iteration; it consumes the same registry).
- New *feature ideas* from outside the current 26 numeric + 4 categorical set,
  except concrete refinements to features the audit flags as badly computed.

---

## Phases

1. **Build the audit tooling + methodology report**
   - Add `scripts/feature_audit.py`: per walk-forward window, seeded
     `permutation_importance` (`n_repeats=25`) for the classifier and regressor
     separately; emit per-feature mean ± SE across folds, z-scores,
     fold sign-stability, and per-component tables to JSON.
   - Add correlation-cluster detection (|r| ≥ 0.8) and grouped-permutation
     importance for those clusters.
   - Write `reports/features.md`: method, thresholds, cut gate, references.

2. **Audit the current features and classify**
   - Run the audit on the 26 numeric + 4 categorical features.
   - Drop-column ablation on the noise shortlist + correlated groups, gated by
     the ±1 SE noise floor across all four headline metrics.
   - Assign each feature a category — **core** / **selectable** / **cut** — and
     flag computation-suspect features; record in `reports/features.md`.

3. **Plug-in registry + plumbing**
   - Create `features/registry.py` as single source of truth: id, category,
     default on/off, builder, impact summary, rationale. All features are still
     computed in the pipeline; only the enabled subset is assembled into the
     training matrix.
   - Wire selection into `f1-train`, `f1-backtest`, `f1-predict` via `config.toml`
     (`[features] enabled = [...]`) and CLI overrides (`--enable-features`,
     `--disable-features`).
   - Key the dataset cache and the model-checkpoint fingerprint to the
     enabled-feature-set hash so toggling invalidates caches and never silently
     reuses stale artifacts.
   - Initial defaults mirror today's all-on behavior; audit-driven defaults
     (core on, selectable off) land in the same change.

4. **Refine flagged features (backtest-gated)**
   - Implement concrete improvements for each computation-suspect feature
     (better rolling windows, per-circuit weighting, cleaner encodings) behind
     its registry entry; each refinement must pass the ±1 SE backtest gate or be
     reverted (a reversible builder swap).

5. **Validation + docs**
   - Re-run the walk-forward backtest with the new defaults; compare against the
     prior report and the grid baseline; confirm no regression beyond the noise floor.
   - Add tests: registry completeness vs `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES`;
     toggling changes matrix columns and fingerprint; defaults round-trip; leakage
     invariants still pass — all offline.
   - Update `README.md`/`OVERVIEW.md` with the registry, categories, and how
     classification is determined.
