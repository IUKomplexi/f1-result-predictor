# Web UI usability: weekend scoreboard + self-explanatory pipeline

Status: **implemented** (2026) — direction A + B from the idea-refine session.

## Problem Statement

How might we make the F1 predictor dashboard self-explanatory and pleasant to
use, so the core loop — check a race prediction, run a pipeline step, judge how
accurate it was — needs no documentation and no guessing, for the owner **and
friends watching race weekends**?

## Recommended Direction

Two complementary fronts, both within "frontend + small API additions":

- **A · Weekend scoreboard** — the Race tab shows predicted-vs-actual deltas
  per driver plus headline accuracy metrics, so the model's value is visible at
  a glance when sharing around a race.
- **B · Self-explanatory pipeline** — a status-driven onboarding checklist built
  on the already-existing `/api/status`, plus a persistent job-queue widget with
  live logs (small `GET /api/jobs` enrichment).

B gets you running; A rewards you.

## Key Assumptions (validated)

- [x] `/api/status` flags cover every pipeline stage the onboarding needs
      (raw cache, checkpoint, calibrators, backtest).
- [x] Actuals are present on verified predictions, so scoreboard metrics need no
      extra API surface (verified live: 20/20 rows with actuals).
- [x] Jobs run from the UI during race weekends (the queue widget earns its
      place) — the tabs exist for exactly that.

## MVP Scope (delivered)

- Scoreboard card on verified races: top-10 hit rate, podium overlap, winner
  pick, MAE, model-vs-actual podium lines.
- Colored per-driver `Δ pts` column (actual − expected) with legend.
- Status tab: 4-step pipeline checklist (fetch → train → calibrate → backtest)
  with Run / Open actions; landing tab whenever any artifact is missing.
- Per-tab prerequisite hints on Train / Calibration / Backtest / Search.
- Header Jobs widget on every tab: running/queued count, per-job status +
  elapsed time (ticks per second), auto-scrolling log, copy-to-clipboard.
- Backend: `GET /api/jobs` slim entries gained `elapsed_s` + `log_lines`.

## Not Doing (and why)

- URL state / deep-linking — not a user-selected pain; bigger architectural touch.
- Settings dirty-state/discard — not selected; `config.toml` risk is real but separate.
- Job cancel — `run_*` steps are atomic; cancel is best-effort-only, low value
  for a single-user queue. Progress is stage-level (status + elapsed + log tail),
  no percent bar.
- Outcome distributions (README's suggestion) — needs modeling + calibration
  rework, contradicts the "small API additions" boundary.
- vitest UI suite — repo's known gap, but outside this scope's size.
