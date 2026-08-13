# AGENTS.md — F1 Result Predictor

Predicts **points per driver** for an F1 race from strictly pre-race info (grid, qualifying, form, circuit history) via a zero-inflated hurdle model (`E[points] = P(top-10) × E(points|top-10)`), trained on 2010–2026 Jolpica F1 API data. Python ≥3.12, managed with **uv**; CLI + FastAPI + Preact dashboard; Dockerized. See `README.md` — the canonical doc (setup, usage, repository map, invariants, extending).

## Commands (run from repo root — config/report paths are CWD-relative)

```bash
uv sync --all-extras          # install + test/lint/web extras into .venv
f1 fetch                      # one-time API fetch → data/raw/ (offline afterwards)
uv run pytest -q              # test suite — fully offline (recorded fixtures)
uv run ruff check .           # lint (0 errors expected)
f1 predict [--season S --round R] [--grid qual.csv]   # next race → reports/prediction.md
f1 train                      # train → data/model/hurdle.joblib
f1 calibrate                  # fit isotonic calibrators (run after every f1 train)
f1 backtest [--no-quantize]   # walk-forward backtest → reports/backtest.md/.json
f1 web [--port 8080]          # FastAPI + dashboard (needs web extra)
docker compose up --build     # self-contained dashboard on :8080
```

One console script, `f1`, registered in `pyproject.toml [project.scripts]`; its
subcommand handlers in `f1core/cli.py` delegate to keyword-only `run_* -> dict`
wrappers (shared by the web job runner, so terminal and dashboard never drift).


## Architecture

Single-direction deps, all packages import shared helpers from `f1core` (no sys.path hacks):
- `f1data/` — polite cached Jolpica client (`F1Client`) + normalized fetchers (`fetch_season`, `fetch_calendar`, …)
- `f1core/` — shared core: `predict.py` (`predict_race`, `get_prediction`, `format_report`, `format_console`), `config.py` (`DEFAULTS`, `load_config`, **`save_config`/`validate_config`/`SCHEMA`** TOML writer), `httpclient.py` (`CachedHTTPClient` base), `reporting.py` (`to_md`, `rank_by`), `cli.py` (the single `f1` CLI + subcommand handlers)
- `features/build.py` — per-start dataset: `build_dataset`, `add_features`, `assemble`; strictly pre-race features + points target
- `model/` — `train.py` (hurdle: HGB classifier + regressor), `evaluate.py` (walk-forward backtest vs grid/championship/zero), `calibrate.py` (isotonic, per-target, deployed only where it improves hold-out Brier). Each exposes a keyword-only `run_* -> dict` wrapper called by the `f1` CLI and the web job runner.
- `f1web/` — `app.py` (`create_app`, JSON API + built-SPA host), `jobs.py` (`JobManager`, threading worker + single-job queue + `reports/jobs/*.json` history) and Preact SPA in `f1web/ui/` (Vite + TS; `@preact/preset-vite` + `preact/compat`); tabs: Status, Race, Race History, Data, Train, Backtest, Settings — deep-linkable via URL hashes (`#/race`, `#/backtest`, …; refresh/back-button safe). Tab components share the `TabProps` contract (`App.tsx`): `onNavigate(tabId, state?)` for cross-tab jumps (e.g. Race History → a specific race), `navState` payload, and `setNavigateGuard` so a tab can veto navigation (Settings' unsaved-changes dialog). Shared UI: `ui/JobRunner`, `ui/JobsWidget` (dialog-grade panel: focus management, Escape, outside-click, `aria-live` status announcements; header shows a running-job pill), `ui/ConfirmDialog`, `ui/RefreshToggle`, `ui/Chart`. The Train job also calibrates. No UI *unit* suite (known gap; `npm run build`/`npm run lint` are the UI checks)
- `scripts/` — `download_fixtures.py` (regenerates the tracked test fixtures)
- `tests/` — offline suite, recorded fixtures (`tests/fixtures/`), `test_e2e.py` runs the full pipeline

Data flow: `data/raw/*.json` → `data/features.parquet` → `data/model/*.joblib`. All `data/` caches are gitignored and regenerable; model checkpoints carry feature-set fingerprints. The dataset cache validates on **feature fingerprint + season coverage + raw-cache mtime** (`_raw_cache_newer_than` in `build_dataset`): newly fetched raw data automatically invalidates the parquet, so Train/Backtest never read a stale dataset.

## Conventions

- **Leakage is the hard invariant** (unit-tested): rolling/cumulative features use `shift(1)`; teammate gaps rolled over prior races only; walk-forward trains on strictly earlier seasons. Never break it.
- Modules start with `from __future__ import annotations`; docstrings on public functions.
- Ruff: line-length 100, `select = E,W,F,I,UP,B,RUF,PIE,ISC,BLE`. No type-check gate; pandas is untyped — scoped `# type: ignore[reportX]` comments are kept as documentation (inert to ruff).
- Tests must stay offline; suite enforces `filterwarnings = ["error::DeprecationWarning"]` — deprecation suppression lives only at checkpoint I/O in `model/train.py`.
- API errors are uniformly `{"error": ...}` (incl. 422 and the `/assets` path-traversal guard).
- `config.toml` holds **overrides**; the built-in defaults in `f1core/config.py` `DEFAULTS` are the baseline, and every `run_*` wrapper + CLI flag resolves from config when not explicitly passed. The dashboard writes the full effective config back via `save_config` (atomic). CLI flags override config. Per-request prediction overrides (`POST /api/predict`) are **ephemeral** — applied in memory only, never written to disk.
- Adding a config field: add it to `DEFAULTS` in `f1core/config.py` (the field type is inferred from the value; add a `_SCHEMA_HINTS` entry for help/min/max); the Settings form, TOML writer, and validation pick it up automatically.
- Pipeline steps expose a keyword-only `run_* -> dict` (JSON-safe) that the `f1` subcommands and web job runner call; a web job wraps it via a handler registered in `f1web/jobs.py`. Jobs are async/threaded, one at a time, results + log recorded in memory + `reports/jobs/*.json`.
- The model's HGB hyperparameters are config-driven: train reads `[model.params]` (defaults in `f1core/config.py` `DEFAULTS`); a params change silently requires retrain (surfaced as a "retrain needed" hint in the UI). Feature-list edits change the checkpoint fingerprint (existing behavior).
- Add a pre-race feature: helper in `features/build.py` + register in `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` + test in `tests/test_features.py`; bump the dataset cache version on schema change.
- UI checks: `npm run build` (`tsc -b && vite build`) and `npm run lint` (oxlint) in `f1web/ui` must stay clean; the built `f1web/ui/dist` is gitignored and rebuilt by the Dockerfile.
- UI navigation goes through the shared contract: tab switches via the hash route + `onNavigate(tabId, state?)`; never ad-hoc `useState`-to-parent prop drilling. A tab that must not lose edits registers a `setNavigateGuard` veto instead of intercepting clicks itself.
- `RefreshToggle` (cache override) belongs only on **Data** ("re-fetch from API") and **Race History** ("recompute predictions") — Train/Backtest rely on the automatic mtime invalidation; do not re-add the toggle there.
- CI (`.github/workflows/ci.yml`): pytest on 3.12/3.13 + ruff on push to `main` and PRs.

## Notes

After completing each working phase, generate a commit message adhering to the **Conventional Commits** specification and commit via **GH CLI**, then wait for CI to run and verify the result.

* **Format:** `<type>(<scope>)[!]: <description>`
* **Primary Types:** `feat` (new features), `fix` (bug fixes), `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
* **Scope:** Optional noun specifying the codebase module (e.g., `fix(features): rolling offset bug`).
* **Breaking Changes:** Indicate with `!` before the colon (e.g., `feat(api)!: change payload schema`) or a `BREAKING CHANGE: <description>` footer.
* **Body & Footers:** Optional contextual body paragraphs and footers (e.g., `Closes #123`) MUST be separated from the header by a single blank line.