# AGENTS.md — F1 Result Predictor

Predicts **points per driver** for an F1 race from strictly pre-race info (grid, qualifying, form, circuit history) via a zero-inflated hurdle model (`E[points] = P(top-10) × E(points|top-10)`), trained on 2010–2025 Jolpica F1 API data. Python ≥3.12, managed with **uv**; CLI + FastAPI + React dashboard; Dockerized. See `README.md` (usage/results) and `OVERVIEW.md` (repository map, invariants, extending).

## Commands (run from repo root — config/report paths are CWD-relative)

```bash
uv sync --all-extras          # install + test/lint/web extras into .venv
uv run scripts/fetch_all.py   # one-time API fetch → data/raw/ (offline afterwards)
uv run pytest -q              # test suite — 139 tests, fully offline (recorded fixtures)
uv run ruff check .           # lint (0 errors expected)
uv run pyright                # type check, basic mode (0 errors expected)
f1-predict [--season S --round R] [--grid qual.csv]   # next race → reports/prediction.md
f1-train                      # train → data/model/hurdle.joblib
f1-calibrate                  # fit isotonic calibrators (run after every f1-train)
f1-backtest [--no-quantize]   # walk-forward backtest → reports/backtest.md/.json
f1-search --n 16 --max-test-season 2019   # hyperparameter tuning
f1-web [--port 8080]          # FastAPI + dashboard (needs web extra)
docker compose up --build     # self-contained dashboard on :8080
```

Console scripts registered in `pyproject.toml [project.scripts]`; each CLI module has `main() -> int` that delegates to a keyword-only `run_* -> dict` wrapper (shared by the web job runner, so terminal and dashboard never drift).

## Architecture

Single-direction deps, all packages import shared helpers from `f1core` (no sys.path hacks):
- `f1data/` — polite cached Jolpica client (`F1Client`) + normalized fetchers (`fetch_season`, `fetch_calendar`, …)
- `f1core/` — shared core: `predict.py` (`predict_race`, `get_prediction`, `format_report`, CLI), `config.py` (`DEFAULTS`, `load_config`, **`save_config`/`validate_config`/`SCHEMA`** TOML writer), `httpclient.py` (`CachedHTTPClient` base), `reporting.py` (`to_md`, `rank_by`)
- `features/build.py` — per-start dataset: `build_dataset`, `add_features`, `assemble`; strictly pre-race features + points target
- `model/` — `train.py` (hurdle: HGB classifier + regressor), `evaluate.py` (walk-forward backtest vs grid/championship/zero), `calibrate.py` (isotonic, per-target, deployed only where it improves hold-out Brier), `search.py` (walk-forward-validated tuning). Each has a `run_*` wrapper + `main()` delegator.
- `f1web/` — `app.py` (`create_app`, JSON API + built-SPA host), `jobs.py` (`JobManager`, threading worker + single-job queue + `reports/jobs/*.json` history) and React SPA in `f1web/ui/` (Vite + TS); tabs: Next Race, Race History, Backtest, Calibration, Pipeline, Settings, Season. No UI *unit* suite (known gap; `npm run build`/`npm run lint` are the UI checks)
- `f1weather/` — Open-Meteo layer, **evaluated but NOT adopted** (kept deliberately; see `reports/weather.md`)
- `scripts/` — `fetch_all.py`, `fetch_weather.py`, `download_fixtures.py`
- `tests/` — offline suite, recorded fixtures (`tests/fixtures/`), `test_e2e.py` runs the full pipeline

Data flow: `data/raw/*.json` → `data/features.parquet` → `data/model/*.joblib`. All `data/` caches are gitignored and regenerable; model checkpoints carry feature-set fingerprints.

## Conventions

- **Leakage is the hard invariant** (unit-tested): rolling/cumulative features use `shift(1)`; teammate gaps rolled over prior races only; walk-forward trains on strictly earlier seasons. Never break it.
- Modules start with `from __future__ import annotations`; docstrings on public functions.
- Ruff: line-length 100, `select = E,W,F,I,UP,B,RUF,PIE,ISC,BLE`. Pyright: basic mode (see `pyrightconfig.json`); pandas is untyped — scoped `# type: ignore[reportX]` comments are the accepted pattern.
- Tests must stay offline; suite enforces `filterwarnings = ["error::DeprecationWarning"]` — deprecation suppression lives only at checkpoint I/O in `model/train.py`.
- API errors are uniformly `{"error": ...}` (incl. 422 and the `/assets` path-traversal guard).
- `config.toml` is the **single source of truth** (mirrors `f1core/config.py` `DEFAULTS`); the dashboard writes it back via `save_config` (atomic). CLI flags override config. Per-request prediction overrides (`POST /api/predict`) are **ephemeral** — applied in memory only, never written to disk.
- Adding a config field: add to `DEFAULTS` + a `SCHEMA` descriptor + a `validate_config` check in `f1core/config.py`; the Settings form and TOML writer pick it up automatically.
- Pipeline steps expose a keyword-only `run_* -> dict` (JSON-safe) that `main()` delegates to; a web job wraps it via a handler registered in `f1web/jobs.py`. Jobs are async/threaded, one at a time, results + log recorded in memory + `reports/jobs/*.json`.
- The model's HGB hyperparameters are config-driven: train reads `[model.params]` (fallback `DEFAULT_PARAMS`); a params change silently requires retrain (surfaced as a "retrain needed" hint in the UI). Feature-list edits change the checkpoint fingerprint (existing behavior).
- Add a pre-race feature: helper in `features/build.py` + register in `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` + test in `tests/test_features.py`; bump the dataset cache version on schema change.
- UI checks: `npm run build` (`tsc -b && vite build`) and `npm run lint` (oxlint) in `f1web/ui` must stay clean; the built `f1web/ui/dist` is gitignored and rebuilt by the Dockerfile.
- CI (`.github/workflows/ci.yml`): pytest on 3.12/3.13 + ruff/pyright on push to `main` and PRs.

## Notes

After completing each working phase, generate a commit message adhering to the **Conventional Commits** specification.

* **Format:** `<type>(<scope>)[!]: <description>`
* **Primary Types:** `feat` (new features), `fix` (bug fixes), `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
* **Scope:** Optional noun specifying the codebase module (e.g., `fix(features): rolling offset bug`).
* **Breaking Changes:** Indicate with `!` before the colon (e.g., `feat(api)!: change payload schema`) or a `BREAKING CHANGE: <description>` footer.
* **Body & Footers:** Optional contextual body paragraphs and footers (e.g., `Closes #123`) MUST be separated from the header by a single blank line.