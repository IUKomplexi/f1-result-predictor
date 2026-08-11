# F1 Result Predictor

Predicts **points per driver** for a Formula 1 race from *pre-race* information
(grid, qualifying, driver/team form, circuit history, championship position),
using a zero-inflated hurdle model trained on 2010–2025 race data from the
[Jolpica F1 API](https://www.jolpi.ca/ergast/) (the Ergast successor).

```
E[points] = P(top-10) × E(points | top-10)
```

with companion classifiers for P(top-3) and P(win). The output is the full
grid ranked by expected points.

> 📖 **New here?** Start with [`OVERVIEW.md`](OVERVIEW.md) — repository map,
> architecture diagrams, and a guide to extending this codebase.

## Setup

```bash
uv sync --all-extras     # install project + test/lint/web deps into .venv (Python 3.12)
uv run scripts/fetch_all.py      # fetch + cache 2010-2025 (one-time)
uv run pytest -q                 # run the test suite
```

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/) (``uv sync`` pins
the environment to ``uv.lock``). The raw API responses are cached under
`data/raw/` (one-time ~2 min fetch; everything after that runs offline). The
install also registers console scripts (`f1-predict`, `f1-train`, `f1-backtest`,
`f1-calibrate`, `f1-search`, `f1-web`) — activate the venv
(`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere) so
they are on your PATH.

## Usage

Every command has two equivalent forms: `uv run python <module>.py ...` from the
repo root, or the installed console script. Config/report paths are relative to
the working directory, so run from the repo root (or pass absolute `--out`/
`--dataset` paths).

| Command (repo root / console script) | What it does |
| --- | --- |
| `uv run scripts/fetch_all.py [--start 2010] [--end 2025]` | fetch and cache raw API data |
| `python model/train.py` · `f1-train` | train the final model → `data/model/hurdle.joblib` |
| `python model/calibrate.py` · `f1-calibrate` | fit isotonic probability calibrators → `data/model/calibrators.joblib` |
| `python model/evaluate.py [--no-quantize]` · `f1-backtest` | walk-forward backtest vs baselines → `reports/backtest.md` (quantized by default) |
| `f1-predict` | predict the **next race** → `reports/prediction.md` |
| `f1-predict --season 2024 --round 22` | predict any race; past races are verified vs actuals |
| `f1-predict --grid qual.csv` | supply a qualifying grid (`driver_id,grid`) for an upcoming race |
| `f1-web [--host 127.0.0.1] [--port 8080]` | local web API + dashboard (needs the `web` extra) |
| `docker compose up --build` | build + start the dashboard in a container |

Examples:

```bash
f1-predict                                       # Dutch GP 2026 (next race)
f1-predict --season 2024 --round 22              # dry run: Las Vegas 2024 + verification
f1-predict --grid qual.csv                       # with known grid for the next race
python model/search.py --n 16 --max-test-season 2019   # re-tune hyperparameters
```

## Web UI

Quickest path — no venv or Node toolchain needed, fully offline:

```bash
docker compose up --build      # build the image, start on http://127.0.0.1:8080/
```

The image is self-contained: it builds the React dashboard (`f1web/ui`),
installs the app, and bakes in the cached raw data, the dataset and the model
checkpoints (`data/`), so everything runs offline. Generated snapshots
(`reports/backtest.json`, `reports/calibration.json`) live in a named `reports`
volume and survive rebuilds; the baked-in data is refreshed only by rebuilding
the image (e.g. after `scripts/fetch_all.py` or `f1-train`):

```bash
docker compose exec web f1-backtest          # refresh /api/backtest
docker compose exec web f1-calibrate         # refresh /api/calibration
```

Without Docker (requires the `web` extra):

```bash
uv sync --extra web          # or uv sync --all-extras
f1-web --port 8080            # open http://127.0.0.1:8080/
```

Endpoints: `/` and `/dashboard` (React dashboard), `/health`, and the JSON API
under `/api/*` (GET `/api/prediction`, GET `/api/predictions/season`,
`/api/backtest`, `/api/calibration`, `/api/calendar`, `/api/standings`,
`/api/status`, GET/PUT `/api/config`, `POST /api/jobs`, `GET /api/jobs`,
`GET /api/jobs/{id}`, `POST /api/predict`). The dashboard is the single
frontend — there is no server-rendered HTML (the FastAPI backend only serves
JSON + the built SPA). Predictions are computed on demand through the same
code path as the CLI and cached on disk under `data/predictions/` (gitignored),
so repeat calls and whole-season fetches are instant. It is a local tool, not a
service.

### Full pipeline control from the dashboard

The dashboard is the control surface for the whole pipeline. The **Settings**
tab edits every `config.toml` value (including the HGB hyperparameters under
`[model.params]` and the feature selection) and writes them back in place; the
**Race** tab shows a single race's prediction with a season selector and
prev/next navigation; **Race History** loads a whole season in one request;
each pipeline step runs from its own tab — **Data** (fetch), **Train**,
**Search**, **Backtest** and **Calibration** — as async background jobs (one at
a time, the rest queued) with live logs and results rendered inline, and can
apply a search's best config. The **Specific Race** tab's prediction panel (and
`POST /api/predict`) accepts *ephemeral* overrides (season/round, a grid CSV,
feature toggles) that are merged over the config in memory only — nothing is
written to `config.toml`.

> ⚠️ **Docker config persistence:** in the container, `config.toml` lives inside
> the image and resets on rebuild. To keep dashboard edits across rebuilds,
> bind-mount it, e.g. `docker compose run -v "$PWD/config.toml:/app/config.toml"` —
> the CLI and web always read/write the same file, so either surface works.
> Jobs are tied to the server process lifetime: `uvicorn --reload` restarts
> kill in-flight jobs (the `reports/jobs/*.json` history records what ran).

### Accessing the dashboard

Open the dashboard in a normal browser at a **concrete** address:

- **This machine:** `http://127.0.0.1:8080/` (or `http://localhost:8080/`)

> ⚠️ **Do not use `http://0.0.0.0:8080/`.** `0.0.0.0` is the "bind to all

## Configuration

`config.toml` (optional; built-in defaults match it): API base URL /
User-Agent, cache paths, season range, model checkpoint, report paths, the
feature selection (`[features] enabled`), and the HGB hyperparameters
(`[model.params]`, read by train before the code defaults). CLI flags override
config values. The file is the single source of truth: the dashboard writes it
back in place (`PUT /api/config`) so the CLI and web always read the same
settings; per-race prediction overrides are ephemeral (in-memory only).

## Architecture

```
scripts/fetch_all.py   ->  data/raw/*.json            (cached API responses)
f1data/                     polite cached client + normalized fetchers
f1weather/                  weather data layer (evaluated, not adopted)
f1core/                     shared core: predict, config, reporting, httpclient
features/build.py           per-start dataset: strictly pre-race features,
                            points target, leakage-tested
features/registry.py        declarative feature registry (id, category,
                            default, builder, rationale) + selection helpers
model/train.py              hurdle model (HGB classifier + regressor)
model/search.py             walk-forward-validated hyperparameter search
model/calibrate.py          isotonic probability calibration (per-target)
model/evaluate.py           walk-forward backtest vs grid/championship/zero
f1web/app.py                FastAPI JSON API + built-SPA host
```

Leakage safety: every rolling/cumulative feature uses `shift(1)` so it only
ever sees races strictly before the target race (unit-tested).

## Feature registry & selection

All 31 features (27 numeric + 4 categorical) are registered in
`features/registry.py` — the single source of truth for id, category, default,
builder, and rationale — and classified by a walk-forward permutation-importance
audit (`scripts/feature_audit.py`, documented in `reports/features.md`):

| category | meaning | default |
| --- | --- | --- |
| `core` | high impact (survived FDR q=0.05 in ≥1 hurdle component) | on |
| `selectable` | low impact; kept for experiments | off |
| `cut` | removal improved the backtest ≥1 SE (ablation gate) | off |

The default enabled set is the 14 core features (`config.toml` `[features]
enabled`). Every feature is still computed; only the enabled subset enters the
training matrix. Toggle per run with `--enable-features`/`--disable-features`
on `f1-train`, `f1-backtest`, `f1-predict`, `f1-calibrate`, `f1-search`;
toggling changes the model-checkpoint fingerprint, so stale checkpoints are
rejected instead of silently reused.

## Probability calibration

The gradient-boosted classifiers' raw probabilities can be overconfident (a
common trait of gradient boosting). `model/calibrate.py` collects genuinely
out-of-sample raw scores from the walk-forward backtest and fits isotonic
calibrators for P(top-10) / P(top-3) / P(win). A calibrator is **deployed only
where it improves Brier on a chronological hold-out** (fit on OOS seasons
2013–2020, evaluated on 2021–2025). With the current tuned model the raw
probabilities are already well-calibrated, so no calibrator is deployed
(calibration would hurt all three targets); the mechanism stays in place and
re-activates automatically if a future model is miscalibrated again. Run it
after every `model/train.py`; `predict.py` applies the saved calibrators
automatically.

## Results (honest)

Walk-forward backtest, train on all seasons strictly before the test season,
evaluate 2013–2025 (mean per race). Model = hurdle on the 14 **core**
registry features (see [Feature registry & selection](#feature-registry--selection)),
tuned hyperparameters (`model/search.py`), and expected points **quantized to
the points table** (adopted because it improved walk-forward
MAE/top-3/Spearman):

| baseline | winner_hit | top3_overlap | spearman | MAE (pts) |
| --- | --- | --- | --- | --- |
| **model** | **0.550** | 0.659 | **0.656** | 2.92 |
| grid order | 0.535 | **0.686** | 0.623 | **2.83** |
| championship | 0.450 | 0.613 | 0.617 | 3.99 |
| zero | 0.535 | 0.686 | 0.623 | 4.99 |

The model **beats the grid-order baseline on winner hit-rate** (0.550 vs
0.535) and on ranking correlation (Spearman 0.656 vs 0.623). Grid order still
edges it on top-3 (0.659 vs 0.686) and MAE (2.92 vs 2.83 — the grid baseline
predicts the exact points-table value whenever the pole sitter wins, which no
probabilistic model can match). The feature audit cut 17 low-impact/redundant
features (kept off by default): winner_hit rose from 0.539 to 0.550 with no
metric moving beyond the fold-to-fold noise floor (`reports/features.md`).

Dry run (Las Vegas 2024, predicted from pre-race info only): winner hit ✓,
top-3 overlap 0.67, Spearman 0.79, MAE 1.95 points.

## Limitations

- Win/podium probabilities are isotonic-calibrated where that improved
  hold-out Brier (top-3 and win); P(top-10) is raw. The **ranking** remains
  the primary output.
- Without qualifying results (`--grid`) the prediction for an upcoming race is
  weaker — grid is the single strongest feature.
- Drivers with no history (rookies, new teams) get missing-feature handling
  (gradient boosting supports missing values natively).
- Sprint points are features, not part of the main-race points target.

## API terms

The Jolpica API asks clients to send a descriptive `User-Agent` (configured in
`config.toml`) and to cache responses — both are handled by `F1Client`
(polite rate limiting, retry/backoff, on-disk cache).

## Weather (evaluated, not adopted)

Race-day weather (Open-Meteo: ERA5 actuals for history, live forecast for
the next race) was implemented and evaluated as a feature set — **it is not
adopted**: the walk-forward gate failed (1 of 3 primary metrics improved,
2 regressed, all within noise). The data layer (`f1weather/`,
`scripts/fetch_weather.py`, `data/weather/`) and feature plumbing
(`WEATHER_FEATURES`, `build_dataset(weather=...)`, `merge_weather`) remain
available for future re-evaluation. See `reports/weather.md` for the full
comparison and how to re-run it.

## Development
- Tests: `uv run pytest -q` — fully offline (recorded fixtures, no network). The
  suite is enforced deprecation-free (`filterwarnings = ["error::DeprecationWarning"]`
  in `pyproject.toml`).
- `tests/test_e2e.py` runs the **full pipeline end-to-end** (fetch →
  assemble → features → train → predict → report) offline against a synthetic
  API session, so CI exercises the whole chain, not just pieces.
- Lint & types: `uv run ruff check .` and `uv run pyright` (installed via the
  `lint` extra) are both at **0 errors**. `pyrightconfig.json` + `.vscode/settings.json`
  pin basic mode and exclude `build/`/`tests/`/`data/`; pandas is untyped, so
  the few remaining `Unknown`-union sites carry scoped, justified
  `# type: ignore[reportX]` comments. Pylance in the editor uses the same
  settings as the CLI.
- CI: `.github/workflows/ci.yml` runs the test matrix on Python 3.12/3.13 plus
  a lint job (ruff + pyright) on every push to `main` and every PR.
- Reproducibility: `uv sync --frozen` (installs the locked deps from `uv.lock`;
  `--all-extras` to include test/lint/web tooling).
- License: MIT (see `LICENSE`).
