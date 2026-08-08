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

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"     # Windows (project + test deps)
.venv/Scripts/python scripts/fetch_all.py            # fetch + cache 2010-2025 (one-time)
.venv/Scripts/python -m pytest -q                    # run the test suite
```

Requires Python ≥ 3.11. The raw API responses are cached under `data/raw/`
(one-time ~2 min fetch; everything after that runs offline). The install also
registers console scripts (`f1-predict`, `f1-train`, `f1-backtest`,
`f1-calibrate`, `f1-search`) — activate the venv (`.venv\Scripts\activate` on
Windows, `source .venv/bin/activate` elsewhere) so they are on your PATH.

## Usage

Every command has two equivalent forms: `python <module>.py ...` from the repo
root, or the installed console script. Config/report paths are relative to the
working directory, so run from the repo root (or pass absolute `--out`/
`--dataset` paths).

| Command (repo root / console script) | What it does |
| --- | --- |
| `python scripts/fetch_all.py [--start 2010] [--end 2025]` | fetch and cache raw API data |
| `python model/train.py` · `f1-train` | train the final model → `data/model/hurdle.joblib` |
| `python model/calibrate.py` · `f1-calibrate` | fit isotonic probability calibrators → `data/model/calibrators.joblib` |
| `python model/evaluate.py [--no-quantize]` · `f1-backtest` | walk-forward backtest vs baselines → `reports/backtest.md` (quantized by default) |
| `python predict.py` · `f1-predict` | predict the **next race** → `reports/prediction.md` |
| `python predict.py --season 2024 --round 22` | predict any race; past races are verified vs actuals |
| `python predict.py --grid qual.csv` | supply a qualifying grid (`driver_id,grid`) for an upcoming race |
| `f1-web [--host 127.0.0.1] [--port 8080]` | local web UI (needs the `web` extra) |

Examples:

```bash
f1-predict                                       # Dutch GP 2026 (next race)
f1-predict --season 2024 --round 22              # dry run: Las Vegas 2024 + verification
f1-predict --grid qual.csv                       # with known grid for the next race
python model/search.py --n 16 --max-test-season 2019   # re-tune hyperparameters
```

## Web UI

```bash
pip install -e ".[web]"       # Flask is an optional extra
f1-web --port 8080            # open http://127.0.0.1:8080/
```

Endpoints: `/` (next-race prediction page), `/prediction?season=&round=` (any
race), `/api/prediction` (JSON), `/backtest` (report), `/health`. Predictions
are computed on demand through the same code path as the CLI (a few seconds
per request — it is a local tool, not a service).

## Configuration

`config.toml` (optional; built-in defaults match it): API base URL /
User-Agent, cache paths, season range, model checkpoint, report paths. CLI
flags override config values.

## Architecture

```
scripts/fetch_all.py   ->  data/raw/*.json            (cached API responses)
f1data/                     polite cached client + normalized fetchers
features/build.py           per-start dataset: strictly pre-race features,
                            points target, leakage-tested
model/train.py              hurdle model (HGB classifier + regressor)
model/search.py             walk-forward-validated hyperparameter search
model/calibrate.py          isotonic probability calibration (per-target)
model/evaluate.py           walk-forward backtest vs grid/championship/zero
predict.py                  next-race prediction + markdown report
```

Leakage safety: every rolling/cumulative feature uses `shift(1)` so it only
ever sees races strictly before the target race (unit-tested).

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
evaluate 2013–2025 (mean per race). Model = hurdle with 9 extra pre-race
features (teammate-relative gaps, sprint, pairing tenure, ...), tuned
hyperparameters (`model/search.py`), and expected points **quantized to the
points table** (adopted because it improved walk-forward MAE/top-3/Spearman):

| baseline | winner_hit | top3_overlap | spearman | MAE (pts) |
| --- | --- | --- | --- | --- |
| **model** | **0.539** | 0.667 | **0.654** | 2.93 |
| grid order | 0.535 | **0.686** | 0.623 | **2.83** |
| championship | 0.450 | 0.613 | 0.617 | 3.99 |
| zero | 0.535 | 0.686 | 0.623 | 4.99 |

The model now **beats the grid-order baseline on winner hit-rate** (0.539 vs
0.535) and on ranking correlation (Spearman 0.654 vs 0.623). Grid order still
edges it on top-3 (0.667 vs 0.686) and MAE (2.93 vs 2.83 — the grid baseline
predicts the exact points-table value whenever the pole sitter wins, which no
probabilistic model can match). Every metric improved over the Phase-6 model
(0.531 / 0.662 / 0.651 / 2.97).

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

## Development

- Tests: `pytest -q` — fully offline (recorded fixtures, no network).
- CI: `.github/workflows/ci.yml` runs the suite on Python 3.11/3.12 on every
  push/PR.
- Reproducibility: `pip install -r requirements.txt` (project + test deps).
