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
.venv/Scripts/python -m pip install -e .            # Windows
.venv/Scripts/python scripts/fetch_all.py            # fetch + cache 2010-2025 (one-time)
.venv/Scripts/python -m pytest -q                    # run the test suite
```

Requires Python ≥ 3.11. The raw API responses are cached under `data/raw/`
(one-time ~2 min fetch; everything after that runs offline).

## Usage

| Command | What it does |
| --- | --- |
| `python scripts/fetch_all.py [--start 2010] [--end 2025]` | fetch and cache raw API data |
| `python model/train.py` | train the final model → `data/model/hurdle.joblib` |
| `python model/calibrate.py` | fit isotonic probability calibrators → `data/model/calibrators.joblib` |
| `python model/evaluate.py` | walk-forward backtest vs baselines → `reports/backtest.md` |
| `python predict.py` | predict the **next race** → `reports/prediction.md` |
| `python predict.py --season 2024 --round 22` | predict any race; past races are verified vs actuals |
| `python predict.py --grid qual.csv` | supply a qualifying grid (`driver_id,grid`) for an upcoming race |

Examples:

```bash
python predict.py                                   # Dutch GP 2026 (next race)
python predict.py --season 2024 --round 22          # dry run: Las Vegas 2024 + verification
python predict.py --grid qual.csv                   # with known grid for the next race
```

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
model/evaluate.py           walk-forward backtest vs grid/championship/zero
predict.py                  next-race prediction + markdown report
```

Leakage safety: every rolling/cumulative feature uses `shift(1)` so it only
ever sees races strictly before the target race (unit-tested).

## Probability calibration

The gradient-boosted classifiers' raw probabilities are overconfident (a
common trait of gradient boosting). `model/calibrate.py` collects genuinely
out-of-sample raw scores from the walk-forward backtest and fits isotonic
calibrators for P(top-10) / P(top-3) / P(win). A calibrator is **deployed only
where it improves Brier on a chronological hold-out** (fit on OOS seasons
2013–2020, evaluated on 2021–2025): top-3 (0.0767 → 0.0756) and win
(0.0330 → 0.0315) are calibrated; P(top-10) stays raw (calibration slightly
hurt it: 0.1606 → 0.1639). Run it after every `model/train.py`; `predict.py`
applies the saved calibrators automatically.

## Results (honest)

Walk-forward backtest, train on all seasons strictly before the test season,
evaluate 2013–2025 (mean per race):

| baseline | winner_hit | top3_overlap | spearman | MAE (pts) |
| --- | --- | --- | --- | --- |
| **model** | 0.498 | 0.620 | **0.637** | 3.01 |
| grid order | **0.535** | **0.686** | 0.623 | **2.83** |
| championship | 0.450 | 0.613 | 0.617 | 3.99 |
| zero | 0.535 | 0.686 | 0.623 | 4.99 |

The model beats the championship and zero baselines broadly and beats
grid-order on ranking correlation (Spearman), but **does not beat "start in
grid order"** on winner/top-3/MAE — grid position is an extremely strong F1
predictor. That gap is the honest state of the model, not a bug.

Dry run (Las Vegas 2024, predicted from pre-race info only): winner hit ✓,
top-3 overlap 0.67, Spearman 0.80, MAE 1.40 points.

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
