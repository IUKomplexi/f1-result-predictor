# Weather features — evaluation report (NOT adopted)

**Date:** 2026-08-08 · **Phase 12**

## Summary

Race-day weather was implemented as an optional feature set and evaluated with
the project's gate: adopt only if the walk-forward backtest improves ≥1
primary metric (winner_hit / top3_overlap / spearman) with no regression in
the others. **The gate failed — weather is not adopted.** The data layer and
feature plumbing remain in the repository for future re-evaluation, but the
shipped model ignores weather.

## Design

- Source: Open-Meteo (free, no API key). ERA5 archive for past races, live
  forecast for the upcoming one.
- Coverage: `scripts/fetch_weather.py` cached daily weather for **329/329**
  race weekends (2010–2025), **100% of starts** — temperature max/min,
  precipitation sum, wind max, humidity mean, cloud cover, plus a derived
  `wet` flag (precip > 1 mm).
- Leakage discipline: features are race-level and strictly pre-race (exact
  (season, round) join). Training uses historical actuals; an upcoming race
  would use the forecast (forecasts are never cached — they change).
- Forecast caveat: Open-Meteo returns nulls beyond its ~7-day operational
  horizon, so a next-race forecast more than a week out feeds NaN features
  until the race approaches.

## Evaluation (walk-forward backtest, 2013–2025, same pipeline)

| metric | without weather (shipped) | with weather | delta |
| --- | --- | --- | --- |
| winner_hit | 0.5387 | 0.5424 | **+0.0037** |
| top3_overlap | 0.6667 | 0.6630 | **−0.0037** |
| spearman | 0.6541 | 0.6520 | **−0.0021** |
| mae (secondary) | 2.9333 | 2.9214 | −0.0119 |

The "without weather" run reproduces the shipped Phase-11 numbers exactly,
so the comparison is apples-to-apples (the weather columns are all-NaN and
dropped as constant by the model, which behaves identically without them).

## Verdict

1 of 3 primary metrics improves (winner_hit +0.004) but 2 regress
(top3_overlap −0.004, spearman −0.002). All deltas are within the noise of
the metric (±0.004 over 200+ races) — weather is a race-level variable that
can only help through learned interactions (e.g. "in rain, grid matters
less"), and the aggregate metrics show no such signal. Per the approved
gate: **not adopted**, no model change.

## What remains available

- `f1weather/` — cached Open-Meteo client + race-level fetch helpers.
- `scripts/fetch_weather.py` — fetch actuals (all seasons) or a forecast.
- `WEATHER_FEATURES`, `merge_weather`, `build_dataset(weather=...)` — join
  weather onto a dataset for experiments.
- `predict._apply_target_weather` (with tests) — forecast/archive wiring for
  the next-race flow, ready to re-enable.

To re-evaluate later (e.g. with finer granularity, per-hour conditions, or
after a model change): `build_dataset(..., weather=<frame>)` and re-run the
with/without walk-forward comparison above.
