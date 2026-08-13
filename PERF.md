# Performance ledger

Measured, kept, and reverted optimization attempts. Read this before proposing
an experiment so a discarded idea stays discarded.

## Baseline (2026-08-13, commit 096c24f)

Environment: Windows 11 local workspace, `.venv` Python 3.14, `data/raw` with
218 cached files. Docker Desktop with BuildKit; image `f1-result-predictor:latest`.

| Item | Baseline |
|---|---|
| `load_config` (TOML parse, per request) | 0.27 ms mean |
| `_data_end_season` (glob over ~218 raw files) | 0.99 ms mean |
| `GET /api/config` | 3.4 ms median (first-call ~350 ms startup skew) |
| `GET /api/status` | 3.6 ms median |
| `GET /health` | 2.1 ms |
| `GET /api/prediction` (warm, disk cache hit) | 2.6 ms median |
| `GET /api/predictions/season?season=2025` (batch) | 38 ms |
| `GET /api/models` | 2.9 ms |
| checkpoint `joblib.load` (`data/model/hurdle.joblib`) | 1727 ms |
| Docker image size | 2.34 GB |
| Docker full rebuild time | 52.8 s cold (see attempts below) |

Note: the local checkpoint's stored feature fingerprint (957fea4e41e1) does not
match the current code defaults (b6d532cb3a40) — a pre-existing stale-checkpoint
state, unrelated to performance work. Warm predictions (disk cache) are
unaffected.

## Attempts

| Idea | Baseline → Result | Verdict | Why |
|---|---|---|---|
| GZip middleware (≥1KB) on the FastAPI app | no compression → `content-encoding: gzip` on every >1KB API response + SPA chunk | kept | API JSON and hashed JS/CSS are the bulk of dashboard transfer; small responses (<1KB) pass through untouched |
| `Cache-Control` on static routes | none → `/assets/*` immutable 1y (Vite-hashed), `/favicon.svg` 1d, `/` no-cache | kept | Browser re-downloads stop for immutable chunks; index.html still revalidates so new asset hashes propagate |
| Memoize `_data_end_season` (raw-cache glob) | 0.99 ms → 0.08 ms mean | kept | 13× faster; keyed on cache-dir string + dir mtime — recomputes exactly when files are added/removed (verified) |
| Memoize `load_config` (TOML re-parse) | 0.27 ms → 0.09 ms mean | kept | 3× faster; keyed on path string + file mtime, `save_config` clears it explicitly; deepcopy on return keeps callers' mutations safe (verified) |
| Endpoint latency (config/status) | `/api/config` 3.4 ms, `/api/status` 3.6 ms → 2.4 / 2.5 ms median | kept | ~30% faster per dashboard poll; the rest is TestClient/server overhead |
| In-process checkpoint cache (`_joblib_load`) | 132 ms cold → 0.3 ms warm | kept | Every grid/model override on the Race tab pays the decode once per process instead of per request; keyed on path+mtime, retrain invalidates |
| Frontend (bundle splitting / memoization) | — | not attempted | Bundle already split + small (59 KB JS / 20 KB gzip); no measured bottleneck justified changes |
| Prediction cache key hardening | already correct | no change | `_model_cache_id` (path+mtime) and `_grid_hash` (content hash) already key the disk cache; dedicated tests pass |
| Docker: cache mounts + `f1web/*.py`-before-sync + drop `scripts/` + logging limits | 2.34 GB → 1.01 GB (57% smaller); rebuild 52.8 s cold | kept | UI-source edits no longer bust the uv layer (UI-edit rebuild 5.2 s); uv/npm caches persist via BuildKit mounts, never in the image |
| Docker: narrow `chown -R` to `data/` + `reports/` | 576 MB chown layer → 11.7 MB | kept | `.venv` and sources are root-owned read-only at runtime; only `data/`/`reports/` are written (dataset builds, prediction cache, job snapshots) — a `chown -R /app` duplicated the whole virtualenv per layer |
