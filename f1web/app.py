"""JSON API + SPA host for the F1 result predictor (FastAPI).

Serves the JSON API consumed by the Preact dashboard (``f1web/ui``) and the
built SPA itself. There is no server-rendered HTML frontend; the dashboard is
the single UI. Usage::

    f1 web --port 8080              # serve http://127.0.0.1:8080/
    f1 web --host 0.0.0.0           # listen on all interfaces

Endpoints::

    /                                 built SPA (dashboard)
    /dashboard                        built SPA (alias of /)
    /health                           {"status": "ok"}

JSON API (all errors share the shape ``{"error": ...}``)::

    GET  /api/prediction?season=&round=    prediction (live, briefly cached)
    POST /api/predict                      prediction with ephemeral overrides
    GET  /api/backtest                     backtest snapshot (reports/backtest.json)
    GET  /api/calibration                  calibration snapshot (reports/calibration.json)
    GET  /api/calendar?season=             race calendar
    GET  /api/standings?season=&round=     driver + constructor standings
    GET  /api/status                       artifact presence + model/dataset paths
    GET  /api/models                       saved model checkpoints + metadata
    GET  /api/config                       effective config + schema metadata
    PUT  /api/config                       validate + write config.toml (single source of truth)
    POST /api/jobs                         queue a pipeline job
                                            (fetch/train/calibrate/backtest/history)
                                            payload keys mirror the CLI options
                                            (see f1web/jobs.JOB_PAYLOAD_KEYS)
    GET  /api/jobs                         job history
    GET  /api/jobs/{id}                    job status + log + result

The prediction is computed on demand through ``predict.get_prediction`` (the
same code path as the CLI) and cached in memory for a short TTL; the other
endpoints read precomputed snapshots or the cached raw API. Pipeline jobs run
asynchronously in-process (``f1web/jobs.py``) and are tied to the server's
lifetime. This is a local tool, not a high-concurrency service.
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from f1core.config import (
    DATA_START_FLOOR,
    MODEL_PARAM_KEYS,
    SCHEMA,
    SEASON_MAX,
    SEASON_MIN,
    load_config,
    save_config,
    validate_config,
)
from f1core.predict import (
    format_report,
    get_prediction,
    predict_season,
    prediction_payload,
)
from f1data import (
    F1APIError,
    F1Client,
    fetch_calendar,
    fetch_constructor_standings,
    fetch_driver_standings,
)
from f1web.jobs import JOB_PAYLOAD_KEYS, JOB_TYPES, JobManager, register_default_handlers
from features.registry import REGISTRY, all_feature_ids, category_meta, default_enabled

# Anchored to the repo root so reports resolve regardless of the server's
# working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_TOML = REPO_ROOT / "config.toml"
BACKTEST_JSON = REPO_ROOT / "reports" / "backtest.json"
CALIBRATION_JSON = REPO_ROOT / "reports" / "calibration.json"
# Disk-backed prediction cache (gitignored; keyed by season/round/feature-fingerprint/params).
PREDICTION_CACHE_DIR = REPO_ROOT / "data" / "predictions"
UI_DIST = REPO_ROOT / "f1web" / "ui" / "dist"
UI_DIST_INDEX = UI_DIST / "index.html"
# Vite writes JS/CSS chunks under dist/assets and the built index.html
# references them as /assets/<file> — so the /assets route resolves here.
UI_ASSETS_DIR = UI_DIST / "assets"

# In-memory cache for live predictions: (season, round) -> (timestamp, payload).
_PREDICTION_CACHE: dict[tuple[int | None, int | None], tuple[float, dict]] = {}
_PREDICTION_TTL = 300.0  # seconds; the SPA's repeated calls stay fast


def _payload(pred: dict) -> dict:
    """JSON-safe representation of a prediction dict (see predict.prediction_payload)."""
    return prediction_payload(pred)


def _cached_prediction(season: int | None, round_: int | None) -> dict:
    """JSON-safe prediction for (season, round), cached in memory + on disk."""
    key = (season, round_)
    now = time.monotonic()
    hit = _PREDICTION_CACHE.get(key)
    if hit is not None and now - hit[0] < _PREDICTION_TTL:
        return hit[1]
    # The disk-backed cache in get_prediction makes repeat calls instant (and
    # skips dataset assembly on a hit); the in-memory TTL avoids even the JSON
    # read within a short window.
    payload = _payload(get_prediction(
        season=season, round_=round_, quiet=True, cache_dir=PREDICTION_CACHE_DIR,
    ))
    _PREDICTION_CACHE[key] = (now, payload)
    return payload


def _read_json(path: Path) -> dict | None:
    """Read a JSON snapshot, or None when missing/corrupt."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _data_end_season(cfg: dict) -> int:
    """Latest season with cached raw data (fallback: the configured end).

    Scans the raw cache dir for ``.../f1/<season>...`` filenames. The result
    caps the pipeline season pickers (train/backtest/search/calibration) so a
    run never silently references seasons that have no data yet — fetching new
    seasons is the Data page's job, which uses the *configured* end instead.
    """
    cache_dir = REPO_ROOT / cfg["data"]["cache_dir"]
    seasons: set[int] = set()
    if cache_dir.is_dir():
        for entry in cache_dir.glob("*.json"):
            match = re.search(r"f1_(\d{4})", entry.name)
            if match:
                seasons.add(int(match.group(1)))
    return max(seasons) if seasons else int(cfg["data"]["end_season"])


def _data_client() -> F1Client:
    """A client for the cached raw API (offline reads for the dashboard)."""
    cfg = load_config(CONFIG_TOML)
    return F1Client(cache_dir=cfg["data"]["cache_dir"], user_agent=cfg["api"]["user_agent"])


def _error(message: str, status: int) -> JSONResponse:
    """Standard error payload (all API errors share the shape ``{"error": ...}``)."""
    return JSONResponse(status_code=status, content={"error": message})


def _grid_path_from_text(text: str) -> str:
    """Validate ``driver_id,grid`` CSV text and return a path to a temp file.

    Raises :class:`ValueError` for a malformed grid (bad columns / non-integer grid).
    """
    import numpy as np
    import pandas as pd

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(text)
        path = fh.name
    try:
        table = pd.read_csv(path)
        for col in ("driver_id", "grid"):
            if col not in table.columns:
                raise ValueError("grid CSV must have columns 'driver_id' and 'grid'")
        grid = np.asarray(pd.to_numeric(table["grid"], errors="raise"), dtype=float)
        if not np.all(grid == np.floor(grid)):
            raise ValueError("grid CSV 'grid' column must be integers")
    except Exception as exc:
        Path(path).unlink(missing_ok=True)
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"could not parse grid CSV: {exc}") from exc
    return path


def _slim_job(job: dict, now: float | None = None) -> dict:
    """A job without the (possibly large) log and result, for the history list.

    ``elapsed_s`` is the running duration (queued jobs have none) or the total
    duration once finished; ``log_lines`` lets the UI offer a live log tail
    without shipping the full log in every list poll.
    """
    now = now if now is not None else time.time()
    start = job.get("started_at")
    elapsed_s = round((job.get("finished_at") or now) - start, 2) if start else None
    return {k: job[k] for k in ("id", "type", "label", "status", "error",
                                "created_at", "started_at", "finished_at")} | {
        "elapsed_s": elapsed_s,
        "log_lines": len(job["log"]),
    }


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="F1 Result Predictor", version="0.1.0")
    manager = job_manager or register_default_handlers(JobManager())
    app.state.jobs = manager

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Keep every API error on the documented {"error": ...} shape, including
        # FastAPI's 422 validation responses for bad query params.
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "query")
        return _error(f"invalid query parameter {loc}: {first.get('msg', 'invalid')}", 422)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        if not UI_DIST_INDEX.exists():
            return _error(
                "Dashboard not built - run `npm run build` in f1web/ui (see README)",
                503,
            )
        return FileResponse(UI_DIST_INDEX)

    @app.get("/assets/{file_path:path}", include_in_schema=False)
    def spa_assets(file_path: str):
        # Assets are static chunks served independently of whether index.html
        # has been written yet, so this route must NOT gate on UI_DIST_INDEX
        # existing. The result_path guard below still prevents escaping the
        # assets dir (and non-existent files 404), so there is no traversal hole.
        target = (UI_ASSETS_DIR / file_path).resolve()
        if not target.is_relative_to(UI_ASSETS_DIR.resolve()) or not target.is_file():
            return _error("not found", 404)
        return FileResponse(target)

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        icon = UI_DIST / "favicon.svg"
        if not icon.is_file():
            return _error("not found", 404)
        return FileResponse(icon)

    # ------------------------------------------------------------------ config

    @app.get("/api/config")
    def config_api() -> dict:
        cfg = load_config(CONFIG_TOML)
        return {
            "config": cfg,
            "schema": SCHEMA,
            "features": {
                "registry": all_feature_ids(),
                "defaults": default_enabled(),
                "categories": {f.id: f.category for f in REGISTRY},
                "category_meta": category_meta(),
            },
            "seasons": {
                "min": SEASON_MIN,
                "max": SEASON_MAX,
                "data_start": DATA_START_FLOOR,
                "data_end": _data_end_season(cfg),
            },
            "model_params_keys": sorted(MODEL_PARAM_KEYS),
            "jobs": sorted(JOB_TYPES),
        }

    @app.put("/api/config")
    async def put_config_api(request: Request):
        try:
            body = await request.json()
        except (ValueError, TypeError):
            return _error("request body must be valid JSON", 400)
        if not isinstance(body, dict):
            return _error("request body must be a JSON object", 400)
        # Merge over DEFAULTS so a partial update can't drop sections, then
        # validate the whole effective config before writing.
        merged = copy.deepcopy(load_config(CONFIG_TOML))
        for section, values in body.items():
            if isinstance(values, dict):
                merged[section] = {**(merged.get(section) or {}), **values}
            else:
                merged[section] = values
        errors = validate_config(merged)
        if errors:
            return _error("; ".join(errors), 422)
        try:
            save_config(merged, CONFIG_TOML)
        except (OSError, ValueError) as exc:
            return _error(f"could not write config: {exc}", 500)
        return {"config": load_config(CONFIG_TOML)}

    # ------------------------------------------------------------------- jobs

    @app.post("/api/jobs")
    async def jobs_api(request: Request):
        try:
            body = await request.json()
        except (ValueError, TypeError):
            return _error("request body must be valid JSON", 400)
        if not isinstance(body, dict) or not isinstance(body.get("type"), str):
            return _error("request body must have a 'type' string", 400)
        job_type = body["type"]
        if job_type not in JOB_TYPES:
            known = ", ".join(sorted(JOB_TYPES))
            return _error(f"unknown job type: {job_type} (known: {known})", 404)
        payload = body.get("payload") or {}
        if not isinstance(payload, dict):
            return _error("'payload' must be a JSON object", 400)
        # Only accept the keys a job type understands; ignore the rest.
        keys = JOB_PAYLOAD_KEYS.get(job_type, ())
        payload = {k: payload[k] for k in keys if k in payload}
        # Validate feature toggles up front so a typo fails fast instead of
        # surfacing mid-run (mirrors enabled_features' unknown-feature check).
        known_features = set(all_feature_ids())
        for name in ("enable_features", "disable_features"):
            group = payload.get(name)
            if group is None:
                continue
            if not isinstance(group, list) or not all(isinstance(f, str) for f in group):
                return _error(f"'{name}' must be a list of strings", 400)
            unknown = sorted(set(group) - known_features)
            if unknown:
                return _error(f"{name}: unknown feature(s): {unknown}", 422)
        if "name" in payload and not isinstance(payload["name"], str):
            return _error("'name' must be a string", 400)
        if "model_path" in payload and payload["model_path"] is not None \
                and not isinstance(payload["model_path"], str):
            return _error("'model_path' must be a string", 400)
        if "model_paths" in payload and payload["model_paths"] is not None:
            if not isinstance(payload["model_paths"], list) or not all(
                isinstance(p, str) for p in payload["model_paths"]
            ):
                return _error("'model_paths' must be a list of strings", 400)
        try:
            job_id = manager.submit(job_type, payload)
        except KeyError as exc:
            return _error(str(exc), 404)
        return JSONResponse(status_code=202, content={"id": job_id, "type": job_type})

    @app.get("/api/jobs")
    def jobs_list() -> dict:
        return {"jobs": [_slim_job(j) for j in manager.list()]}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = manager.get(job_id)
        if job is None:
            return _error(f"job not found: {job_id}", 404)
        return job

    # -------------------------------------------------------------- prediction

    @app.get("/api/prediction")
    def prediction_api(
        season: int | None = Query(default=None),
        round: int | None = Query(default=None),
        refresh: bool = Query(default=False),
    ):
        try:
            if refresh:
                # CLI --refresh: bypass the in-memory + raw-data caches.
                payload = _payload(get_prediction(
                    season=season, round_=round, quiet=True,
                    cache_dir=PREDICTION_CACHE_DIR, refresh=True,
                ))
                return payload
            return _cached_prediction(season, round)
        except SystemExit as exc:
            return _error(str(exc), 409)
        except (ValueError, F1APIError) as exc:
            return _error(str(exc), 400)

    @app.get("/api/predictions/season")
    def predictions_season_api(season: int):
        """All completed rounds of a season in one dataset pass (Race History)."""
        try:
            preds = predict_season(season, quiet=True, cache_dir=PREDICTION_CACHE_DIR)
        except SystemExit as exc:
            return _error(str(exc), 409)
        except (ValueError, F1APIError) as exc:
            return _error(str(exc), 400)
        return {"season": season, "predictions": [_payload(p) for p in preds]}

    @app.post("/api/predict")
    async def predict_api(request: Request):
        """Prediction with ephemeral overrides merged over config in memory only.

        Body: ``{"season": int?, "round": int?, "grid_csv": str?,
        "enable_features": [str]?, "disable_features": [str]?,
        "refresh": bool?, "model_path": str?, "write_report": bool?}``.
        ``refresh`` ignores the raw-data cache (CLI ``--refresh``), ``model_path``
        overrides the checkpoint (CLI ``--model``), and ``write_report`` writes
        the same Markdown report the CLI produces to
        ``config.toml [report] prediction`` (CLI ``--out``). Nothing is written
        to ``config.toml``; the overrides apply to this request only.
        """
        try:
            body = await request.json()
        except (ValueError, TypeError):
            return _error("request body must be valid JSON", 400)
        if not isinstance(body, dict):
            return _error("request body must be a JSON object", 400)

        season = body.get("season")
        round_ = body.get("round")
        if season is not None and not isinstance(season, int):
            return _error("'season' must be an integer", 400)
        if round_ is not None and not isinstance(round_, int):
            return _error("'round' must be an integer", 400)

        # Validate feature toggles before creating the grid temp file, so no
        # temp file is left behind on a validation error.
        enable = body.get("enable_features") or []
        disable = body.get("disable_features") or []
        for name, group in (("enable_features", enable), ("disable_features", disable)):
            if not isinstance(group, list) or not all(isinstance(f, str) for f in group):
                return _error(f"'{name}' must be a list of strings", 400)

        refresh = body.get("refresh", False)
        model_path = body.get("model_path")
        write_report = body.get("write_report", False)
        if not isinstance(refresh, bool):
            return _error("'refresh' must be a boolean", 400)
        if model_path is not None and not isinstance(model_path, str):
            return _error("'model_path' must be a string", 400)
        if not isinstance(write_report, bool):
            return _error("'write_report' must be a boolean", 400)

        grid_path = None
        grid_csv = body.get("grid_csv")
        if grid_csv is not None:
            if not isinstance(grid_csv, str):
                return _error("'grid_csv' must be a string", 400)
            try:
                grid_path = _grid_path_from_text(grid_csv)
            except ValueError as exc:
                return _error(str(exc), 400)

        try:
            pred = get_prediction(
                season=season, round_=round_, grid_csv=grid_path,
                enable_features=enable, disable_features=disable,
                refresh=refresh, model_path=model_path, quiet=True,
            )
        except SystemExit as exc:
            return _error(str(exc), 409)
        except (ValueError, F1APIError) as exc:
            return _error(str(exc), 400)
        finally:
            if grid_path:
                Path(grid_path).unlink(missing_ok=True)

        if write_report:
            # Mirror `f1 predict`'s --out: write the same Markdown report the
            # CLI produces, to the path configured in [report] prediction.
            try:
                report = format_report(
                    pred["result"], pred["season"], pred["round"], pred["meta"],
                    verified=pred["verified"], checkpoint=pred["checkpoint"],
                    calibrated=pred["calibrated"],
                )
                out = REPO_ROOT / load_config(CONFIG_TOML)["report"]["prediction"]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(report, encoding="utf-8")
            except (OSError, ValueError) as exc:
                return _error(f"could not write report: {exc}", 500)
        return _payload(pred)

    @app.get("/api/backtest")
    def backtest_api():
        data = _read_json(BACKTEST_JSON)
        if data is None:
            return _error(f"{BACKTEST_JSON} not found - run `f1 backtest` first", 404)
        return data

    @app.get("/api/calibration")
    def calibration_api():
        data = _read_json(CALIBRATION_JSON)
        if data is None:
            return _error(f"{CALIBRATION_JSON} not found - run `f1 calibrate` first", 404)
        return data

    @app.get("/api/calendar")
    def calendar_api(season: int):
        try:
            calendar = fetch_calendar(_data_client(), season)
        except (F1APIError, KeyError, TypeError) as exc:
            return _error(f"could not fetch calendar for {season}: {exc}", 502)
        return {"season": season, "calendar": calendar}

    @app.get("/api/standings")
    def standings_api(
        season: int,
        round: int | None = Query(default=None),
    ):
        client = _data_client()
        try:
            driver = fetch_driver_standings(client, season, round)
            constructor = fetch_constructor_standings(client, season, round)
        except (F1APIError, KeyError, TypeError) as exc:
            return _error(f"could not fetch standings for {season}: {exc}", 502)
        return {"season": season, "round": round, "driver": driver, "constructor": constructor}

    @app.get("/api/models")
    def models_api() -> dict:
        """Saved model checkpoints + metadata (from data/model/index.json).

        The index is written by every train job (see model.train.update_model_index);
        if it is missing/empty, fall back to listing ``*.joblib`` files in the
        model dir so the selector still works on images built before the index
        existed.
        """
        index_path = REPO_ROOT / "data" / "model" / "index.json"
        models: dict = {}
        if index_path.is_file():
            try:
                models = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                models = {}
        if not models:
            calibrators = Path(load_config(CONFIG_TOML)["model"]["calibrators"]).name
            model_dir = REPO_ROOT / "data" / "model"
            for joblib in sorted(model_dir.glob("*.joblib")):
                if joblib.name == calibrators:
                    continue
                models[joblib.stem] = {"checkpoint": str(joblib.relative_to(REPO_ROOT))}
        return {"models": models, "default": load_config(CONFIG_TOML)["model"]["checkpoint"]}

    @app.get("/api/status")
    def status_api() -> dict:
        cfg = load_config(CONFIG_TOML)

        def exists(rel: str) -> bool:
            return (REPO_ROOT / rel).exists()

        return {
            "seasons": {
                "start": cfg["data"]["start_season"],
                "end": cfg["data"]["end_season"],
                "data_start": DATA_START_FLOOR,
                "data_end": _data_end_season(cfg),
            },
            "model": {
                "checkpoint": cfg["model"]["checkpoint"],
                "calibrators": cfg["model"]["calibrators"],
                "has_checkpoint": exists(cfg["model"]["checkpoint"]),
                "has_calibrators": exists(cfg["model"]["calibrators"]),
            },
            "data": {
                "dataset": cfg["data"]["dataset"],
                "has_dataset": exists(cfg["data"]["dataset"]),
                "has_raw_cache": (REPO_ROOT / cfg["data"]["cache_dir"]).is_dir(),
            },
            "reports": {
                "has_backtest": BACKTEST_JSON.exists(),
                "has_calibration": CALIBRATION_JSON.exists(),
            },
            "dashboard": {"built": UI_DIST_INDEX.exists()},
        }

    return app


