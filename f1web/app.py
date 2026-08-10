"""JSON API + SPA host for the F1 result predictor (FastAPI).

Serves the JSON API consumed by the React dashboard (``f1web/ui``) and the
built SPA itself. There is no server-rendered HTML frontend; the dashboard is
the single UI. Usage::

    f1-web --port 8080              # serve http://127.0.0.1:8080/
    f1-web --host 0.0.0.0           # listen on all interfaces

Endpoints::

    /                                 built SPA (dashboard)
    /dashboard                        built SPA (alias of /)
    /health                           {"status": "ok"}

JSON API (all errors share the shape ``{"error": ...}``)::

    /api/prediction?season=&round=    prediction (live, briefly cached)
    /api/backtest                     backtest snapshot (reports/backtest.json)
    /api/calibration                  calibration snapshot (reports/calibration.json)
    /api/calendar?season=             race calendar
    /api/standings?season=&round=     driver + constructor standings
    /api/status                       artifact presence + model/dataset paths

The prediction is computed on demand through ``predict.get_prediction`` (the
same code path as the CLI) and cached in memory for a short TTL; the other
endpoints read precomputed snapshots or the cached raw API. This is a local
tool, not a high-concurrency service.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from f1core.config import load_config
from f1core.predict import get_prediction
from f1data import (
    F1APIError,
    F1Client,
    fetch_calendar,
    fetch_constructor_standings,
    fetch_driver_standings,
)

# Anchored to the repo root so reports resolve regardless of the server's
# working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_JSON = REPO_ROOT / "reports" / "backtest.json"
CALIBRATION_JSON = REPO_ROOT / "reports" / "calibration.json"
UI_DIST = REPO_ROOT / "f1web" / "ui" / "dist"
UI_DIST_INDEX = UI_DIST / "index.html"
# Vite writes JS/CSS chunks under dist/assets and the built index.html
# references them as /assets/<file> — so the /assets route resolves here.
UI_ASSETS_DIR = UI_DIST / "assets"

# In-memory cache for live predictions: (season, round) -> (timestamp, payload).
_PREDICTION_CACHE: dict[tuple[int | None, int | None], tuple[float, dict]] = {}
_PREDICTION_TTL = 300.0  # seconds; the SPA's repeated calls stay fast


def _payload(pred: dict) -> dict:
    """JSON-safe representation of a prediction dict."""
    rows = json.loads(pred["result"].to_json(orient="records"))
    return {
        "season": pred["season"],
        "round": pred["round"],
        "race": pred["meta"],
        "synthetic": pred["synthetic"],
        "verified": pred["verified"],
        "calibrated": pred["calibrated"],
        "checkpoint": pred["checkpoint"],
        "drivers": rows,
    }


def _cached_prediction(season: int | None, round_: int | None) -> dict:
    """JSON-safe prediction for (season, round), cached for the TTL."""
    key = (season, round_)
    now = time.monotonic()
    hit = _PREDICTION_CACHE.get(key)
    if hit is not None and now - hit[0] < _PREDICTION_TTL:
        return hit[1]
    payload = _payload(get_prediction(season=season, round_=round_, quiet=True))
    _PREDICTION_CACHE[key] = (now, payload)
    return payload


def _read_json(path: Path) -> dict | None:
    """Read a JSON snapshot, or None when missing/corrupt."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _data_client() -> F1Client:
    """A client for the cached raw API (offline reads for the dashboard)."""
    cfg = load_config()
    return F1Client(cache_dir=cfg["data"]["cache_dir"], user_agent=cfg["api"]["user_agent"])


def _error(message: str, status: int) -> JSONResponse:
    """Standard error payload (all API errors share the shape ``{"error": ...}``)."""
    return JSONResponse(status_code=status, content={"error": message})


def create_app() -> FastAPI:
    app = FastAPI(title="F1 Result Predictor", version="0.1.0")

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
        if not UI_DIST_INDEX.exists():
            return _error(
                "Dashboard not built - run `npm run build` in f1web/ui (see README)",
                503,
            )
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

    @app.get("/api/prediction")
    def prediction_api(
        season: int | None = Query(default=None),
        round: int | None = Query(default=None),
    ):
        try:
            return _cached_prediction(season, round)
        except SystemExit as exc:
            return _error(str(exc), 409)
        except (ValueError, F1APIError) as exc:
            return _error(str(exc), 400)

    @app.get("/api/backtest")
    def backtest_api():
        data = _read_json(BACKTEST_JSON)
        if data is None:
            return _error(f"{BACKTEST_JSON} not found - run `f1-backtest` first", 404)
        return data

    @app.get("/api/calibration")
    def calibration_api():
        data = _read_json(CALIBRATION_JSON)
        if data is None:
            return _error(f"{CALIBRATION_JSON} not found - run `f1-calibrate` first", 404)
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

    @app.get("/api/status")
    def status_api() -> dict:
        cfg = load_config()

        def exists(rel: str) -> bool:
            return (REPO_ROOT / rel).exists()

        return {
            "seasons": {
                "start": cfg["data"]["start_season"],
                "end": cfg["data"]["end_season"],
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


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    level = "debug" if args.debug else "info"
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
