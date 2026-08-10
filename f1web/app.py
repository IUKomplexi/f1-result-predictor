"""Web UI + JSON API for the F1 result predictor (Flask).

Serves both the classic server-rendered pages and the JSON API consumed by
the React dashboard (``f1web/ui``). Usage::

    f1-web --port 8080              # serve http://127.0.0.1:8080/
    f1-web --host 0.0.0.0           # listen on all interfaces

Endpoints::

    /                                 next-race prediction page
    /prediction?season=&round=        prediction for a specific race
    /dashboard                        React dashboard (built SPA from ui/dist)
    /backtest                         walk-forward backtest report
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

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from f1data import (
    F1APIError,
    F1Client,
    fetch_calendar,
    fetch_constructor_standings,
    fetch_driver_standings,
)
from predict import get_prediction

# Anchored to the repo root so reports resolve regardless of the server's
# working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_REPORT = REPO_ROOT / "reports" / "backtest.md"
BACKTEST_JSON = REPO_ROOT / "reports" / "backtest.json"
CALIBRATION_JSON = REPO_ROOT / "reports" / "calibration.json"
UI_DIST = REPO_ROOT / "f1web" / "ui" / "dist"
UI_DIST_INDEX = UI_DIST / "index.html"

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


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/")
    @app.get("/prediction")
    def prediction_page() -> Response:
        try:
            season, round_ = _prediction_args()
            pred = get_prediction(season=season, round_=round_, quiet=True)
        except SystemExit as exc:  # e.g. no upcoming race in the data
            return _error_page(str(exc), 409)
        except (ValueError, F1APIError) as exc:
            return _error_page(str(exc), 400)
        return Response(
            render_template(
                "prediction.html", pred=pred, rows=pred["result"].to_dict("records")
            )
        )

    @app.get("/dashboard")
    def dashboard() -> Response:
        if not UI_DIST_INDEX.exists():
            return Response(
                "Dashboard not built - run `npm run build` in f1web/ui (see README)",
                status=503,
                content_type="text/plain",
            )
        return Response(UI_DIST_INDEX.read_text(encoding="utf-8"), content_type="text/html")

    @app.get("/assets/<path:filename>")
    def spa_assets(filename: str) -> Response:
        return send_from_directory(UI_DIST, filename)

    @app.get("/api/prediction")
    def prediction_api() -> Response:
        try:
            season, round_ = _prediction_args()
            return jsonify(_cached_prediction(season, round_))
        except SystemExit as exc:
            return _error_json(str(exc), 409)
        except (ValueError, F1APIError) as exc:
            return _error_json(str(exc), 400)

    @app.get("/api/backtest")
    def backtest_api() -> Response:
        data = _read_json(BACKTEST_JSON)
        if data is None:
            return _error_json(
                f"{BACKTEST_JSON} not found - run `f1-backtest` first", 404
            )
        return jsonify(data)

    @app.get("/api/calibration")
    def calibration_api() -> Response:
        data = _read_json(CALIBRATION_JSON)
        if data is None:
            return _error_json(
                f"{CALIBRATION_JSON} not found - run `f1-calibrate` first", 404
            )
        return jsonify(data)

    @app.get("/api/calendar")
    def calendar_api() -> Response:
        season = request.args.get("season", type=int)
        if season is None:
            return _error_json("season is required", 400)
        try:
            calendar = fetch_calendar(_data_client(), season)
        except (F1APIError, KeyError, TypeError) as exc:
            return _error_json(f"could not fetch calendar for {season}: {exc}", 502)
        return jsonify({"season": season, "calendar": calendar})

    @app.get("/api/standings")
    def standings_api() -> Response:
        season = request.args.get("season", type=int)
        if season is None:
            return _error_json("season is required", 400)
        round_ = request.args.get("round", type=int)
        client = _data_client()
        try:
            driver = fetch_driver_standings(client, season, round_)
            constructor = fetch_constructor_standings(client, season, round_)
        except (F1APIError, KeyError, TypeError) as exc:
            return _error_json(f"could not fetch standings for {season}: {exc}", 502)
        return jsonify(
            {"season": season, "round": round_, "driver": driver, "constructor": constructor}
        )

    @app.get("/api/status")
    def status_api() -> Response:
        cfg = load_config()

        def exists(rel: str) -> bool:
            return (REPO_ROOT / rel).exists()

        return jsonify(
            {
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
        )

    @app.get("/backtest")
    def backtest() -> Response:
        if not BACKTEST_REPORT.exists():
            return Response(
                "reports/backtest.md not found - run `f1-backtest` first",
                status=404,
            )
        return Response(
            render_template("backtest.html", content=BACKTEST_REPORT.read_text(encoding="utf-8"))
        )

    return app


def _prediction_args() -> tuple[int | None, int | None]:
    """Parse season/round query args; non-integer values raise ValueError."""
    args = request.args
    season, round_ = args.get("season", type=int), args.get("round", type=int)
    if (args.get("season") is not None and season is None) or (
        args.get("round") is not None and round_ is None
    ):
        raise ValueError("season/round must be integers")
    return season, round_


def _error_page(message: str, status: int) -> Response:
    return Response(render_template("error.html", message=message), status=status)


def _error_json(message: str, status: int) -> Response:
    resp = jsonify({"error": message})
    resp.status_code = status
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
