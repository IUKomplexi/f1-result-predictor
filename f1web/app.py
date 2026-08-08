"""Minimal web UI for the F1 result predictor (Flask).

Usage::

    f1-web --port 8080              # serve http://127.0.0.1:8080/
    f1-web --host 0.0.0.0           # listen on all interfaces

Endpoints::

    /                                 next-race prediction page
    /prediction?season=&round=        prediction for a specific race
    /api/prediction?season=&round=    same prediction as JSON
    /backtest                         walk-forward backtest report
    /health                           {"status": "ok"}

The prediction is computed on demand through ``predict.get_prediction`` (the
same code path as the CLI). It takes a few seconds per request; this is a
local tool, not a high-concurrency service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from flask import Flask, Response, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1data import F1APIError  # noqa: E402
from predict import get_prediction  # noqa: E402

# Anchored to the repo root so the report resolves regardless of the server's
# working directory.
BACKTEST_REPORT = Path(__file__).resolve().parent.parent / "reports" / "backtest.md"


def _payload(pred: Dict) -> Dict:
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


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/")
    @app.get("/prediction")
    def prediction_page() -> Response:
        try:
            pred = _request_prediction()
        except SystemExit as exc:  # e.g. no upcoming race in the data
            return _error_page(str(exc), 409)
        except (ValueError, F1APIError) as exc:
            return _error_page(str(exc), 400)
        return render_template(
            "prediction.html", pred=pred, rows=pred["result"].to_dict("records")
        )

    @app.get("/api/prediction")
    def prediction_api() -> Response:
        try:
            return jsonify(_payload(_request_prediction()))
        except SystemExit as exc:
            return jsonify({"error": str(exc)}), 409
        except (ValueError, F1APIError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/backtest")
    def backtest() -> Response:
        if not BACKTEST_REPORT.exists():
            return Response(
                "reports/backtest.md not found - run `f1-backtest` first",
                status=404,
            )
        return render_template(
            "backtest.html", content=BACKTEST_REPORT.read_text(encoding="utf-8")
        )

    return app


def _request_prediction() -> Dict:
    """Compute the prediction for the current request's query arguments.

    Invalid (non-integer) ``season``/``round`` query values raise ValueError
    instead of silently falling back to the next race.
    """
    args = request.args
    season, round_ = args.get("season", type=int), args.get("round", type=int)
    if (args.get("season") is not None and season is None) or (
        args.get("round") is not None and round_ is None
    ):
        raise ValueError("season/round must be integers")
    return get_prediction(season=season, round_=round_, quiet=True)


def _error_page(message: str, status: int) -> Response:
    return render_template("error.html", message=message), status


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
