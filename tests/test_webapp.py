"""Tests for the f1web FastAPI app (HTTP layer with a canned prediction)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from f1data import F1APIError
from f1web.app import _payload, create_app


def _canned_prediction() -> dict:
    import numpy as np

    return {
        "result": pd.DataFrame(
            {
                "pred_rank": [np.int64(1), np.int64(2)],
                "driver_id": ["russell", "leclerc"],
                "constructor_id": ["mercedes", "ferrari"],
                "grid": [np.int64(1), np.int64(2)],
                "expected_points": [np.float64(15.0), np.float64(12.0)],
                "p_scored": [np.float64(0.9), np.float64(0.8)],
                "p_top3": [np.float64(0.6), np.float64(0.5)],
                "p_win": [np.float64(0.5), np.float64(0.2)],
                "actual_points": [np.float64(25.0), np.float64(12.0)],
                "actual_position": [np.int64(1), np.int64(4)],
            }
        ),
        "meta": {
            "race_name": "Las Vegas",
            "circuit_id": "vegas",
            "date": pd.Timestamp("2024-11-23"),
        },
        "season": 2024,
        "round": 22,
        "synthetic": False,
        "verified": True,
        "calibrated": False,
        "checkpoint": "data/model/hurdle.joblib",
    }


@pytest.fixture
def client(monkeypatch):
    import f1web.app as app_module

    app_module._PREDICTION_CACHE.clear()  # no cross-test cache pollution
    monkeypatch.setattr(app_module, "get_prediction", lambda **kw: _canned_prediction())
    return TestClient(create_app())


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_api_prediction_json(client):
    resp = client.get("/api/prediction?season=2024&round=22")
    assert resp.status_code == 200
    data = resp.json()
    assert data["season"] == 2024 and data["round"] == 22
    assert data["race"]["race_name"] == "Las Vegas"
    assert data["drivers"][0]["driver_id"] == "russell"
    assert data["drivers"][0]["expected_points"] == 15.0


def test_payload_is_json_safe():
    payload = _payload(_canned_prediction())
    assert payload["drivers"][0]["expected_points"] == 15.0
    assert isinstance(payload["drivers"][0]["p_scored"], float)


def test_error_paths(client, monkeypatch):
    import f1web.app as app_module

    def boom(**kw):
        raise ValueError("no rows for season 2024 round 99")

    monkeypatch.setattr(app_module, "get_prediction", boom)
    api = client.get("/api/prediction?season=2024&round=99")
    assert api.status_code == 400
    assert api.json()["error"].startswith("no rows for season")


def test_systemexit_maps_to_409(client, monkeypatch):
    import f1web.app as app_module

    def no_next(**kw):
        raise SystemExit("no upcoming race: seasons complete")

    monkeypatch.setattr(app_module, "get_prediction", no_next)
    api = client.get("/api/prediction").json()
    assert api["error"] == "no upcoming race: seasons complete"


def test_invalid_query_params_are_422(client):
    # FastAPI rejects non-integer season/round with 422 (validation error).
    assert client.get("/api/prediction?season=abc").status_code == 422
    assert client.get("/api/prediction?round=xyz").status_code == 422
    assert client.get("/api/calendar?season=abc").status_code == 422
    assert client.get("/api/standings?season=abc").status_code == 422
    assert client.get("/api/predictions/season").status_code == 422
    assert client.get("/api/predictions/season?season=abc").status_code == 422


def test_api_predictions_season_batch(client, monkeypatch):
    import f1web.app as app_module

    def fake_season(season, **kw):
        assert season == 2024
        second = {**_canned_prediction(), "round": 23}
        return [_canned_prediction(), second]

    monkeypatch.setattr(app_module, "predict_season", fake_season)
    resp = client.get("/api/predictions/season?season=2024")
    assert resp.status_code == 200
    data = resp.json()
    assert data["season"] == 2024
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["round"] == 22
    assert data["predictions"][0]["drivers"][0]["driver_id"] == "russell"
    assert data["predictions"][1]["round"] == 23


def test_api_predictions_season_error_is_400(client, monkeypatch):
    import f1web.app as app_module

    def boom(season, **kw):
        raise ValueError("no cached data for 2024")

    monkeypatch.setattr(app_module, "predict_season", boom)
    resp = client.get("/api/predictions/season?season=2024")
    assert resp.status_code == 400
    assert "no cached data" in resp.json()["error"]


def test_prediction_is_cached_within_ttl(client, monkeypatch):
    """Two identical /api/prediction calls hit get_prediction only once."""
    import f1web.app as app_module

    calls = {"n": 0}

    def counting(**kw):
        calls["n"] += 1
        return _canned_prediction()

    monkeypatch.setattr(app_module, "get_prediction", counting)
    client.get("/api/prediction?season=2024&round=22")
    client.get("/api/prediction?season=2024&round=22")
    assert calls["n"] == 1


def _write_json(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_api_backtest_snapshot(client, tmp_path, monkeypatch):
    import f1web.app as app_module

    snap = _write_json(tmp_path, "backtest.json", {"overall": {"model": {"mae": 1.25}}})
    monkeypatch.setattr(app_module, "BACKTEST_JSON", snap)
    resp = client.get("/api/backtest")
    assert resp.status_code == 200
    assert resp.json()["overall"]["model"]["mae"] == 1.25


def test_api_backtest_missing_is_404(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(app_module, "BACKTEST_JSON", Path("does/not/exist.json"))
    resp = client.get("/api/backtest")
    assert resp.status_code == 404
    assert "f1-backtest" not in resp.json()["error"]
    assert "run `f1 backtest` first" in resp.json()["error"]


def test_api_calibration_snapshot(client, tmp_path, monkeypatch):
    import f1web.app as app_module

    snap = _write_json(tmp_path, "calibration.json", {"deployed": ["win"]})
    monkeypatch.setattr(app_module, "CALIBRATION_JSON", snap)
    resp = client.get("/api/calibration")
    assert resp.status_code == 200
    assert resp.json()["deployed"] == ["win"]


def test_api_calibration_missing_is_404(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(app_module, "CALIBRATION_JSON", Path("does/not/exist.json"))
    resp = client.get("/api/calibration")
    assert resp.status_code == 404
    assert "f1-calibrate" not in resp.json()["error"]
    assert "run `f1 calibrate` first" in resp.json()["error"]


def test_api_calendar(client, monkeypatch):
    import f1web.app as app_module

    def fake_calendar(client_, season):
        return [{"season": season, "round": 1, "race_name": "Bahrain", "date": "2024-03-02"}]

    monkeypatch.setattr(app_module, "fetch_calendar", fake_calendar)
    resp = client.get("/api/calendar?season=2024")
    assert resp.status_code == 200
    data = resp.json()
    assert data["season"] == 2024
    assert data["calendar"][0]["race_name"] == "Bahrain"


def test_api_calendar_requires_season(client):
    assert client.get("/api/calendar").status_code == 422


def test_api_calendar_error_is_502(client, monkeypatch):
    import f1web.app as app_module

    def boom(client_, season):
        raise F1APIError("boom")

    monkeypatch.setattr(app_module, "fetch_calendar", boom)
    resp = client.get("/api/calendar?season=2024")
    assert resp.status_code == 502
    assert "boom" in resp.json()["error"]


def test_api_standings(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(
        app_module,
        "fetch_driver_standings",
        lambda client_, season, round_: [{"driver_id": "verstappen"}],
    )
    monkeypatch.setattr(
        app_module,
        "fetch_constructor_standings",
        lambda client_, season, round_: [{"constructor_id": "red_bull"}],
    )
    resp = client.get("/api/standings?season=2024&round=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["season"] == 2024 and data["round"] == 10
    assert data["driver"][0]["driver_id"] == "verstappen"
    assert data["constructor"][0]["constructor_id"] == "red_bull"


def test_api_standings_requires_season(client):
    assert client.get("/api/standings").status_code == 422


def test_api_status(client):
    data = client.get("/api/status").json()
    assert set(data) == {"seasons", "model", "data", "reports", "dashboard"}
    assert set(data["reports"]) == {"has_backtest", "has_calibration"}
    assert set(data["model"]) == {"checkpoint", "calibrators", "has_checkpoint", "has_calibrators"}


def test_api_status_exposes_season_hints(client):
    """/api/status carries the picker clamp (modern era floor + data ceiling)."""
    seasons = client.get("/api/status").json()["seasons"]
    assert seasons["data_start"] == 2014
    assert seasons["data_end"] >= seasons["data_start"]
    assert seasons["start"] <= seasons["end"]


def test_dashboard_not_built_is_503(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(app_module, "UI_DIST_INDEX", Path("does/not/exist.html"))
    resp = client.get("/dashboard")
    assert resp.status_code == 503
    assert "npm run build" in resp.text


def test_dashboard_serves_built_spa(client, tmp_path, monkeypatch):
    import f1web.app as app_module

    index = tmp_path / "index.html"
    index.write_text("<html><body>dashboard</body></html>", encoding="utf-8")
    monkeypatch.setattr(app_module, "UI_DIST_INDEX", index)
    monkeypatch.setattr(app_module, "UI_DIST", tmp_path)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "dashboard" in resp.text
    # Root `/` serves the same dashboard (single frontend).
    resp = client.get("/")
    assert resp.status_code == 200
    assert "dashboard" in resp.text


def test_spa_assets_served(client, tmp_path, monkeypatch):
    import f1web.app as app_module

    assets = tmp_path / "assets"
    assets.mkdir()
    asset = assets / "app.js"
    asset.write_text("console.log('x')", encoding="utf-8")
    monkeypatch.setattr(app_module, "UI_DIST", tmp_path)
    monkeypatch.setattr(app_module, "UI_ASSETS_DIR", assets)
    resp = client.get("/assets/app.js")
    assert resp.status_code == 200
    assert "console.log" in resp.text
