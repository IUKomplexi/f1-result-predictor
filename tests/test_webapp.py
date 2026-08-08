"""Tests for the f1web Flask app (HTTP layer with a canned prediction)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
        "meta": {"race_name": "Las Vegas", "circuit_id": "vegas",
                 "date": pd.Timestamp("2024-11-23")},
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
    return create_app().test_client()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_index_renders_prediction(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Las Vegas" in html
    assert "russell" in html and "mercedes" in html
    assert "verified vs actuals" in html


def test_prediction_page_with_query(client):
    resp = client.get("/prediction?season=2024&round=22")
    assert resp.status_code == 200
    assert "Round 22" in resp.get_data(as_text=True)


def test_api_prediction_json(client):
    resp = client.get("/api/prediction?season=2024&round=22")
    assert resp.status_code == 200
    data = resp.get_json()
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
    page = client.get("/prediction?season=2024&round=99")
    assert page.status_code == 400
    assert "no rows for season" in page.get_data(as_text=True)
    api = client.get("/api/prediction?season=2024&round=99")
    assert api.status_code == 400
    assert api.get_json()["error"].startswith("no rows for season")


def test_backtest_route(client, monkeypatch):
    import f1web.app as app_module

    if app_module.BACKTEST_REPORT.exists():
        resp = client.get("/backtest")
        assert resp.status_code == 200
        assert "Walk-forward" in resp.get_data(as_text=True)

    # 404 when the report file is missing.
    monkeypatch.setattr(app_module, "BACKTEST_REPORT", Path("does/not/exist.md"))
    assert client.get("/backtest").status_code == 404


def test_systemexit_maps_to_409(client, monkeypatch):
    import f1web.app as app_module

    def no_next(**kw):
        raise SystemExit("no upcoming race: seasons complete")

    monkeypatch.setattr(app_module, "get_prediction", no_next)
    page = client.get("/")
    assert page.status_code == 409
    assert "no upcoming race" in page.get_data(as_text=True)
    api = client.get("/api/prediction").get_json()
    assert api["error"] == "no upcoming race: seasons complete"


def test_invalid_query_params_are_400(client):
    assert client.get("/prediction?season=abc").status_code == 400
    assert client.get("/api/prediction?round=xyz").status_code == 400


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
    assert resp.get_json()["overall"]["model"]["mae"] == 1.25


def test_api_backtest_missing_is_404(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(app_module, "BACKTEST_JSON", Path("does/not/exist.json"))
    resp = client.get("/api/backtest")
    assert resp.status_code == 404
    assert "f1-backtest" in resp.get_json()["error"]


def test_api_calibration_snapshot(client, tmp_path, monkeypatch):
    import f1web.app as app_module

    snap = _write_json(tmp_path, "calibration.json", {"deployed": ["win"]})
    monkeypatch.setattr(app_module, "CALIBRATION_JSON", snap)
    resp = client.get("/api/calibration")
    assert resp.status_code == 200
    assert resp.get_json()["deployed"] == ["win"]


def test_api_calibration_missing_is_404(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(app_module, "CALIBRATION_JSON", Path("does/not/exist.json"))
    resp = client.get("/api/calibration")
    assert resp.status_code == 404
    assert "f1-calibrate" in resp.get_json()["error"]


def test_api_calendar(client, monkeypatch):
    import f1web.app as app_module

    def fake_calendar(client_, season):
        return [{"season": season, "round": 1, "race_name": "Bahrain", "date": "2024-03-02"}]

    monkeypatch.setattr(app_module, "fetch_calendar", fake_calendar)
    resp = client.get("/api/calendar?season=2024")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["season"] == 2024
    assert data["calendar"][0]["race_name"] == "Bahrain"


def test_api_calendar_requires_season(client):
    assert client.get("/api/calendar").status_code == 400


def test_api_calendar_error_is_502(client, monkeypatch):
    import f1web.app as app_module

    def boom(client_, season):
        raise F1APIError("boom")

    monkeypatch.setattr(app_module, "fetch_calendar", boom)
    resp = client.get("/api/calendar?season=2024")
    assert resp.status_code == 502
    assert "boom" in resp.get_json()["error"]


def test_api_standings(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(
        app_module, "fetch_driver_standings",
        lambda client_, season, round_: [{"driver_id": "verstappen"}],
    )
    monkeypatch.setattr(
        app_module, "fetch_constructor_standings",
        lambda client_, season, round_: [{"constructor_id": "red_bull"}],
    )
    resp = client.get("/api/standings?season=2024&round=10")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["season"] == 2024 and data["round"] == 10
    assert data["driver"][0]["driver_id"] == "verstappen"
    assert data["constructor"][0]["constructor_id"] == "red_bull"


def test_api_standings_requires_season(client):
    assert client.get("/api/standings").status_code == 400


def test_api_status(client):
    data = client.get("/api/status").get_json()
    assert set(data) == {"seasons", "model", "data", "reports", "dashboard"}
    assert set(data["reports"]) == {"has_backtest", "has_calibration"}
    assert set(data["model"]) == {"checkpoint", "has_checkpoint", "has_calibrators"}


def test_dashboard_not_built_is_503(client, monkeypatch):
    import f1web.app as app_module

    monkeypatch.setattr(app_module, "UI_DIST_INDEX", Path("does/not/exist.html"))
    resp = client.get("/dashboard")
    assert resp.status_code == 503
    assert "npm run build" in resp.get_data(as_text=True)


def test_dashboard_serves_built_spa(client, tmp_path, monkeypatch):
    import f1web.app as app_module

    index = tmp_path / "index.html"
    index.write_text("<html><body>dashboard</body></html>", encoding="utf-8")
    monkeypatch.setattr(app_module, "UI_DIST_INDEX", index)
    monkeypatch.setattr(app_module, "UI_DIST", tmp_path)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "dashboard" in resp.get_data(as_text=True)


def test_spa_assets_served(client, tmp_path, monkeypatch):
    import f1web.app as app_module

    asset = tmp_path / "app.js"
    asset.write_text("console.log('x')", encoding="utf-8")
    monkeypatch.setattr(app_module, "UI_DIST", tmp_path)
    resp = client.get("/assets/app.js")
    assert resp.status_code == 200
    assert "console.log" in resp.get_data(as_text=True)
