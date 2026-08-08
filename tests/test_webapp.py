"""Tests for the f1web Flask app (HTTP layer with a canned prediction)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
