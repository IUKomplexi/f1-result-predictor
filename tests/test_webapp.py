"""Tests for the f1web Flask app (HTTP layer with a canned prediction)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from f1web.app import _payload, create_app


def _canned_prediction() -> dict:
    return {
        "result": pd.DataFrame(
            {
                "pred_rank": [1, 2],
                "driver_id": ["russell", "leclerc"],
                "constructor_id": ["mercedes", "ferrari"],
                "grid": [1, 2],
                "expected_points": [15.0, 12.0],
                "p_scored": [0.9, 0.8],
                "p_top3": [0.6, 0.5],
                "p_win": [0.5, 0.2],
                "actual_points": [25.0, 12.0],
                "actual_position": [1, 4],
            }
        ),
        "meta": {"race_name": "Las Vegas", "circuit_id": "vegas",
                 "date": "2024-11-23"},
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

    if Path("reports/backtest.md").exists():
        resp = client.get("/backtest")
        assert resp.status_code == 200
        assert "Walk-forward" in resp.get_data(as_text=True)

    class FakePath:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self):
            return False

        def read_text(self, *args, **kwargs):
            raise AssertionError("should not be called")

    monkeypatch.setattr(app_module, "Path", FakePath)
    assert client.get("/backtest").status_code == 404
