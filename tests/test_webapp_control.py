"""Tests for the dashboard-control endpoints: config read/write, async jobs,
and prediction with ephemeral overrides."""

from __future__ import annotations

import time

import pandas as pd
from fastapi.testclient import TestClient

from f1core.config import load_config
from f1web.app import create_app
from f1web.jobs import JobManager


def _config_client(tmp_path, monkeypatch):
    """A TestClient whose config.toml is a temp file (never the repo's)."""
    import f1web.app as app_module

    cfg_path = tmp_path / "config.toml"
    # Seed the temp config with the current repo values so writes are realistic.
    from f1core.config import config_to_toml

    cfg_path.write_text(config_to_toml(load_config()), encoding="utf-8")
    monkeypatch.setattr(app_module, "CONFIG_TOML", cfg_path)
    return TestClient(create_app())


# ------------------------------------------------------------------ config


def test_get_config_returns_schema_and_features(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    data = client.get("/api/config").json()
    assert "config" in data and "schema" in data
    assert "registry" in data["features"] and "defaults" in data["features"]
    assert data["seasons"]["min"] <= data["seasons"]["max"]
    assert any(d["section"] == "model" and d["key"] == "params" for d in data["schema"])


def test_put_config_writes_back_and_roundtrips(tmp_path, monkeypatch):
    import f1web.app as app_module

    client = _config_client(tmp_path, monkeypatch)
    data = client.get("/api/config").json()["config"]
    data["model"]["params"]["learning_rate"] = 0.09
    resp = client.put("/api/config", json=data)
    assert resp.status_code == 200
    assert resp.json()["config"]["model"]["params"]["learning_rate"] == 0.09
    # The file (not just the response) now holds the change.
    written = load_config(app_module.CONFIG_TOML)
    assert written["model"]["params"]["learning_rate"] == 0.09


def test_put_config_rejects_invalid(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    data = client.get("/api/config").json()["config"]
    data["features"]["enabled"] = ["bogus_feature"]
    resp = client.put("/api/config", json=data)
    assert resp.status_code == 422
    assert "unknown feature" in resp.json()["error"]


def test_put_config_bad_season_range(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    data = client.get("/api/config").json()["config"]
    data["data"]["start_season"] = 2030
    data["data"]["end_season"] = 2020
    resp = client.put("/api/config", json=data)
    assert resp.status_code == 422
    assert "start_season" in resp.json()["error"]


def test_put_config_partial_keeps_other_sections(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    # Send only a partial update; other sections must survive (merged over defaults).
    resp = client.put("/api/config", json={"data": {"end_season": 2026}})
    assert resp.status_code == 200
    cfg = resp.json()["config"]
    assert cfg["data"]["end_season"] == 2026
    assert cfg["api"]["base_url"].startswith("http")


# --------------------------------------------------------------------- jobs


def _wait_for(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def _jobs_client(tmp_path, monkeypatch):
    """A client whose job history is isolated to tmp and handlers are fakes."""
    import f1web.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR", tmp_path / "reports" / "jobs")
    manager = JobManager()

    def fake_handler(payload, log):
        log("started")
        log(f"payload={payload}")
        return {"echo": payload, "n": 1}

    manager.register("fetch", fake_handler)
    return TestClient(create_app(job_manager=manager))


def test_post_job_runs_and_records_log_result(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    resp = client.post("/api/jobs", json={"type": "fetch", "payload": {"refresh": True}})
    assert resp.status_code == 202
    job_id = resp.json()["id"]
    job = _wait_for(client, job_id)
    assert job["status"] == "done"
    assert "started" in job["log"]
    assert job["result"]["echo"]["refresh"] is True


def test_job_history_and_single_job_detail(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    ids = []
    for _ in range(2):
        ids.append(client.post("/api/jobs", json={"type": "fetch"}).json()["id"])
    for job_id in ids:
        _wait_for(client, job_id)
    history = client.get("/api/jobs").json()["jobs"]
    assert len(history) == 2
    # history entries are slim (no log/result), detail has them.
    assert all("result" not in j and "log" not in j for j in history)
    detail = client.get(f"/api/jobs/{ids[0]}").json()
    assert "result" in detail and "log" in detail


def test_job_failure_is_recorded(tmp_path, monkeypatch):
    import f1web.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR", tmp_path / "reports" / "jobs")
    manager = JobManager()

    def boom(payload, log):
        raise RuntimeError("dataset too small")

    manager.register("train", boom)
    client = TestClient(create_app(job_manager=manager))
    job_id = client.post("/api/jobs", json={"type": "train"}).json()["id"]
    job = _wait_for(client, job_id)
    assert job["status"] == "failed"
    assert "dataset too small" in job["error"]


def test_unknown_job_type_is_404(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    resp = client.post("/api/jobs", json={"type": "nope"})
    assert resp.status_code == 404
    assert "unknown job type" in resp.json()["error"]


def test_unknown_job_id_is_404(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_worker_survives_handler_systemexit(tmp_path, monkeypatch):
    """A handler raising SystemExit is recorded as a failure, not a crash.

    Regression: an uncaught SystemExit/KeyboardInterrupt would kill the worker
    thread and leave every later job stuck 'queued' forever.
    """
    import f1web.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR", tmp_path / "reports" / "jobs")
    manager = jobs_module.JobManager()

    def boom(payload, log):
        raise SystemExit("no upcoming race")

    def ok(payload, log):
        return {"n": 1}

    manager.register("train", boom)
    manager.register("backtest", ok)
    client = TestClient(create_app(job_manager=manager))

    failed = client.post("/api/jobs", json={"type": "train"}).json()["id"]
    assert _wait_for(client, failed)["status"] == "failed"

    # The worker must still process subsequent jobs.
    good = client.post("/api/jobs", json={"type": "backtest"}).json()["id"]
    assert _wait_for(client, good)["status"] == "done"


def test_predict_rejects_non_integer_grid(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/predict",
        json={"grid_csv": "driver_id,grid\nrussell,1.5"},
    )
    assert resp.status_code == 400
    assert "integer" in resp.json()["error"]


# ------------------------------------------------------------------ predict


def test_predict_override_passes_args_and_leaves_config(tmp_path, monkeypatch):
    import f1web.app as app_module

    client = _config_client(tmp_path, monkeypatch)
    seen = {}

    def fake_prediction(**kw):
        seen.update(kw)
        import numpy as np

        return {
            "result": pd.DataFrame(
                {
                    "pred_rank": [np.int64(1)],
                    "driver_id": ["russell"],
                    "constructor_id": ["mercedes"],
                    "grid": [np.int64(1)],
                    "expected_points": [np.float64(15.0)],
                    "p_scored": [np.float64(0.9)],
                    "p_top3": [np.float64(0.6)],
                    "p_win": [np.float64(0.5)],
                    "actual_points": [np.float64(25.0)],
                    "actual_position": [np.int64(1)],
                }
            ),
            "meta": {"race_name": "Test", "circuit_id": "test", "date": "2024-01-01"},
            "season": 2024,
            "round": 1,
            "synthetic": True,
            "verified": False,
            "calibrated": False,
            "checkpoint": "data/model/hurdle.joblib",
        }

    monkeypatch.setattr(app_module, "get_prediction", fake_prediction)
    before = app_module.CONFIG_TOML.read_bytes()

    resp = client.post(
        "/api/predict",
        json={"season": 2024, "round": 1, "enable_features": ["grid"]},
    )
    assert resp.status_code == 200
    assert resp.json()["season"] == 2024
    # The override reached get_prediction.
    assert seen["season"] == 2024 and seen["round_"] == 1
    assert seen["enable_features"] == ["grid"]
    # The config file was not touched by the override.
    assert app_module.CONFIG_TOML.read_bytes() == before


def test_predict_invalid_grid_csv_is_400(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    resp = client.post("/api/predict", json={"grid_csv": "driver_id\nrussell"})
    assert resp.status_code == 400
    assert "columns" in resp.json()["error"]


def test_predict_bad_json_is_400(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/predict",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
