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
    # slim entries carry the queue-widget fields: elapsed and log-line count.
    for entry in history:
        assert entry["status"] == "done"
        assert isinstance(entry["elapsed_s"], (int, float)) and entry["elapsed_s"] >= 0
        assert entry["log_lines"] == 2  # fake handler logs "started" + payload
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


# ------------------------------------------- jobs: CLI option parity


def test_job_payload_forwards_all_cli_options(tmp_path, monkeypatch):
    """Every result-affecting CLI option reaches the job handler unchanged.

    The dashboard jobs mirror the options ``f1 train/calibrate/backtest``
    expose (see f1web/jobs.JOB_PAYLOAD_KEYS).
    """
    import f1web.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "JOBS_DIR", tmp_path / "reports" / "jobs")
    manager = JobManager()

    def fake_handler(payload, log):
        log(f"payload={payload}")
        return {"echo": payload}

    for job_type in ("train", "calibrate", "backtest", "history"):
        manager.register(job_type, fake_handler)
    client = TestClient(create_app(job_manager=manager))

    payloads = {
        "history": {"start": 2024, "end": 2026, "refresh": True},
        "train": {
            "start": 2012, "end": 2024, "refresh": True, "name": "my-model",
            "enable_features": ["grid"], "disable_features": ["season"],
        },
        "calibrate": {
            "start": 2012, "end": 2024, "refresh": True,
            "fit_through_season": 2021, "eval_from_season": 2022,
            "enable_features": ["grid"], "disable_features": ["season"],
            "model_path": "data/model/other.joblib",
        },
        "backtest": {
            "start": 2012, "end": 2024, "refresh": True, "quantize": False,
            "use_checkpoint": True, "model_path": "data/model/other.joblib",
            "model_paths": ["data/model/other.joblib", "data/model/exp.joblib"],
            "enable_features": ["grid"], "disable_features": ["season"],
        },
    }
    for job_type, payload in payloads.items():
        job_id = client.post(
            "/api/jobs", json={"type": job_type, "payload": payload}
        ).json()["id"]
        job = _wait_for(client, job_id)
        assert job["status"] == "done", job
        assert job["result"]["echo"] == payload, f"{job_type}: payload not forwarded"


def test_job_rejects_unknown_feature_id(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/jobs",
        json={"type": "train", "payload": {"enable_features": ["bogus_feature"]}},
    )
    assert resp.status_code == 422
    assert "unknown feature" in resp.json()["error"]


# ------------------------------------------- jobs: train runs calibration


def test_train_handler_calibrates_named_checkpoint(tmp_path, monkeypatch):
    """Train folds calibration in: a named model is calibrated on its own
    checkpoint path; the shared wrappers get the same range and features."""
    import f1web.jobs as jobs_module
    import model.calibrate as calibrate_module
    import model.train as train_module
    from f1core.config import config_to_toml, load_config

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(config_to_toml(load_config()), encoding="utf-8")
    monkeypatch.setattr(jobs_module, "REPO_ROOT", tmp_path)
    calls = {}

    def fake_train(**kw):
        calls["train"] = kw
        return {"checkpoint": kw["out"], "rows": 42}

    def fake_calibrate(**kw):
        calls["calibrate"] = kw
        return {"calibrators": kw["out"], "deployed": ["scored"]}

    monkeypatch.setattr(train_module, "run", fake_train)
    monkeypatch.setattr(calibrate_module, "run", fake_calibrate)

    result = jobs_module._train_handler({"name": "my-model"}, lambda line: None)

    assert calls["train"]["out"] == "data/model/my-model.joblib"
    # The named checkpoint is calibrated on its own path.
    assert calls["calibrate"]["model_path"] == "data/model/my-model.joblib"
    assert calls["calibrate"]["start"] == calls["train"]["start"]
    assert calls["calibrate"]["end"] == calls["train"]["end"]
    assert result["calibrated"] is True
    # The handler surfaces whatever calibrator path the shared wrapper reports.
    assert result["calibrators"] == calls["calibrate"]["out"]


def test_train_handler_default_model_uses_shared_calibrators(tmp_path, monkeypatch):
    """Without a name, calibration targets the shared [model] calibrators path
    (walk-forward mode, no model_path)."""
    import f1web.jobs as jobs_module
    import model.calibrate as calibrate_module
    import model.train as train_module
    from f1core.config import config_to_toml, load_config

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(config_to_toml(load_config()), encoding="utf-8")
    monkeypatch.setattr(jobs_module, "REPO_ROOT", tmp_path)
    calls = {}

    def fake_train(**kw):
        calls["train"] = kw
        return {"checkpoint": kw["out"], "rows": 42}

    def fake_calibrate(**kw):
        calls["calibrate"] = kw
        return {"calibrators": kw["out"]}

    monkeypatch.setattr(train_module, "run", fake_train)
    monkeypatch.setattr(calibrate_module, "run", fake_calibrate)

    cfg = load_config(cfg_path)
    result = jobs_module._train_handler({}, lambda line: None)

    assert "model_path" not in calls["calibrate"]
    assert calls["calibrate"]["out"] == cfg["model"]["calibrators"]
    assert result["calibrated"] is True


def test_train_handler_calibration_failure_keeps_model(tmp_path, monkeypatch):
    """A calibration ValueError (e.g. no out-of-sample seasons left) must not
    fail the train job — the model is still trained, predictions just stay
    on raw probabilities."""
    import f1web.jobs as jobs_module
    import model.calibrate as calibrate_module
    import model.train as train_module
    from f1core.config import config_to_toml, load_config

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(config_to_toml(load_config()), encoding="utf-8")
    monkeypatch.setattr(jobs_module, "REPO_ROOT", tmp_path)

    def fake_train(**kw):
        return {"checkpoint": kw["out"], "rows": 42}

    def boom_calibrate(**kw):
        raise ValueError("no out-of-sample seasons remain in 2010-2026")

    monkeypatch.setattr(train_module, "run", fake_train)
    monkeypatch.setattr(calibrate_module, "run", boom_calibrate)

    result = jobs_module._train_handler({}, lambda line: None)

    assert result["calibrated"] is False
    assert "no out-of-sample" in result["calibrate_error"]
    assert result["checkpoint"]  # the train result is preserved


def test_job_feature_toggles_must_be_list_of_strings(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/jobs",
        json={"type": "backtest", "payload": {"disable_features": "grid"}},
    )
    assert resp.status_code == 400
    assert "list of strings" in resp.json()["error"]


def test_job_rejects_non_string_model_path(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    resp = client.post(
        "/api/jobs",
        json={"type": "backtest", "payload": {"model_path": 123}},
    )
    assert resp.status_code == 400
    assert "model_path" in resp.json()["error"]


def test_job_rejects_non_list_model_paths(tmp_path, monkeypatch):
    client = _jobs_client(tmp_path, monkeypatch)
    for bad in ("data/model/x.joblib", [123], ["ok", 42]):
        resp = client.post(
            "/api/jobs",
            json={"type": "backtest", "payload": {"model_paths": bad}},
        )
        assert resp.status_code == 400
        assert "model_paths" in resp.json()["error"]


# ------------------------------------------------------------- model registry


def test_models_api_lists_index(tmp_path, monkeypatch):
    import json

    import f1web.app as app_module

    client = _config_client(tmp_path, monkeypatch)
    model_dir = tmp_path / "data" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "index.json").write_text(
        json.dumps({"hurdle": {"checkpoint": "data/model/hurdle.joblib", "rows": 208}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "REPO_ROOT", tmp_path)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"]["hurdle"]["rows"] == 208
    assert body["default"] == "data/model/hurdle.joblib"


def test_models_api_fallback_lists_joblibs_excluding_calibrators(tmp_path, monkeypatch):
    import f1web.app as app_module

    client = _config_client(tmp_path, monkeypatch)
    model_dir = tmp_path / "data" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "hurdle.joblib").write_bytes(b"x")
    (model_dir / "experiment.joblib").write_bytes(b"x")
    (model_dir / "calibrators.joblib").write_bytes(b"x")  # not a model — excluded
    monkeypatch.setattr(app_module, "REPO_ROOT", tmp_path)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert set(resp.json()["models"]) == {"hurdle", "experiment"}


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


# -------------------------------------------- predict: CLI-option overrides


def _canned_prediction():
    import numpy as np
    import pandas as pd

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


def test_predict_forwards_refresh_and_model_path(tmp_path, monkeypatch):
    """POST /api/predict mirrors the CLI's --refresh and --model."""
    import f1web.app as app_module

    client = _config_client(tmp_path, monkeypatch)
    seen = {}

    def fake_prediction(**kw):
        seen.update(kw)
        return _canned_prediction()

    monkeypatch.setattr(app_module, "get_prediction", fake_prediction)
    resp = client.post(
        "/api/predict",
        json={"refresh": True, "model_path": "data/model/other.joblib"},
    )
    assert resp.status_code == 200
    assert seen["refresh"] is True
    assert seen["model_path"] == "data/model/other.joblib"


def test_predict_rejects_bad_override_types(tmp_path, monkeypatch):
    client = _config_client(tmp_path, monkeypatch)
    assert client.post("/api/predict", json={"refresh": "yes"}).status_code == 400
    assert client.post("/api/predict", json={"model_path": 42}).status_code == 400
    assert client.post("/api/predict", json={"write_report": "yep"}).status_code == 400


def test_predict_write_report_writes_markdown(tmp_path, monkeypatch):
    """write_report mirrors `f1 predict --out`: same report, config path."""
    import f1web.app as app_module

    client = _config_client(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "REPO_ROOT", tmp_path)

    def fake_prediction(**kw):
        return _canned_prediction()

    monkeypatch.setattr(app_module, "get_prediction", fake_prediction)
    resp = client.post(
        "/api/predict", json={"season": 2024, "round": 1, "write_report": True}
    )
    assert resp.status_code == 200
    report = tmp_path / "reports" / "prediction.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "# Prediction:" in text
    assert "2024 Round 1" in text
