"""In-process async job runner for pipeline steps (``threading``-based).

Pipeline jobs (fetch data, precompute history, train, calibrate, backtest)
can take minutes, so they must not block a FastAPI request. This module runs each job
on a dedicated worker thread behind a single-job queue (jobs share ``data/``
artifacts, so only one runs at a time) and records progress + JSON-safe
results in memory plus a durable ``reports/jobs/*.json`` history.

Jobs are tied to the server process lifetime: uvicorn ``--reload`` restarts
kill in-flight jobs. The ``reports/jobs/*.json`` history survives restarts and
is reloaded into memory at startup — finished jobs reappear in the dashboard,
while jobs left ``queued``/``running`` by the dead process are marked
``interrupted`` (they never ran to completion).
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = REPO_ROOT / "reports" / "jobs"

# type -> human-readable label, shown in the dashboard Pipeline page.
JOB_TYPES: dict[str, str] = {
    "fetch": "Fetch data",
    "history": "Precompute race history",
    "train": "Train",
    "calibrate": "Calibrate",
    "backtest": "Backtest",
}

# Terminal states a job can be in when the server restarts; anything else is
# a job the dead process never finished.
_FINISHED_STATUSES = ("done", "failed", "interrupted", "cancelled")

# Payload keys each job type accepts (defaults are read from config / sensible
# fallbacks; only these keys are passed through to the underlying ``run_*``).
# This is the web-side mirror of the CLI subcommand options in
# ``f1core/cli.py``: every result-affecting flag the CLI exposes is available
# to a job payload (path overrides stay config-managed; see README).
JOB_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "fetch": ("start", "end", "refresh"),
    "history": ("start", "end", "refresh"),
    "train": ("start", "end", "refresh", "name", "params", "enable_features", "disable_features"),
    "calibrate": (
        "start", "end", "refresh",
        "fit_through_season", "eval_from_season",
        "enable_features", "disable_features", "model_path",
    ),
    "backtest": (
        "start", "end", "refresh", "quantize", "use_checkpoint",
        "enable_features", "disable_features", "model_path", "model_paths",
    ),
}


class JobManager:
    """Owns the job store, the single-job queue, and the worker thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[[dict, Callable[[str], None]], dict]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(
            target=self._run_loop, name="pipeline-jobs", daemon=True
        )
        self._worker.start()
        self._reload_history()

    def register(self, job_type: str, handler: Callable) -> None:
        """Register the callable that runs a job type: ``handler(payload, log) -> dict``."""
        self._handlers[job_type] = handler

    def submit(self, job_type: str, payload: dict | None = None) -> str:
        """Queue a job and return its id. Raises ``KeyError`` for unknown types."""
        if job_type not in self._handlers:
            raise KeyError(f"unknown job type: {job_type}")
        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "id": job_id,
            "type": job_type,
            "label": JOB_TYPES.get(job_type, job_type),
            "payload": payload or {},
            "status": "queued",
            "log": [],
            "result": None,
            "error": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> str | None:
        """Cancel a queued job; returns an error message, or None on success.

        Only queued jobs can be cancelled safely — a running job is atomic
        (pipeline steps share ``data/`` artifacts and cannot be torn down
        mid-run). The worker skips cancelled jobs when it dequeues them.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return f"job not found: {job_id}"
            if job["status"] != "queued":
                return f"job is {job['status']}; only queued jobs can be cancelled"
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
        self._persist(job_id)
        return None

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda j: j["created_at"], reverse=True
            )

    def reset(self) -> None:
        """Drop all in-memory job state (used after a clear-data wipe)."""
        with self._lock:
            self._jobs.clear()

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            # A job cancelled while queued must never run.
            if job is None or job["status"] != "queued":
                continue
            self._process(job_id)

    def _process(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        with self._lock:
            job["status"] = "running"
            job["started_at"] = time.time()

        def log(line: str) -> None:
            with self._lock:
                job["log"].append(line)

        handler = self._handlers[job["type"]]
        try:
            result = handler(job["payload"], log)
            with self._lock:
                job["result"] = result
                job["status"] = "done"
        except BaseException as exc:  # noqa: BLE001 - a job failure is recorded, not fatal to the runner
            with self._lock:
                job["error"] = str(exc)
                job["status"] = "failed"
        with self._lock:
            job["finished_at"] = time.time()
        self._persist(job_id)

    def _persist(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        try:
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            (JOBS_DIR / f"{job_id}.json").write_text(
                json.dumps(job, indent=2), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError):
            # Disk or serialization failure in the durable history must not
            # kill the worker; the in-memory job record is still authoritative.
            pass

    def _reload_history(self) -> None:
        """Reload the durable ``reports/jobs/*.json`` history into memory.

        Finished jobs reappear in the dashboard after a server restart. Jobs
        the dead process left ``queued``/``running`` are marked
        ``interrupted`` (jobs never run across process lifetimes), so the
        history does not claim they completed. Reloaded jobs are never
        re-queued.
        """
        if not JOBS_DIR.is_dir():
            return
        for path in sorted(JOBS_DIR.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                continue
            status = record.get("status")
            if status in ("queued", "running"):
                record["status"] = "interrupted"
                record["error"] = record.get("error") or (
                    f"server restarted while this job was {status}"
                )
                record["finished_at"] = record.get("finished_at") or time.time()
            self._jobs[record["id"]] = record


# --------------------------------------------------------------------------
# Default handlers: dispatch a job payload to the shared ``run_*`` wrappers
# --------------------------------------------------------------------------

def _cfg_start_end(cfg: dict, payload: dict) -> tuple[int, int]:
    return (
        int(payload.get("start", cfg["data"]["start_season"])),
        int(payload.get("end", cfg["data"]["end_season"])),
    )


def _feature_toggles(payload: dict) -> tuple[list[str], list[str]]:
    """(enable_features, disable_features) string lists from a job payload.

    Mirrors the CLI's ``--enable-features/--disable-features``; non-string
    entries are dropped (the API layer already validates the shape).
    """
    return (
        [f for f in (payload.get("enable_features") or []) if isinstance(f, str)],
        [f for f in (payload.get("disable_features") or []) if isinstance(f, str)],
    )


def _fetch_handler(payload: dict, log) -> dict:
    from f1data.fetch import run as run_fetch

    cfg = _load_config()
    start, end = _cfg_start_end(cfg, payload)
    return run_fetch(
        start=start, end=end, refresh=bool(payload.get("refresh", False)),
        cfg=cfg, log=log,
    )


def _history_handler(payload: dict, log) -> dict:
    """Pre-warm the Race History cache: score every round of ``start..end``.

    Writes whole-season snapshots under ``data/predictions`` (keyed by season +
    feature fingerprint + params hash); repeat Race History opens then read the
    cache instead of rebuilding the featured frame. ``refresh`` clears the
    cache first so newly fetched rounds are picked up.
    """
    from f1core.predict import predict_season

    cfg = _load_config()
    start, end = _cfg_start_end(cfg, payload)
    cache_dir = REPO_ROOT / "data" / "predictions"
    if payload.get("refresh"):
        shutil.rmtree(cache_dir, ignore_errors=True)
        log("Cleared prediction cache (refresh)")
    summary = {}
    for season in range(start, end + 1):
        t0 = time.time()
        preds = predict_season(season, cfg=cfg, quiet=True, cache_dir=cache_dir)
        summary[str(season)] = {
            "rounds": len(preds), "elapsed_s": round(time.time() - t0, 2),
        }
        log(f"season {season}: {len(preds)} rounds ({time.time() - t0:.1f}s)")
    return {"start": start, "end": end, "seasons": summary}


def _checkpoint_path(cfg: dict, name: str | None) -> str:
    """Resolve the train-job checkpoint path.

    A non-empty ``name`` writes to ``data/model/<name>.joblib`` (sanitized —
    no path separators, so no traversal); blank keeps the config checkpoint
    so the default train→predict flow is unchanged.
    """
    if not name:
        return cfg["model"]["checkpoint"]
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip(".-") or "hurdle"
    return f"data/model/{slug}.joblib"


def _train_handler(payload: dict, log) -> dict:
    from model.calibrate import run as run_calibrate
    from model.train import run as run_train

    cfg = _load_config()
    start, end = _cfg_start_end(cfg, payload)
    enable, disable = _feature_toggles(payload)
    checkpoint = _checkpoint_path(cfg, payload.get("name"))
    result = run_train(
        start=start, end=end, refresh=bool(payload.get("refresh", False)),
        out=checkpoint,
        params=payload.get("params"),
        enable_features=enable, disable_features=disable,
        cfg=cfg, log=log,
    )
    # Calibration is part of Train: a trained model should predict with
    # calibrated probabilities right away. A named model is calibrated on its
    # own out-of-sample seasons (calibrators written next to the checkpoint);
    # the config-default model uses the shared [model] calibrators file fitted
    # on walk-forward out-of-sample scores.
    cal_kwargs = {
        "start": start, "end": end, "refresh": bool(payload.get("refresh", False)),
        "out": cfg["model"]["calibrators"],
        "enable_features": enable, "disable_features": disable,
        "cfg": cfg, "log": log,
    }
    if payload.get("name"):
        cal_kwargs["model_path"] = checkpoint
    try:
        cal = run_calibrate(**cal_kwargs)
        result["calibrated"] = True
        result["calibrators"] = cal.get("calibrators")
    except ValueError as exc:
        # E.g. the model was trained through the newest season, so no
        # out-of-sample seasons remain to fit calibrators on: the model itself
        # is fine, predictions just stay on raw probabilities.
        log(f"Calibration skipped: {exc}")
        result["calibrated"] = False
        result["calibrate_error"] = str(exc)
    return result


def _calibrate_handler(payload: dict, log) -> dict:
    from model.calibrate import run as run_calibrate

    cfg = _load_config()
    start, end = _cfg_start_end(cfg, payload)
    enable, disable = _feature_toggles(payload)
    return run_calibrate(
        start=start, end=end, refresh=bool(payload.get("refresh", False)),
        out=cfg["model"]["calibrators"],
        enable_features=enable, disable_features=disable,
        cfg=cfg, log=log,
        fit_through_season=_int_or_none(payload.get("fit_through_season")),
        eval_from_season=_int_or_none(payload.get("eval_from_season")),
        model_path=_str_or_none(payload.get("model_path")),
    )


def _int_or_none(value) -> int | None:
    """Coerce a job payload value to an int, or None when empty/invalid."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value) -> str | None:
    """A non-empty string payload value, else None."""
    if not isinstance(value, str) or value.strip() == "":
        return None
    return value.strip()


def _backtest_handler(payload: dict, log) -> dict:
    from model.evaluate import run as run_backtest

    cfg = _load_config()
    start, end = _cfg_start_end(cfg, payload)
    enable, disable = _feature_toggles(payload)
    model_paths = [p for p in (payload.get("model_paths") or []) if isinstance(p, str)]
    return run_backtest(
        start=start, end=end,
        quantize=bool(payload.get("quantize", True)),
        use_checkpoint=bool(payload.get("use_checkpoint", False)),
        model_path=_str_or_none(payload.get("model_path")),
        model_paths=model_paths,
        refresh=bool(payload.get("refresh", False)),
        enable_features=enable, disable_features=disable,
        cfg=cfg, log=log,
    )


def _load_config() -> dict:
    from f1core.config import load_config

    # Anchor to the repo root so jobs read/write the same config.toml as the
    # /api/config endpoints regardless of the server's working directory.
    return load_config(REPO_ROOT / "config.toml")


def register_default_handlers(manager: JobManager) -> JobManager:
    """Attach the default fetch/history/train/calibrate/backtest handlers."""
    manager.register("fetch", _fetch_handler)
    manager.register("history", _history_handler)
    manager.register("train", _train_handler)
    manager.register("calibrate", _calibrate_handler)
    manager.register("backtest", _backtest_handler)
    return manager
