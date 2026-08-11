"""In-process async job runner for pipeline steps (``threading``-based).

Pipeline jobs (fetch data, train, calibrate, backtest, search) can take
minutes, so they must not block a FastAPI request. This module runs each job
on a dedicated worker thread behind a single-job queue (jobs share ``data/``
artifacts, so only one runs at a time) and records progress + JSON-safe
results in memory plus a durable ``reports/jobs/*.json`` history.

Jobs are tied to the server process lifetime: uvicorn ``--reload`` restarts
kill in-flight jobs. The ``reports/jobs/*.json`` history survives restarts as a
record of what ran, but is not automatically reloaded into memory.
"""

from __future__ import annotations

import json
import queue
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
    "train": "Train",
    "calibrate": "Calibrate",
    "backtest": "Backtest",
    "search": "Search",
}

# Payload keys each job type accepts (defaults are read from config / sensible
# fallbacks; only these keys are passed through to the underlying ``run_*``).
JOB_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "fetch": ("start", "end", "refresh"),
    "train": ("refresh",),
    "calibrate": ("fit_through_season", "eval_from_season", "refresh"),
    "backtest": ("quantize", "refresh"),
    "search": ("n", "max_test_season", "seed", "refresh"),
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

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda j: j["created_at"], reverse=True
            )

    def _run_loop(self) -> None:
        while True:
            self._process(self._queue.get())

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


# --------------------------------------------------------------------------
# Default handlers: dispatch a job payload to the shared ``run_*`` wrappers
# --------------------------------------------------------------------------

def _cfg_start_end(cfg: dict, payload: dict) -> tuple[int, int]:
    return (
        int(payload.get("start", cfg["data"]["start_season"])),
        int(payload.get("end", cfg["data"]["end_season"])),
    )


def _fetch_handler(payload: dict, log) -> dict:
    from scripts.fetch_all import run as run_fetch

    cfg = _load_config()
    start, end = _cfg_start_end(cfg, payload)
    return run_fetch(
        start=start, end=end, refresh=bool(payload.get("refresh", False)),
        cfg=cfg, log=log,
    )


def _train_handler(payload: dict, log) -> dict:
    from model.train import run as run_train

    cfg = _load_config()
    return run_train(refresh=bool(payload.get("refresh", False)), cfg=cfg, log=log)


def _calibrate_handler(payload: dict, log) -> dict:
    from model.calibrate import run as run_calibrate

    cfg = _load_config()
    return run_calibrate(
        refresh=bool(payload.get("refresh", False)), cfg=cfg, log=log,
        fit_through_season=_int_or_none(payload.get("fit_through_season")),
        eval_from_season=_int_or_none(payload.get("eval_from_season")),
    )


def _int_or_none(value) -> int | None:
    """Coerce a job payload value to an int, or None when empty/invalid."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _backtest_handler(payload: dict, log) -> dict:
    from model.evaluate import run as run_backtest

    cfg = _load_config()
    return run_backtest(
        quantize=bool(payload.get("quantize", True)),
        refresh=bool(payload.get("refresh", False)),
        cfg=cfg, log=log,
    )


def _search_handler(payload: dict, log) -> dict:
    from model.search import run as run_search

    cfg = _load_config()
    return run_search(
        n=int(payload.get("n", 16)),
        seed=int(payload.get("seed", 0)),
        max_test_season=int(payload.get("max_test_season", 2019)),
        refresh=bool(payload.get("refresh", False)),
        cfg=cfg, log=log,
    )


def _load_config() -> dict:
    from f1core.config import load_config

    # Anchor to the repo root so jobs read/write the same config.toml as the
    # /api/config endpoints regardless of the server's working directory.
    return load_config(REPO_ROOT / "config.toml")


def register_default_handlers(manager: JobManager) -> JobManager:
    """Attach the default fetch/train/calibrate/backtest/search handlers."""
    manager.register("fetch", _fetch_handler)
    manager.register("train", _train_handler)
    manager.register("calibrate", _calibrate_handler)
    manager.register("backtest", _backtest_handler)
    manager.register("search", _search_handler)
    return manager
