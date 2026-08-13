"""Fetch + cache race data for a range of seasons into data/raw.

The dashboard fetch job (``f1web/jobs.py``) and the ``scripts/fetch_all.py``
CLI shim share this ``run`` wrapper. It lives in the installed ``f1data``
package — not ``scripts/`` — so the web worker never depends on the repo root
being importable (the Docker image only installs declared packages).
"""

from __future__ import annotations

import time

from f1core.config import load_config
from f1data import F1Client, fetch_season


def run(
    *,
    start: int = 2010,
    end: int = 2026,
    refresh: bool = False,
    cache_dir: str = "data/raw",
    sleep: float = 0.2,
    cfg: dict | None = None,
    log=None,
) -> dict:
    """Fetch + cache seasons ``start``..``end``; returns a JSON-safe summary.

    ``log`` is an optional progress callback (web job runner); all arguments
    are keyword-only so the CLI and dashboard share one code path.
    """
    log = log or (lambda msg: print(msg, flush=True))
    cfg = cfg or load_config()
    client = F1Client(
        cache_dir=cache_dir, refresh=refresh, sleep_seconds=sleep,
        user_agent=cfg["api"]["user_agent"],
    )
    t0 = time.time()
    per_season = {}
    for season in range(start, end + 1):
        t = time.time()
        data = fetch_season(client, season)
        n_rounds = len(data["calendar"])
        n_results = sum(len(v) for v in data["results"].values())
        n_sprints = len(data["sprints"])
        per_season[str(season)] = {
            "rounds": n_rounds, "results": n_results, "sprints": n_sprints,
        }
        log(
            f"season {season}: {n_rounds} rounds, {n_results} result rows, "
            f"{n_sprints} sprints ({time.time() - t:.1f}s)"
        )
    elapsed = time.time() - t0
    log(f"total: {elapsed:.1f}s")
    return {"seasons": per_season, "start": start, "end": end, "elapsed_s": round(elapsed, 2)}
