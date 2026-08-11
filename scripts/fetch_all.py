"""Fetch and cache race data for a range of seasons into data/raw.

Usage::

    python scripts/fetch_all.py [--start 2010] [--end 2025] [--refresh] [--sleep 0.2]

A fresh run pulls every season from the Jolpica API and caches each raw JSON
response under ``data/raw/``; subsequent runs (even with ``--sleep 0``) work
fully offline from the cache. The dashboard triggers the same work through
:func:`run` as an async job.
"""

from __future__ import annotations

import argparse
import sys
import time

from f1core.config import load_config
from f1data import F1Client, fetch_season


def run(
    *,
    start: int = 2010,
    end: int = 2025,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true", help="refetch even if cached")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between requests")
    args = parser.parse_args()

    run(
        start=args.start, end=args.end, refresh=args.refresh,
        cache_dir=args.cache_dir, sleep=args.sleep,
        log=lambda msg: print(msg, flush=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
