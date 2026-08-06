"""Fetch and cache race data for a range of seasons into data/raw.

Usage::

    python scripts/fetch_all.py [--start 2010] [--end 2025] [--refresh] [--sleep 0.2]

A fresh run pulls every season from the Jolpica API and caches each raw JSON
response under ``data/raw/``; subsequent runs (even with ``--sleep 0``) work
fully offline from the cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1data import F1Client, fetch_season  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true", help="refetch even if cached")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between requests")
    args = parser.parse_args()

    client = F1Client(cache_dir=args.cache_dir, refresh=args.refresh, sleep_seconds=args.sleep)
    t0 = time.time()
    for season in range(args.start, args.end + 1):
        t = time.time()
        data = fetch_season(client, season)
        n_rounds = len(data["calendar"])
        n_results = sum(len(v) for v in data["results"].values())
        n_sprints = len(data["sprints"])
        print(
            f"season {season}: {n_rounds} rounds, {n_results} result rows, "
            f"{n_sprints} sprints ({time.time() - t:.1f}s)",
            flush=True,
        )
    print(f"total: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
