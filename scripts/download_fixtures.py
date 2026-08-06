"""Download recorded API fixtures used by the offline test suite.

Run manually (network required)::

    python scripts/download_fixtures.py

Writes JSON files under tests/fixtures/. The paged calendar fixtures are
synthesized from the full 2024 calendar by splitting it into two pages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
BASE = "https://api.jolpi.ca/ergast/f1"

FIXTURES = {
    "results_2024_r1.json": f"{BASE}/2024/1/results.json",
    "qualifying_2024_r1.json": f"{BASE}/2024/1/qualifying.json",
    "calendar_2024.json": f"{BASE}/2024.json",
    "sprint_2024_r5.json": f"{BASE}/2024/5/sprint.json",
    "standings_2024.json": f"{BASE}/2024/driverstandings.json",
}

HEADERS = {"User-Agent": "f1-result-predictor fixture downloader (dev@example.com)"}


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FIXTURES.items():
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        (FIXTURE_DIR / name).write_text(resp.text, encoding="utf-8")
        print(f"saved {name} ({len(resp.text)} bytes)")

    # Synthesize three pages of the 2024 calendar (total=24, limit=10).
    full = json.loads((FIXTURE_DIR / "calendar_2024.json").read_text(encoding="utf-8"))
    races = full["MRData"]["RaceTable"]["Races"]
    for page, start in ((1, 0), (2, 10), (3, 20)):
        doc = json.loads(json.dumps(full))
        doc["MRData"]["limit"] = "10"
        doc["MRData"]["offset"] = str(start)
        doc["MRData"]["total"] = str(len(races))
        doc["MRData"]["RaceTable"]["Races"] = races[start : start + 10]
        path = FIXTURE_DIR / f"calendar_2024_p{page}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        print(f"saved {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
