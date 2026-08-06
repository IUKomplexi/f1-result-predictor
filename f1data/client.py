"""Thin HTTP client for the Jolpica F1 API (Ergast-compatible endpoints).

Responsibilities:
  - send a descriptive ``User-Agent`` header (required by the API terms of use)
  - polite rate limiting between network requests
  - retry with exponential backoff on 429 / 5xx / network errors
  - transparent on-disk caching of raw JSON responses under ``data/raw/``
  - ``limit``/``offset`` pagination for responses larger than one page

Usage::

    from f1data import F1Client

    client = F1Client(user_agent="my-app/0.1 (contact@example.com)")
    data = client.get_json("/2024/1/results.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_USER_AGENT = (
    "f1-result-predictor/0.1.0 "
    "(https://github.com/example/f1-result-predictor; contact: dev@example.com)"
)
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 100
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class F1APIError(RuntimeError):
    """Raised when the API response cannot be fetched or interpreted."""


def _safe_name(url: str) -> str:
    """Map a URL to a filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", url)


class F1Client:
    """A cached HTTP client for the Jolpica F1 (Ergast) API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        cache_dir: str | Path = "data/raw",
        refresh: bool = False,
        sleep_seconds: float = 0.25,
        timeout: float = 30.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()
        self.session.headers["User-Agent"] = self.user_agent

    # -- URL / cache helpers -------------------------------------------------

    def _resolve(self, url_or_path: str) -> str:
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return url_or_path
        path = url_or_path if url_or_path.startswith("/") else f"/{url_or_path}"
        return f"{self.base_url}{path}"

    def _cache_key(self, url: str, params: dict) -> Path:
        if params:
            digest = hashlib.sha1(
                json.dumps(params, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            return self.cache_dir / f"{_safe_name(url)}__{digest}.json"
        return self.cache_dir / f"{_safe_name(url)}.json"

    def _read_cache(self, url: str, params: dict) -> Optional[dict]:
        if self.refresh:
            return None
        path = self._cache_key(url, params)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Ignoring unreadable cache file %s", path)
            return None

    def _write_cache(self, url: str, params: dict, payload: dict) -> None:
        path = self._cache_key(url, params)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError:
            logger.warning("Could not write cache file %s", path)

    # -- request -------------------------------------------------------------

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after and retry_after.isdigit():
            return float(retry_after)
        return min(2.0 ** attempt, 10.0)

    def get_json(self, url_or_path: str, params: Optional[dict] = None) -> dict:
        """GET a JSON document, using cache and retry/backoff as needed."""
        url = self._resolve(url_or_path)
        params = dict(params or {})
        cached = self._read_cache(url, params)
        if cached is not None:
            return cached

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(self._backoff(attempt - 1, None))
            try:
                if self.sleep_seconds:
                    # Jittered polite delay between requests.
                    time.sleep(self.sleep_seconds * (1.0 + random.random()))
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                continue

            if resp.status_code in RETRYABLE_STATUS:
                last_error = F1APIError(
                    f"GET {url} returned HTTP {resp.status_code}"
                )
                if attempt >= self.max_retries:
                    break
                time.sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                continue
            if resp.status_code >= 400:
                raise F1APIError(f"GET {url} failed with HTTP {resp.status_code}")

            try:
                payload = resp.json()
            except json.JSONDecodeError as exc:
                raise F1APIError(f"GET {url} returned invalid JSON: {exc}") from exc

            self._write_cache(url, params, payload)
            return payload

        raise F1APIError(
            f"GET {url} failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def get_paged(
        self,
        url_or_path: str,
        params: Optional[dict] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict:
        """GET a document, following ``limit``/``offset`` pagination.

        Returns the merged ``MRData`` document with all items concatenated.
        Non-paginated documents (no RaceTable/StandingsTable) are returned
        unchanged.
        """
        params = dict(params or {})
        limit = min(int(params.get("limit", page_size)), MAX_PAGE_SIZE)
        offset = int(params.get("offset", 0))
        page_params = {**params, "limit": limit, "offset": offset}

        first = self.get_json(url_or_path, page_params)
        mdata = first.get("MRData", {})
        table_key = (
            "RaceTable"
            if "RaceTable" in mdata
            else "StandingsTable" if "StandingsTable" in mdata else None
        )
        if table_key is None:
            return first
        table = mdata[table_key]
        item_key = (
            "Races"
            if "Races" in table
            else "StandingsLists" if "StandingsLists" in table else None
        )
        if item_key is None:
            return first

        total = int(mdata.get("total", len(table[item_key])))
        items = list(table[item_key])
        offset += len(table[item_key])

        # A page that returns fewer items than the requested limit is the last
        # page. This also protects against endpoints where ``total`` counts
        # nested driver entries rather than paginated items (e.g. per-round
        # endpoints return one Race containing all results, with
        # total == number of drivers): one page is already complete.
        while offset < total and len(items) >= limit:
            page_params = {**params, "limit": limit, "offset": offset}
            page = self.get_json(url_or_path, page_params)
            page_table = page.get("MRData", {}).get(table_key, {})
            chunk = page_table.get(item_key, [])
            if not chunk:
                break
            items.extend(chunk)
            offset += len(chunk)
            if len(chunk) < limit:
                break

        merged = dict(first)
        merged["MRData"] = {
            **mdata,
            table_key: {**table, item_key: items},
            "total": str(total),
        }
        return merged
