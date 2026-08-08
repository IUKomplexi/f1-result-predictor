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

import json
import time
from pathlib import Path

import requests

from httpclient import RETRYABLE_STATUS, CachedHTTPClient

DEFAULT_BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_USER_AGENT = (
    "f1-result-predictor/0.1.0 "
    "(https://github.com/example/f1-result-predictor; contact: dev@example.com)"
)
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 100


class F1APIError(RuntimeError):
    """Raised when the API response cannot be fetched or interpreted."""


class F1Client(CachedHTTPClient):
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
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(
            user_agent=user_agent,
            cache_dir=cache_dir,
            refresh=refresh,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
        )
        self.base_url = base_url.rstrip("/")

    # -- URL -----------------------------------------------------------------

    def _resolve(self, url_or_path: str) -> str:
        if url_or_path.startswith(("http://", "https://")):
            return url_or_path
        path = url_or_path if url_or_path.startswith("/") else f"/{url_or_path}"
        return f"{self.base_url}{path}"

    # -- request -------------------------------------------------------------

    def get_json(self, url_or_path: str, params: dict | None = None) -> dict:
        """GET a JSON document, using cache and retry/backoff as needed."""
        url = self._resolve(url_or_path)
        params = dict(params or {})
        cached = self._read_cache(url, params)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(self._backoff(attempt - 1))
            try:
                resp = self._get(url, params)
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
        params: dict | None = None,
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
