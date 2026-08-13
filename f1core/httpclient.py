"""Shared plumbing for the project's polite cached HTTP clients.

``f1data.F1Client`` wraps the Jolpica API with a disciplined request
pattern: a descriptive ``User-Agent``, polite rate limiting, retry/backoff
on 429/5xx, and transparent on-disk caching of raw responses. The identical
pieces — request setup, cache keying/read/write, backoff, and the jittered
request itself — live here; the client keeps its own ``get_json`` because
URL resolution, ``Retry-After`` handling, error types, and per-request
cache bypass genuinely differ per API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _safe_name(url: str) -> str:
    """Map a URL to a filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", url)


class CachedHTTPClient:
    """Polite, cached, retrying HTTP client for a JSON API.

    Subclasses pass their ``User-Agent`` and ``cache_dir`` through
    :meth:`__init__` and implement :meth:`get_json` on top of :meth:`_get`,
    :meth:`_backoff` and the ``_*_cache`` helpers.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path = "data/raw",
        refresh: bool = False,
        sleep_seconds: float = 0.25,
        timeout: float = 30.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()
        self.session.headers["User-Agent"] = self.user_agent

    # -- cache ---------------------------------------------------------------

    def _cache_key(self, url: str, params: dict) -> Path:
        if params:
            digest = hashlib.sha1(
                json.dumps(params, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            return self.cache_dir / f"{_safe_name(url)}__{digest}.json"
        return self.cache_dir / f"{_safe_name(url)}.json"

    def _read_cache(self, url: str, params: dict) -> dict | None:
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

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after and retry_after.isdigit():
            return float(retry_after)
        return min(2.0 ** attempt, 10.0)

    def _get(self, url: str, params: dict) -> requests.Response:
        """One network attempt with the polite jittered delay."""
        if self.sleep_seconds:
            # Jittered polite delay between requests.
            time.sleep(self.sleep_seconds * (1.0 + random.random()))
        return self.session.get(url, params=params, timeout=self.timeout)
