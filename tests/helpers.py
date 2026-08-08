"""Shared test doubles: a fake requests.Session serving recorded fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import requests

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _response(status: int, payload=None, text: str | None = None,
              headers: dict[str, str] | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    if payload is not None:
        resp._content = json.dumps(payload).encode("utf-8")
    elif text is not None:
        resp._content = text.encode("utf-8")
    else:
        resp._content = b""
    resp.headers = requests.models.CaseInsensitiveDict(headers or {})
    return resp


def ok_response(payload: dict) -> requests.Response:
    """A 200 response carrying a JSON payload."""
    return _response(200, payload=payload)


class QueueSession:
    """Serves responses from a queue in order; repeats the last one forever.

    Records every ``(url, params)`` call so tests can assert request counts.
    """

    def __init__(self) -> None:
        self.calls: list = []
        self.queue: list = []
        self.headers = requests.models.CaseInsensitiveDict()

    def add(self, status: int = 200, payload=None, text: str | None = None,
            headers: dict[str, str] | None = None) -> None:
        self.queue.append(_response(status, payload, text, headers))

    def get(self, url: str, params=None, timeout=None) -> requests.Response:
        self.calls.append((url, dict(params or {})))
        if self.queue:
            return self.queue.pop(0)
        return _response(200, {"MRData": {}})


class RouteSession:
    """Serves fixture files by matching the URL suffix."""

    def __init__(self, routes: dict[str, Path]) -> None:
        self.routes = routes
        self.calls: list = []
        self.headers = requests.models.CaseInsensitiveDict()

    def get(self, url: str, params=None, timeout=None) -> requests.Response:
        self.calls.append((url, dict(params or {})))
        for suffix, path in self.routes.items():
            if url.endswith(suffix):
                return _response(200, json.loads(path.read_text(encoding="utf-8")))
        raise AssertionError(f"No fixture route for {url}")

    @classmethod
    def for_fixture_names(cls, names: dict[str, str]) -> RouteSession:
        routes = {suffix: FIXTURES / name for suffix, name in names.items()}
        return cls(routes)
