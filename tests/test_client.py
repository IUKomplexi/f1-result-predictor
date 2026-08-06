"""Unit tests for f1data.client.F1Client: caching, retries, pagination, headers."""

from __future__ import annotations

import json

import pytest
import requests

from f1data import F1APIError, F1Client

from helpers import FIXTURES, QueueSession


def test_get_json_parses_and_caches(tmp_path):
    session = QueueSession()
    session.add(status=200, payload={"MRData": {"total": "1"}})
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    first = client.get_json("/2024/1/results.json")
    second = client.get_json("/2024/1/results.json")

    assert first == {"MRData": {"total": "1"}}
    assert second == first
    # Only one network call; second hit served from the on-disk cache.
    assert len(session.calls) == 1
    assert list(tmp_path.glob("*.json"))


def test_cache_key_includes_params(tmp_path):
    session = QueueSession()
    session.add(status=200, payload={"a": 1})
    session.add(status=200, payload={"a": 2})
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    client.get_json("/x.json", {"limit": 10, "offset": 0})
    client.get_json("/x.json", {"limit": 10, "offset": 10})

    assert len(session.calls) == 2  # different params => separate cache entries


def test_refresh_bypasses_cache(tmp_path):
    session = QueueSession()
    session.add(status=200, payload={"a": 1})
    session.add(status=200, payload={"a": 1})
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0, refresh=True)

    client.get_json("/x.json")
    client.get_json("/x.json")

    assert len(session.calls) == 2


def test_retries_on_429_honoring_retry_after_then_succeeds(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("f1data.client.time.sleep", lambda s: sleeps.append(s))
    session = QueueSession()
    session.add(status=429, headers={"Retry-After": "1"})
    session.add(status=429)
    session.add(status=200, payload={"MRData": {"ok": True}})
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0, max_retries=3)

    payload = client.get_json("/x.json")

    assert payload == {"MRData": {"ok": True}}
    assert len(session.calls) == 3
    assert sleeps[0] == 1.0  # Retry-After honored


def test_retries_on_connection_error(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("f1data.client.time.sleep", lambda s: sleeps.append(s))
    calls = []

    def flaky_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if len(calls) < 3:
            raise requests.ConnectionError("boom")
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b'{"MRData": {"ok": true}}'
        return resp

    session = QueueSession()
    session.get = flaky_get  # type: ignore[method-assign]

    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0, max_retries=3)
    assert client.get_json("/x.json") == {"MRData": {"ok": True}}
    assert len(calls) == 3
    assert sleeps  # backoff happened


def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("f1data.client.time.sleep", lambda s: None)
    session = QueueSession()
    session.add(status=429)
    session.add(status=429)
    session.add(status=429)
    session.add(status=429)
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0, max_retries=3)

    with pytest.raises(F1APIError, match="after 4 attempts"):
        client.get_json("/x.json")
    assert len(session.calls) == 4


def test_malformed_json_raises(tmp_path):
    session = QueueSession()
    session.add(status=200, text="<html>not json</html>")
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    with pytest.raises(F1APIError, match="invalid JSON"):
        client.get_json("/x.json")


def test_non_retryable_http_error_raises_immediately(tmp_path):
    session = QueueSession()
    session.add(status=404, text="not found")
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0, max_retries=3)

    with pytest.raises(F1APIError, match="HTTP 404"):
        client.get_json("/x.json")
    assert len(session.calls) == 1


def test_user_agent_is_sent():
    client = F1Client(sleep_seconds=0)
    assert client.session.headers["User-Agent"] == client.user_agent
    assert "f1-result-predictor" in client.user_agent


def test_pagination_merges_pages(tmp_path):
    session = QueueSession()
    pages = [
        json.loads((FIXTURES / f"calendar_2024_p{n}.json").read_text(encoding="utf-8"))
        for n in (1, 2, 3)
    ]
    for page in pages:
        session.add(status=200, payload=page)
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    data = client.get_paged("/2024.json", page_size=10)

    races = data["MRData"]["RaceTable"]["Races"]
    assert len(races) == 24
    assert races[0]["round"] == "1"
    assert races[-1]["round"] == "24"
    assert data["MRData"]["total"] == "24"


def test_pagination_requests_second_page_with_offset(tmp_path):
    session = QueueSession()
    pages = [
        json.loads((FIXTURES / f"calendar_2024_p{n}.json").read_text(encoding="utf-8"))
        for n in (1, 2, 3)
    ]
    for page in pages:
        session.add(status=200, payload=page)
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    client.get_paged("/2024.json", page_size=10)

    offsets = [params.get("offset") for _, params in session.calls]
    assert offsets == [0, 10, 20]


def test_per_round_endpoint_does_not_paginate(tmp_path):
    # Jolpica's per-round endpoints report total == number of drivers while the
    # item list holds a single Race containing all of them. get_paged must not
    # page through the driver count — one request is enough.
    results = json.loads((FIXTURES / "results_2024_r1.json").read_text(encoding="utf-8"))
    session = QueueSession()
    session.add(status=200, payload=results)
    client = F1Client(cache_dir=tmp_path, session=session, sleep_seconds=0)

    data = client.get_paged("/2024/1/results.json")

    races = data["MRData"]["RaceTable"]["Races"]
    assert len(races) == 1
    assert len(races[0]["Results"]) == 20
    assert len(session.calls) == 1  # regression: was 20 requests before the fix
