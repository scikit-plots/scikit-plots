from __future__ import annotations

"""Tests for tracking server readiness checks.

Focus: deterministic fallback from HEAD to GET when a server returns 405.
"""

import urllib.error

from scikitplot.mlflow._readiness import wait_tracking_ready


class DummyResp:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"{}"


def test_wait_tracking_ready_fallback(monkeypatch) -> None:
    import scikitplot.mlflow._readiness as r

    # Keep the test fast and deterministic (no real sleeps).
    monkeypatch.setattr(r.time, "sleep", lambda *_a, **_k: None)

    calls = {"n": 0}

    def urlopen(req, timeout=2):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 405, "Method Not Allowed", hdrs=None, fp=None)
        return DummyResp(200)

    monkeypatch.setattr(r.urllib.request, "urlopen", urlopen)
    wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.05, server=None)
    assert calls["n"] == 2
