# scikitplot/mlflow/tests/test__readiness.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._readiness.

Naming convention: test__<module_name>.py

Covers
------
- wait_tracking_ready : non-HTTP URI raises ValueError,
                        immediate 200 response returns without sleeping,
                        ConnectionRefusedError retries until TimeoutError,
                        TimeoutError message includes attempt count,
                        404 on search → fallback to list endpoint,
                        405 on search → fallback to list endpoint,
                        non-404/405 HTTP error does NOT trigger fallback (e.g. 500),
                        custom poll_interval_s value is passed to time.sleep,
                        server exits early → RuntimeError with "exited before becoming ready",
                        _build_url constructs correct REST path

Notes
-----
All I/O is mocked via monkeypatch.  time.sleep is patched to prevent actual delays.
SpawnedServer is constructed with a fake Popen-like object that has poll() returning
a non-None value to simulate early server exit.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import pytest

import scikitplot.mlflow._readiness as _r
from scikitplot.mlflow._readiness import wait_tracking_ready
from scikitplot.mlflow._server import SpawnedServer


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Resp200:
    """Minimal HTTP response stub that reports status=200."""

    status = 200

    def __enter__(self) -> "_Resp200":
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def read(self) -> bytes:
        return b"{}"


class _FakeProcess:
    """Minimal Popen stub used to simulate a process that has exited."""

    pid = 99
    returncode: Optional[int]
    stdout = None

    def __init__(self, *, exited: bool = True) -> None:
        self.returncode = 1 if exited else None

    def poll(self) -> Optional[int]:
        return self.returncode


def _make_spawned_server(*, exited: bool = True) -> SpawnedServer:
    return SpawnedServer(
        _process=_FakeProcess(exited=exited),  # type: ignore[arg-type]
        _command=["python", "-m", "mlflow", "server"],
        _started_at=time.time(),
    )


# ===========================================================================
# wait_tracking_ready
# ===========================================================================


class TestWaitTrackingReady:
    """Tests for wait_tracking_ready()."""

    def test_invalid_uri_raises_value_error(self) -> None:
        """Non-HTTP(S) URI must raise ValueError immediately, before any network call."""
        with pytest.raises(ValueError):
            wait_tracking_ready("file:///tmp/mlruns", timeout_s=0.1)

    def test_sqlite_uri_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            wait_tracking_ready("sqlite:///mlflow.db", timeout_s=0.1)

    def test_immediate_200_returns_without_sleeping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 200 on the first attempt must return immediately (no sleep called)."""
        sleep_calls: list = []
        monkeypatch.setattr(_r.urllib.request, "urlopen", lambda *a, **k: _Resp200())
        monkeypatch.setattr(_r.time, "sleep", lambda s: sleep_calls.append(s))
        wait_tracking_ready("http://127.0.0.1:5000", timeout_s=1.0)
        assert sleep_calls == [], "No sleep should occur on immediate success"

    def test_connection_refused_retries_until_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ConnectionRefusedError must trigger retries and eventually TimeoutError."""
        monkeypatch.setattr(
            _r.urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()),
        )
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        with pytest.raises(TimeoutError):
            wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.01)

    def test_timeout_error_message_includes_attempt_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TimeoutError message must mention attempt count for diagnosability."""

        def _always_refuse(*a: Any, **k: Any) -> None:
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(_r.urllib.request, "urlopen", _always_refuse)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        with pytest.raises(TimeoutError, match="attempt"):
            wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.005)

    def test_fallback_405_uses_list_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        405 from the search endpoint must trigger fallback to the list endpoint.

        Notes
        -----
        MLflow < 2.x does not expose the POST /experiments/search endpoint and
        returns 405 Method Not Allowed.  The fallback to GET /experiments/list
        ensures compatibility with those older server versions.
        """
        calls: list = []

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(
                    None, 405, "Not Allowed", {}, None  # type: ignore[arg-type]
                )
            return _Resp200()

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.5)
        assert len(calls) == 2

    def test_fallback_404_uses_list_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """404 from the search endpoint must also trigger fallback."""
        calls: list = []

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(
                    None, 404, "Not Found", {}, None  # type: ignore[arg-type]
                )
            return _Resp200()

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.5)
        assert len(calls) == 2

    def test_http_500_does_not_trigger_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A 500 Internal Server Error must NOT trigger the list-endpoint fallback.

        Notes
        -----
        500 means the server is up but broken; retrying on the list endpoint would
        produce false readiness signals.  Only 404/405 indicate endpoint absence.
        """
        per_attempt_calls: list = []

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            per_attempt_calls.append(getattr(req, "full_url", "?"))
            raise urllib.error.HTTPError(
                None, 500, "Internal Server Error", {}, None  # type: ignore[arg-type]
            )

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        with pytest.raises(TimeoutError):
            wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.01)

        # Only the search endpoint should have been called — no list fallback.
        search_calls = [u for u in per_attempt_calls if "search" in u]
        list_calls = [u for u in per_attempt_calls if "list" in u]
        assert len(search_calls) >= 1
        assert len(list_calls) == 0, (
            "500 error must not trigger fallback to list endpoint"
        )

    def test_custom_poll_interval_passed_to_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """poll_interval_s must be forwarded to time.sleep on each retry."""
        sleep_vals: list = []
        call_no = [0]

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            call_no[0] += 1
            if call_no[0] < 3:
                raise ConnectionRefusedError()
            return _Resp200()

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda s: sleep_vals.append(s))
        wait_tracking_ready(
            "http://127.0.0.1:5000", timeout_s=5.0, poll_interval_s=0.07
        )
        assert len(sleep_vals) >= 1
        assert all(s == pytest.approx(0.07) for s in sleep_vals)

    def test_server_exits_early_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When the managed server process exits before becoming ready, must raise
        RuntimeError with a message indicating premature exit.
        """
        monkeypatch.setattr(
            _r.urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()),
        )
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        sv = _make_spawned_server(exited=True)
        with pytest.raises(RuntimeError, match="exited before becoming ready"):
            wait_tracking_ready("http://127.0.0.1:5000", timeout_s=5.0, server=sv)

    def test_no_server_object_does_not_check_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When server=None, must not attempt process.poll() (no AttributeError)."""
        call_no = [0]

        def urlopen(req: Any, timeout: float = 2.0) -> Any:
            call_no[0] += 1
            if call_no[0] < 2:
                raise ConnectionRefusedError()
            return _Resp200()

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        # Must not raise AttributeError for missing .process
        wait_tracking_ready("http://127.0.0.1:5000", timeout_s=1.0, server=None)


# ===========================================================================
# _build_url (internal helper)
# ===========================================================================


class TestBuildUrl:
    """Tests for _readiness._build_url()."""

    def test_appends_path_correctly(self) -> None:
        url = _r._build_url("http://127.0.0.1:5000", "/api/2.0/mlflow/experiments/search")
        assert url == "http://127.0.0.1:5000/api/2.0/mlflow/experiments/search"

    def test_strips_trailing_slash_from_base(self) -> None:
        url = _r._build_url("http://host:5000/", "/api/path")
        assert url == "http://host:5000/api/path"

    def test_non_http_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _r._build_url("file:///tmp/x", "/api/path")

    def test_https_accepted(self) -> None:
        url = _r._build_url("https://secure.host:443", "/api/path")
        assert url.startswith("https://")


# ===========================================================================
# Module structure
# ===========================================================================


class TestModuleStructure:
    """Tests for public API surface of _readiness."""

    def test_all_exports_present(self) -> None:
        assert set(_r.__all__) == {"wait_tracking_ready"}

    def test_default_poll_interval_is_positive(self) -> None:
        assert _r._DEFAULT_POLL_INTERVAL_S > 0

    def test_default_request_timeout_is_positive(self) -> None:
        assert _r._DEFAULT_REQUEST_TIMEOUT_S > 0


# ===========================================================================
# Backward-compatible module-level test retained from original test__readiness.py
# ===========================================================================


def test_wait_tracking_ready_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """HEAD→GET fallback on 405 (original integration test)."""
    monkeypatch.setattr(_r.time, "sleep", lambda *_a, **_k: None)
    calls: dict = {"n": 0}

    def urlopen(req: Any, timeout: float = 2.0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 405, "Method Not Allowed", {}, None  # type: ignore[arg-type]
            )
        return _Resp200()

    monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
    wait_tracking_ready("http://127.0.0.1:5000", timeout_s=0.05, server=None)
    assert calls["n"] == 2


# ===========================================================================
# Gap-fill: non-200 status from search endpoint (lines 153-156)
# ===========================================================================


class TestReadinessNon200Paths:
    """
    Cover the branches where urlopen succeeds (no exception) but status != 200.
    These are lines 153-156 (search non-200) and 178-186 (list non-200 / error).
    """

    def test_non_200_status_from_search_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When the search endpoint returns a non-200 status code (not via exception),
        the poller must retry rather than treating it as success.
        """
        call_no = [0]

        class _Resp:

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b""

        class _Resp202(_Resp):
            status = 202   # non-200: should not be accepted as "ready"

        class _Resp200(_Resp):
            status = 200

        def urlopen(req, timeout=2.0):
            call_no[0] += 1
            if call_no[0] < 3:
                return _Resp202()   # non-200 → must retry
            return _Resp200()       # eventually 200

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        wait_tracking_ready("http://127.0.0.1:5000", timeout_s=5.0)
        assert call_no[0] >= 3

    def test_list_endpoint_non_200_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        After 405→fallback to list endpoint, if the list endpoint also returns
        non-200, the poller must retry (lines 178-180).
        """
        call_no = [0]

        class _Resp:

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b""

        class _Resp202(_Resp):
            status = 202

        class _Resp200(_Resp):
            status = 200

        def urlopen(req, timeout=2.0):
            call_no[0] += 1
            full_url = getattr(req, "full_url", "") or ""
            if call_no[0] == 1:
                # First call: search → 405
                raise urllib.error.HTTPError(None, 405, "Not Allowed", {}, None)  # type: ignore[arg-type]
            if call_no[0] == 2:
                # Second call (list fallback): non-200
                return _Resp202()
            # Subsequent search attempts succeed
            return _Resp200()

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        wait_tracking_ready("http://127.0.0.1:5000", timeout_s=5.0)
        assert call_no[0] >= 3

    def test_list_endpoint_exception_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        After 405→fallback, if the list endpoint raises (network error),
        the poller must continue retrying from the top (lines 184-186).
        """
        call_no = [0]

        class _Resp200:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return b""

        def urlopen(req, timeout=2.0):
            call_no[0] += 1
            if call_no[0] == 1:
                raise urllib.error.HTTPError(None, 405, "Not Allowed", {}, None)  # type: ignore[arg-type]
            if call_no[0] == 2:
                # list fallback raises
                raise ConnectionResetError("reset by peer")
            return _Resp200()  # eventual success

        monkeypatch.setattr(_r.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(_r.time, "sleep", lambda _: None)
        wait_tracking_ready("http://127.0.0.1:5000", timeout_s=5.0)
        assert call_no[0] >= 3
