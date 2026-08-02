# corpus/tests/test__url_handler_redirects.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Per-hop SSRF redirect-validation gate (CORPUS-NET-001)
======================================================

``_get_with_validated_redirects`` is the single transport that follows HTTP
redirects. The invariant under test: **no request connects to a blocked/private
address at any redirect hop**. Automatic redirects are disabled and every hop
(initial URL and each ``Location``) is SSRF-validated *before* the connection is
opened, so a public URL can never be transparently redirected to a loopback,
RFC-1918, link-local, IPv6-local, or cloud-metadata target.

A fake session and a deterministic hostname-based ``_is_private_ip`` avoid real
network/DNS, so the suite is hermetic.

Run with::

    pytest scikitplot/corpus/tests/test__url_handler_redirects.py -v
"""

from __future__ import annotations

import pytest

from scikitplot.corpus import _url_handler as U

_BLOCKED_HOSTS = {"metadata", "internal"}


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch):
    # Real _validate_url_security now resolves via getaddrinfo; map test hosts
    # deterministically. Blocked hostnames stand in for internal targets; IP
    # literals (127.0.0.1, ::1, ...) are classified without DNS.
    import socket

    def fake(host, *a, **k):
        ip = "10.0.0.1" if host in _BLOCKED_HOSTS else "93.184.216.34"
        return [(2, 1, 6, "", (ip, 0))]

    monkeypatch.setattr(U.socket, "getaddrinfo", fake)


class _Resp:
    def __init__(self, status, location=None):
        self.status_code = status
        self.headers = {"Location": location} if location else {}

    def close(self):
        pass

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self, script):
        self.script = script
        self.connected: list[str] = []
        self.kwargs: list[dict] = []

    def get(self, url, **kw):
        self.connected.append(url)
        self.kwargs.append(kw)
        return self.script[url]

    head = get


def _get(session, url, **kw):
    return U._get_with_validated_redirects(session, url, timeout=5, **kw)


class TestRedirectSSRF:
    def test_redirect_to_metadata_blocked_before_connecting(self):
        s = _Session({"http://pub/f": _Resp(302, "http://metadata/latest/")})
        with pytest.raises(ValueError):
            _get(s, "http://pub/f")
        assert all("metadata" not in u for u in s.connected)   # never connected
        assert s.connected == ["http://pub/f"]

    @pytest.mark.parametrize("target", [
        "http://127.0.0.1/x", "http://10.0.0.1/x", "http://[::1]/x",
        "http://169.254.169.254/x",
    ])
    def test_redirect_to_private_blocked(self, target):
        s = _Session({"http://pub/s": _Resp(301, target)})
        with pytest.raises(ValueError):
            _get(s, "http://pub/s")
        assert all("pub" in u for u in s.connected)            # only the origin

    def test_blocked_initial_url_never_connects(self):
        s = _Session({"http://internal/secret": _Resp(200)})
        with pytest.raises(ValueError):
            _get(s, "http://internal/secret")
        assert s.connected == []

    def test_auto_redirects_disabled_each_hop(self):
        s = _Session({
            "http://a/1": _Resp(302, "http://b/2"),
            "http://b/2": _Resp(200),
        })
        resp = _get(s, "http://a/1", stream=True)
        assert resp.status_code == 200
        assert s.connected == ["http://a/1", "http://b/2"]
        assert all(k.get("allow_redirects") is False for k in s.kwargs)

    def test_max_redirects_enforced(self):
        script = {f"http://h/{i}": _Resp(302, f"http://h/{i + 1}") for i in range(10)}
        s = _Session(script)
        with pytest.raises(ValueError, match="max_redirects"):
            _get(s, "http://h/0", max_redirects=3)
        assert len(s.connected) == 4  # initial + 3 redirects

    def test_relative_redirect_resolved(self):
        s = _Session({
            "http://site.example/a/b": _Resp(302, "/c"),
            "http://site.example/c": _Resp(200),
        })
        _get(s, "http://site.example/a/b")
        assert s.connected[-1] == "http://site.example/c"

    def test_validate_false_opts_out(self):
        s = _Session({"http://internal/ok": _Resp(200)})
        resp = _get(s, "http://internal/ok", validate=False)
        assert resp.status_code == 200

    def test_callable_validate_runs_each_hop(self):
        seen: list[str] = []

        def policy(u):
            seen.append(u)

        s = _Session({
            "http://ok/1": _Resp(302, "http://ok/2"),
            "http://ok/2": _Resp(200),
        })
        _get(s, "http://ok/1", validate=policy)
        assert seen == ["http://ok/1", "http://ok/2"]
