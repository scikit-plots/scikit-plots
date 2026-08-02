# corpus/tests/test__url_handler_ssrf.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Fail-closed SSRF resolution gate (CORPUS-NET-002)
=================================================

``_resolve_and_validate`` / ``_validate_url_security`` must (a) validate **every**
resolved A/AAAA record, and (b) **fail closed** — a host that cannot be resolved
is denied, not allowed. Previously a resolution error returned "not private"
(fail-open), silently bypassing the SSRF filter.

DNS is faked via a controllable ``getaddrinfo`` so the suite is hermetic.

Run with::

    pytest scikitplot/corpus/tests/test__url_handler_ssrf.py -v
"""

from __future__ import annotations

import socket

import pytest

from scikitplot.corpus import _url_handler as U


def _resolver(monkeypatch, mapping):
    """mapping: host -> list[ip], or 'FAIL' to raise gaierror."""
    def fake(host, *a, **k):
        val = mapping.get(host)
        if val == "FAIL" or val is None:
            raise socket.gaierror(-2, "Name or service not known")
        return [(2, 1, 6, "", (ip, 0)) for ip in val]
    monkeypatch.setattr(U.socket, "getaddrinfo", fake)


class TestFailClosed:
    def test_dns_error_denies(self, monkeypatch):
        _resolver(monkeypatch, {"broken.example": "FAIL"})
        with pytest.raises(ValueError, match="fail-closed"):
            U._resolve_and_validate("broken.example")

    def test_validate_url_fails_closed(self, monkeypatch):
        _resolver(monkeypatch, {"broken.example": "FAIL"})
        with pytest.raises(ValueError):
            U._validate_url_security("http://broken.example/x")

    def test_empty_resolution_denies(self, monkeypatch):
        _resolver(monkeypatch, {"empty.example": []})
        with pytest.raises(ValueError, match="no addresses"):
            U._resolve_and_validate("empty.example")

    def test_allow_private_still_fails_closed(self, monkeypatch):
        _resolver(monkeypatch, {"h": "FAIL"})
        with pytest.raises(ValueError):
            U._resolve_and_validate("h", allow_private=True)


class TestAllRecords:
    def test_public_allowed_returns_ips(self, monkeypatch):
        _resolver(monkeypatch, {"pub.example": ["93.184.216.34"]})
        assert U._resolve_and_validate("pub.example") == ("93.184.216.34",)

    def test_any_private_record_blocks(self, monkeypatch):
        # public + private in the same answer -> the private one must block
        _resolver(monkeypatch, {"rebind.example": ["93.184.216.34", "10.0.0.5"]})
        with pytest.raises(ValueError, match="10.0.0.5"):
            U._resolve_and_validate("rebind.example")

    def test_ipv6_local_record_blocks(self, monkeypatch):
        _resolver(monkeypatch, {"v6.example": ["2606:2800:220:1:248:1893:25c8:1946", "::1"]})
        with pytest.raises(ValueError):
            U._resolve_and_validate("v6.example")

    def test_allow_private_permits(self, monkeypatch):
        _resolver(monkeypatch, {"internal.host": ["10.1.2.3"]})
        assert U._resolve_and_validate("internal.host", allow_private=True) == ("10.1.2.3",)


class TestIpLiterals:
    def test_public_literal_no_dns(self):
        assert U._resolve_and_validate("8.8.8.8") == ("8.8.8.8",)

    @pytest.mark.parametrize("lit", [
        "127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "0.0.0.0", "224.0.0.1",
    ])
    def test_blocked_literals(self, lit):
        with pytest.raises(ValueError):
            U._resolve_and_validate(lit)


class TestSchemeHostname:
    @pytest.mark.parametrize("url", ["ftp://x/y", "file:///etc/passwd"])
    def test_rejects_non_http(self, url):
        with pytest.raises(ValueError):
            U._validate_url_security(url)

    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError):
            U._validate_url_security("http:///nohost")
