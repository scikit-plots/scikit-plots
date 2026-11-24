"""The ``fake_dns`` fixture answers like a resolver, and the URL check still fails closed."""

from __future__ import annotations

import socket

import pytest

from .._url_handler import _validate_url_security


def test_unlisted_names_resolve_to_a_public_address(fake_dns):
    _validate_url_security("https://example.org/page")
    assert fake_dns.calls == ["example.org"]


def test_ipv6_and_ip_literals(fake_dns):
    fake_dns.hosts["v6.example"] = ["2606:2800:220:1:248:1893:25c8:1946"]
    family, *_ = fake_dns("v6.example")[0]
    assert family == socket.AF_INET6
    assert fake_dns("93.184.216.34")[0][4] == ("93.184.216.34", 0)
    _validate_url_security("https://v6.example/")


@pytest.mark.parametrize("address", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "fd00::1"])
def test_a_name_resolving_to_a_private_address_is_still_refused(fake_dns, address):
    fake_dns.hosts["internal.example"] = [address]
    with pytest.raises(ValueError):
        _validate_url_security("https://internal.example/")


def test_resolution_failure_fails_closed(fake_dns):
    fake_dns.hosts["missing.example"] = None
    with pytest.raises(ValueError, match="resolve"):
        _validate_url_security("https://missing.example/")


def test_a_test_without_fake_dns_never_reaches_the_real_resolver():
    # The autouse guard answers like an unreachable resolver: fail closed.
    with pytest.raises(ValueError, match="resolve"):
        _validate_url_security("https://example.org/")
