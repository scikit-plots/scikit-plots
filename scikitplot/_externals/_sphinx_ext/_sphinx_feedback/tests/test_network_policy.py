"""Endpoint and origin policies: public, intranet, local and same-origin."""

from __future__ import annotations

import pytest

from _sphinx_ext._sphinx_feedback._config import FeedbackConfigError, validate_endpoint
from _sphinx_ext._sphinx_feedback._network import (
    NetworkPolicyError,
    is_private_host,
    normalize_origin,
    normalize_policy,
    normalize_private_suffixes,
    validate_endpoint_url,
)
from _sphinx_ext._sphinx_feedback._service.app import (
    FeedbackASGIConfigError,
    create_app,
    parse_allowed_origins,
)


@pytest.mark.parametrize(
    "endpoint",
    [
        "/v1/feedback",
        "/api/feedback/",
        "https://feedback.example.org/v1/feedback",
        "http://localhost:8000/v1/feedback",
        "http://127.0.0.2:9000/v1/feedback",
        "https://docs.localhost:8443/v1/feedback",
        "http://[::1]:8000/v1/feedback",
    ],
)
def test_strict_allows_public_https_loopback_and_same_origin(endpoint):
    assert validate_endpoint(endpoint) == endpoint.rstrip("/")


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://feedback.example.org:8443/v1/feedback",
        "http://feedback.internal/v1/feedback",
        "http://10.0.0.5/v1/feedback",
        "//evil.example/v1/feedback",
        "/v1/../feedback",
        "/v1/feedback?x=1",
        "v1/feedback",
        "ftp://example.org/x",
    ],
)
def test_strict_refuses_intranet_http_ports_and_ambiguous_paths(endpoint):
    with pytest.raises(FeedbackConfigError):
        validate_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://feedback.example.org:8443/v1/feedback",
        "http://feedback/v1/feedback",
        "http://feedback.internal:8080/v1/feedback",
        "http://docs.lan.home.arpa/v1/feedback",
        "http://printer.local/v1/feedback",
        "http://10.1.2.3/v1/feedback",
        "http://172.20.0.4/v1/feedback",
        "http://192.168.1.10/v1/feedback",
        "http://100.100.1.1/v1/feedback",
        "http://[fd12:3456::1]/v1/feedback",
    ],
)
def test_private_network_allows_intranet_endpoints(endpoint):
    assert validate_endpoint(endpoint, policy="private-network") == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://feedback.example.org/v1/feedback",
        "http://8.8.8.8/v1/feedback",
        "http://172.32.0.1/v1/feedback",
        "http://feedback.corp.example/v1/feedback",
    ],
)
def test_private_network_still_refuses_public_http(endpoint):
    with pytest.raises(FeedbackConfigError, match="private network"):
        validate_endpoint(endpoint, policy="private-network")


def test_operator_suffixes_extend_the_private_network():
    endpoint = "http://feedback.corp.example/v1/feedback"
    assert (
        validate_endpoint(
            endpoint, policy="private-network", private_suffixes=["corp.example"]
        )
        == endpoint
    )


def test_any_allows_public_http():
    endpoint = "http://feedback.example.org/v1/feedback"
    assert validate_endpoint(endpoint, policy="any") == endpoint


def test_credentials_queries_and_fragments_are_refused_under_every_policy():
    for policy in ("strict", "private-network", "any"):
        for bad in (
            "https://u:p@example.org/x",
            "https://example.org/x?q=1",
            "https://example.org/x#f",
        ):
            with pytest.raises(NetworkPolicyError):
                validate_endpoint_url(bad, policy=policy)


def test_policy_and_suffix_inputs_are_validated():
    assert normalize_policy(None, name="p") == "strict"
    assert normalize_policy(" Private-Network ", name="p") == "private-network"
    with pytest.raises(NetworkPolicyError):
        normalize_policy("open", name="p")
    assert normalize_private_suffixes("corp.example, .LAB.example", name="s") == (
        ".corp.example",
        ".lab.example",
    )
    with pytest.raises(NetworkPolicyError):
        normalize_private_suffixes(["bad_suffix!"], name="s")
    with pytest.raises(FeedbackConfigError):
        validate_endpoint("/v1/feedback", policy="wide-open")


def test_private_host_classification_is_lexical_and_explicit():
    assert is_private_host("feedback")
    assert is_private_host("10.0.0.1")
    assert not is_private_host("172.15.255.255")
    assert not is_private_host("example.org")
    assert not is_private_host("")


def test_origins_follow_the_same_policy():
    assert parse_allowed_origins("http://docs.internal:8080", policy="private-network") == (
        "http://docs.internal:8080",
    )
    assert parse_allowed_origins("https://docs.example.org:8443", policy="private-network") == (
        "https://docs.example.org:8443",
    )
    with pytest.raises(FeedbackASGIConfigError):
        parse_allowed_origins("http://docs.internal:8080")
    with pytest.raises(FeedbackASGIConfigError):
        parse_allowed_origins("http://docs.example.org", policy="private-network")
    assert normalize_origin("HTTP://[FD00::1]:80/", policy="private-network") == "http://[fd00::1]"


def test_create_app_reads_origin_policy_from_the_environment(tmp_path):
    env = {
        "FEEDBACK_REVIEW_MODE": "sqlite",
        "FEEDBACK_SQLITE_PATH": str(tmp_path / "f.sqlite3"),
        "FEEDBACK_ORIGIN_POLICY": "private-network",
        "FEEDBACK_PRIVATE_HOST_SUFFIXES": "corp.example",
        "FEEDBACK_ALLOWED_ORIGINS": "http://docs.corp.example:8000",
    }
    app = create_app(env)
    assert app.allowed_origins == ("http://docs.corp.example:8000",)
    with pytest.raises(FeedbackASGIConfigError):
        create_app({**env, "FEEDBACK_ORIGIN_POLICY": "everything"})
