"""Tests for composable Corpus policy bundles."""

from __future__ import annotations

import pytest

from scikitplot.corpus import (
    BackendPolicy,
    CorpusPolicyBundle,
    DownloadPolicy,
    ErrorPolicy,
    RuntimePolicy,
    corpus_policy_bundle,
)


def test_default_bundle_matches_existing_defaults() -> None:
    bundle = CorpusPolicyBundle.default()
    assert bundle.runtime == RuntimePolicy.offline()
    assert bundle.backend == BackendPolicy.resilient()
    assert bundle.download == DownloadPolicy.secure()
    assert bundle.errors is ErrorPolicy.COLLECT


def test_safe_local_composes_without_collapsing_policy_dimensions() -> None:
    bundle = CorpusPolicyBundle.safe_local()
    assert bundle.runtime.allow_network is False
    assert bundle.backend.allow_network is False
    assert bundle.backend.allow_download is False
    assert bundle.download.verify_ssl is True
    assert bundle.download.block_private_ips is True
    assert bundle.errors is ErrorPolicy.COLLECT


def test_strict_local_raises_and_stays_offline() -> None:
    bundle = CorpusPolicyBundle.strict_local()
    assert bundle.runtime.allow_network is False
    assert bundle.backend.allow_network is False
    assert bundle.backend.allow_download is False
    assert bundle.backend.on_exhausted is ErrorPolicy.RAISE
    assert bundle.errors is ErrorPolicy.RAISE


def test_bundle_round_trip_and_nested_validation() -> None:
    original = CorpusPolicyBundle.from_config({
        "preset": "safe-local",
        "name": "team-local",
        "backend": {
            "preset": "offline",
            "order": ["company-asr", "faster-whisper"],
            "include_unlisted": False,
        },
        "download": {"preset": "constrained", "max_retries": 0},
        "errors": "collect",
    })
    restored = CorpusPolicyBundle.from_config(original.to_dict())
    assert restored == original


@pytest.mark.parametrize(
    "config,match",
    [
        ({"typo": True}, "unknown CorpusPolicyBundle"),
        ({"runtime": {"allow_network": "false"}}, "allow_network must be bool"),
        ({"backend": {"allow_download": "false"}}, "allow_download must be bool"),
        ({"download": {"verify_ssl": "false"}}, "verify_ssl must be bool"),
    ],
)
def test_bundle_rejects_ambiguous_or_truthy_security_config(config, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        CorpusPolicyBundle.from_config(config)


def test_bundle_explain_is_json_shaped_and_explicit() -> None:
    data = CorpusPolicyBundle.docs_ci().explain()
    assert data["derived"]["url_sources_allowed"] is False
    assert data["derived"]["backend_download_allowed"] is False
    assert data["derived"]["tls_verification"] is True
    assert data["errors"] == "collect"


def test_bundle_helpers_keep_ownership_explicit() -> None:
    bundle = CorpusPolicyBundle.safe_local()
    reader_kwargs = bundle.reader_kwargs(model_size="tiny")
    assert reader_kwargs["backend_policy"] == bundle.backend
    assert reader_kwargs["model_size"] == "tiny"

    cfg = bundle.builder_config(chunker="paragraph")
    assert cfg.download_policy == bundle.download
    assert cfg.chunker == "paragraph"

    with pytest.raises(ValueError, match="backend_policy"):
        bundle.reader_kwargs(backend_policy="strict")
    with pytest.raises(ValueError, match="download_policy"):
        bundle.builder_config(download_policy="secure")


def test_bundle_resolver_aliases() -> None:
    assert corpus_policy_bundle("offline") == CorpusPolicyBundle.safe_local()
    assert corpus_policy_bundle("ci") == CorpusPolicyBundle.docs_ci()
    assert corpus_policy_bundle(None) == CorpusPolicyBundle.default()
