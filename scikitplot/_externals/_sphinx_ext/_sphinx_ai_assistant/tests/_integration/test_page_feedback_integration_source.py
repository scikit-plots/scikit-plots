"""
Page feedback: the proxy, the site configuration and the shipped snapshot agree.

Notes
-----
The proxy and the shipped aggregate travel with the extension stack, so the
tests of those two run in every checkout. The site's ``conf.py`` exists only
in a documentation checkout; the tests that read it skip elsewhere and say
why. Paths come from :mod:`tests._paths`, which finds the documentation source
by what is on disk. An earlier version counted parent directories, pointed at
a ``conf.py`` that exists in neither checkout, and failed at collection.
"""
from __future__ import annotations

import json

import pytest

from .._paths import DOCS_SOURCE_ROOT, RUNTIME_ROOT, STACK_ROOT

PROXY_APP = RUNTIME_ROOT / "_hf_spaces_proxy" / "app.py"
GENERIC_FEEDBACK_AGGREGATE = (
    STACK_ROOT / "_sphinx_feedback" / "_static" / "page-feedback-aggregate.json"
)

PROXY_TEXT = PROXY_APP.read_text(encoding="utf-8")


def _site_conf_text() -> str:
    """Return the site's ``conf.py``, or skip where there is no site."""
    if DOCS_SOURCE_ROOT is None:
        pytest.skip(
            "no documentation source owns this stack in this checkout; the "
            "site configuration is tested in the documentation checkout"
        )
    return (DOCS_SOURCE_ROOT / "conf.py").read_text(encoding="utf-8")


def test_proxy_generic_feedback_site_authority_is_the_public_site_id():
    assert '"FEEDBACK_ALLOWED_SITE_IDS",\n    "scikit-plots-learn",' in PROXY_TEXT


def test_site_feedback_site_id_matches_the_proxy_authority():
    assert 'feedback_site_id = "scikit-plots-learn"' in _site_conf_text()



def test_site_feedback_counter_reads_the_embedded_snapshot():
    conf_text = _site_conf_text()
    assert 'feedback_counter_enabled = True' in conf_text
    assert 'feedback_counter_source = "embedded"' in conf_text
    assert 'feedback_aggregate_file = "/page-feedback-aggregate.json"' in conf_text
    assert not (DOCS_SOURCE_ROOT / "_feedback" / "page-feedback-aggregate.json").exists()


def test_generic_feedback_snapshot_is_the_complete_v3_reviewed_one():
    assert GENERIC_FEEDBACK_AGGREGATE.is_file()
    # Reviewed event PRs still use docs/source/_feedback/pages/...; moving the
    # build snapshot must not redirect submission/review storage into _static.
    assert '"FEEDBACK_GITHUB_PATH",\n    "docs/source/_feedback",' in PROXY_TEXT

    payload = json.loads(GENERIC_FEEDBACK_AGGREGATE.read_text(encoding="utf-8"))
    assert payload == {
        "complete": True,
        "contract": "page.feedback-aggregate.v3",
        "pages": {},
        "site_id": "scikit-plots-learn",
    }

def test_proxy_validates_generic_feedback_authority_before_rate_limit_admission():
    start = PROXY_TEXT.index("async def _page_feedback_dispatch")
    end = PROXY_TEXT.index('@app.post("/v1/feedback")', start)
    body = PROXY_TEXT[start:end]
    assert body.index("PAGE_FEEDBACK_SERVICE.validate_request") < body.index(
        "_consume_page_feedback_rate"
    )


def test_proxy_feedback_route_accepts_only_current_generic_contract():
    start = PROXY_TEXT.index('@app.post("/v1/feedback")')
    body = PROXY_TEXT[start:]
    assert 'return await _page_feedback_dispatch(request, raw)' in body
    assert 'telemetryConsent' not in body
    assert 'FEEDBACK_PERSIST_ENABLED' not in body
    assert 'normalize_feedback_record' not in body

    dispatch_start = PROXY_TEXT.index("async def _page_feedback_dispatch")
    dispatch_end = PROXY_TEXT.index('@app.post("/v1/feedback")', dispatch_start)
    dispatch = PROXY_TEXT[dispatch_start:dispatch_end]
    assert 'decode_page_feedback_request(raw)' in dispatch
    assert 'PAGE_FEEDBACK_SERVICE.validate_request' in dispatch
    assert 'page_feedback_request_hash(validated_payload)' in dispatch


def test_proxy_has_no_legacy_assistant_feedback_telemetry_fallback():
    assert 'Explicit feedback telemetry permission is required.' not in PROXY_TEXT
    assert 'FEEDBACK_TELEMETRY_CONSENT_VERSION' not in PROXY_TEXT
    assert 'FEEDBACK_TELEMETRY_SCHEMA_VERSION' not in PROXY_TEXT
    assert 'FEEDBACK_PERSIST_ENABLED' not in PROXY_TEXT
    assert 'normalize_feedback_record' not in PROXY_TEXT
