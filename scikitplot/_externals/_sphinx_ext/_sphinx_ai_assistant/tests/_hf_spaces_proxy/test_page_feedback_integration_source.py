from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve()
PROXY_APP = HERE.parents[2] / "_hf_spaces_proxy" / "app.py"
DOCS_SOURCE = HERE.parents[4]
CONF = DOCS_SOURCE / "conf.py"
GENERIC_FEEDBACK_AGGREGATE = (
    DOCS_SOURCE
    / "_sphinx_ext"
    / "_sphinx_feedback"
    / "_static"
    / "page-feedback-aggregate.json"
)
LEGACY_GENERIC_FEEDBACK_AGGREGATE = DOCS_SOURCE / "_feedback" / "page-feedback-aggregate.json"


PROXY_TEXT = PROXY_APP.read_text(encoding="utf-8")
CONF_TEXT = CONF.read_text(encoding="utf-8")
GENERIC_FEEDBACK_AGGREGATE_TEXT = GENERIC_FEEDBACK_AGGREGATE.read_text(encoding="utf-8")


def test_proxy_generic_feedback_site_authority_matches_sphinx_public_site_id():
    assert 'feedback_site_id = "scikit-plots-learn"' in CONF_TEXT
    assert '"FEEDBACK_ALLOWED_SITE_IDS",\n    "scikit-plots-learn",' in PROXY_TEXT



def test_scikitplots_generic_feedback_counter_uses_complete_v3_reviewed_snapshot():
    assert 'feedback_counter_enabled = True' in CONF_TEXT
    assert 'feedback_counter_source = "embedded"' in CONF_TEXT
    assert 'feedback_aggregate_file = "/page-feedback-aggregate.json"' in CONF_TEXT
    assert GENERIC_FEEDBACK_AGGREGATE.is_file()
    assert not LEGACY_GENERIC_FEEDBACK_AGGREGATE.exists()
    # Reviewed event PRs still use docs/source/_feedback/pages/...; moving the
    # build snapshot must not redirect submission/review storage into _static.
    assert '"FEEDBACK_GITHUB_PATH",\n    "docs/source/_feedback",' in PROXY_TEXT

    import json

    payload = json.loads(GENERIC_FEEDBACK_AGGREGATE_TEXT)
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
