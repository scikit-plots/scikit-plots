from __future__ import annotations

from types import SimpleNamespace

import pytest

from _sphinx_ext._sphinx_feedback._config import (
    FeedbackConfigError,
    load_aggregate,
    page_enabled,
    resolve_endpoint,
    validate_config,
    validate_endpoint,
)
from _sphinx_ext._sphinx_feedback._service._config import (
    FeedbackServiceConfigError,
    parse_storage_targets,
)


def config(**overrides):
    values = dict(
        feedback_page_enabled=True,
        feedback_position="sidebar",
        feedback_page_main=True,
        feedback_position_fallback="main-bottom",
        feedback_quick_enabled=True,
        feedback_detailed_enabled=True,
        feedback_comment_enabled=True,
        feedback_contributor_enabled=True,
        feedback_counter_enabled=True,
        feedback_counter_source="embedded",
        feedback_endpoint="https://feedback.example.org/v1/feedback",
        feedback_site_id="docs",
        feedback_page_revision="rev-1",
        feedback_include=["**"],
        feedback_exclude=["search", "learn-ai/**"],
        feedback_sidebar_selectors=None,
        feedback_main_selectors=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_https_and_localhost_http_endpoints_allowed():
    assert validate_endpoint("https://example.org/v1/feedback") == "https://example.org/v1/feedback"
    assert validate_endpoint("http://127.0.0.1:8000/v1/feedback") == "http://127.0.0.1:8000/v1/feedback"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/v1/feedback",
        "https://user:pass@example.org/v1/feedback",
        "https://example.org:8443/v1/feedback",
        "https://example.org/v1/feedback#frag",
        "ftp://example.org/x",
        "https://",
    ],
)
def test_unsafe_endpoints_rejected(url):
    with pytest.raises(FeedbackConfigError):
        validate_endpoint(url)


def test_feedback_endpoint_is_independent_from_assistant_profiles():
    cfg = config(feedback_endpoint="")
    cfg.ai_assistant_endpoint_default_profile = "default"
    cfg.ai_assistant_endpoint_profiles = {
        "default": {
            "base": "https://proxy.example.org",
            "feedback": "https://other.invalid/v1/feedback",
        }
    }
    assert resolve_endpoint(cfg) == ""
    with pytest.raises(FeedbackConfigError, match="explicit feedback_endpoint"):
        validate_config(cfg)


@pytest.mark.parametrize("name", [
    "feedback_page_enabled", "feedback_page_main", "feedback_quick_enabled",
    "feedback_detailed_enabled", "feedback_comment_enabled",
    "feedback_contributor_enabled", "feedback_counter_enabled",
])
def test_boolean_config_is_not_truthy_coerced(name):
    with pytest.raises(FeedbackConfigError, match="boolean"):
        validate_config(config(**{name: "false"}))


def test_enabled_interactive_feedback_requires_endpoint():
    with pytest.raises(FeedbackConfigError, match="explicit feedback_endpoint"):
        validate_config(config(feedback_endpoint=""))


def test_enabled_feedback_requires_an_interaction_mode():
    with pytest.raises(FeedbackConfigError, match="quick and/or detailed"):
        validate_config(config(feedback_quick_enabled=False, feedback_detailed_enabled=False))


def test_disabled_feedback_may_have_no_endpoint():
    got = validate_config(config(feedback_page_enabled=False, feedback_endpoint=""))
    assert got["endpoint"] == ""


def test_page_revision_control_character_rejected():
    with pytest.raises(FeedbackConfigError):
        validate_config(config(feedback_page_revision="bad\nrev"))


def test_include_exclude_patterns_are_deterministic():
    cfg = validate_config(config())
    assert page_enabled("guide/install", include=cfg["include"], exclude=cfg["exclude"])
    assert not page_enabled("learn-ai/topic", include=cfg["include"], exclude=cfg["exclude"])


def test_invalid_selector_payload_rejected():
    with pytest.raises(FeedbackConfigError):
        validate_config(config(feedback_sidebar_selectors=["</script>"]))


def write_aggregate(path, *, site_id="docs", pages=None, **extra):
    import json
    payload = {
        "contract": "page.feedback-aggregate.v3",
        "site_id": site_id,
        "pages": pages if pages is not None else {"guide/install": {
            "count": 2, "score": 1,
            "positive_count": 1, "negative_count": 0, "neutral_count": 1,
        }},
    }
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_aggregate_means_unknown_not_zero(tmp_path):
    assert load_aggregate(tmp_path, "", expected_site_id="docs") == {}


def test_site_scoped_aggregate_loads(tmp_path):
    path = tmp_path / "feedback.json"
    write_aggregate(path)
    assert load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs") == {
        "guide/install": {
            "count": 2, "score": 1,
            "positive_count": 1, "negative_count": 0, "neutral_count": 1,
        }
    }


def test_aggregate_site_mismatch_fails_closed(tmp_path):
    path = tmp_path / "feedback.json"
    write_aggregate(path, site_id="other")
    with pytest.raises(FeedbackConfigError, match="site_id"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")


def test_aggregate_path_must_be_root_relative_and_cannot_escape_asset_root(tmp_path):
    for configured in (
        "feedback.json",
        "../outside.json",
        "/../outside.json",
        "//outside.json",
        "/x/../outside.json",
        "/feedback.json?x=1",
        "/feedback.json#x",
        "/x\\feedback.json",
    ):
        with pytest.raises(FeedbackConfigError):
            load_aggregate(tmp_path, configured, expected_site_id="docs")


def test_aggregate_rejects_impossible_score(tmp_path):
    path = tmp_path / "feedback.json"
    write_aggregate(path, pages={"x": {"count": 1, "score": 6, "positive_count": 1, "negative_count": 0, "neutral_count": 0}})
    with pytest.raises(FeedbackConfigError, match="impossible"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")


def test_aggregate_rejects_unknown_top_level_field(tmp_path):
    path = tmp_path / "feedback.json"
    write_aggregate(path, telemetry=True)
    with pytest.raises(FeedbackConfigError, match="unsupported field"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")




def test_aggregate_rejects_duplicate_json_fields(tmp_path):
    path = tmp_path / "feedback.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"pages":{"guide/install":{"count":1,"count":2,"score":1,'
        '"positive_count":1,"negative_count":0,"neutral_count":0}}}',
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match="duplicate field"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")


def test_aggregate_rejects_nonfinite_json_numbers(tmp_path):
    path = tmp_path / "feedback.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"pages":{"guide/install":{"count":1,"score":NaN,'
        '"positive_count":1,"negative_count":0,"neutral_count":0}}}',
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match="non-finite"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")


@pytest.mark.parametrize("path", ["/feedback", "feedback/", "../feedback", "feedback/../x", "feedback//x", "feedback/./x"])
def test_feedback_storage_path_rejects_absolute_trailing_and_traversal_forms(path):
    raw = [{
        "id": "primary", "label": "Primary", "authority": "feedback",
        "provider": "github", "role": "primary", "repo": "org/repo",
        "branch": "main", "paths": {"feedback": path},
        "token_env": ["FEEDBACK_GITHUB_TOKEN"],
    }]
    with pytest.raises(FeedbackServiceConfigError, match="feedback path"):
        parse_storage_targets(raw)


def test_provider_repo_rejects_dot_path_segments():
    for repo in ["../repo", "org/..", "./repo", "org/."]:
        raw = [{
            "id": "primary", "label": "Primary", "authority": "feedback",
            "provider": "github", "role": "primary", "repo": repo,
            "branch": "main", "paths": {"feedback": "feedback"},
            "token_env": ["FEEDBACK_GITHUB_TOKEN"],
        }]
        with pytest.raises(FeedbackServiceConfigError, match="owner/repo"):
            parse_storage_targets(raw)


def test_sqlite_memory_database_is_rejected_because_connections_are_per_operation():
    with pytest.raises(FeedbackServiceConfigError, match=":memory:"):
        parse_storage_targets([{
            "id": "primary", "label": "SQLite", "authority": "feedback",
            "provider": "sqlite", "role": "primary", "database": ":memory:",
        }])


def test_aggregate_pathological_integer_is_wrapped_as_config_error(tmp_path):
    path = tmp_path / "feedback.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"pages":{"guide/install":{"count":1,"score":' + "9" * 5000 + ','
        '"positive_count":1,"negative_count":0,"neutral_count":0}}}',
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match="not valid UTF-8 JSON"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")


def test_public_config_rejects_invalid_unicode_early():
    cfg = config(feedback_page_revision="\ud800")
    with pytest.raises(FeedbackConfigError, match="invalid Unicode"):
        validate_config(cfg)


def test_aggregate_loader_validates_expected_site_id_even_when_file_is_disabled(tmp_path):
    with pytest.raises(FeedbackConfigError, match="site_id"):
        load_aggregate(tmp_path, "", expected_site_id="../bad")


def test_revisioned_aggregate_requires_exact_current_revision(tmp_path):
    path = tmp_path / "feedback.json"
    write_aggregate(
        path,
        pages={"guide/install": {
            "count": 2, "score": 1,
            "positive_count": 1, "negative_count": 0, "neutral_count": 1,
        }},
        page_revision="rev-2",
    )
    assert load_aggregate(
        tmp_path,
        "/feedback.json",
        expected_site_id="docs",
        expected_page_revision="rev-2",
    )["guide/install"]["positive_count"] == 1

    with pytest.raises(FeedbackConfigError, match="does not match"):
        load_aggregate(
            tmp_path,
            "/feedback.json",
            expected_site_id="docs",
            expected_page_revision="rev-3",
        )
    with pytest.raises(FeedbackConfigError, match="requires feedback_page_revision"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")


@pytest.mark.parametrize("contract", ["page.feedback-aggregate.v1", "page.feedback-aggregate.v2"])
def test_retired_aggregate_contracts_are_rejected(tmp_path, contract):
    path = tmp_path / "feedback.json"
    path.write_text(
        '{"contract":"' + contract + '","site_id":"docs","pages":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match="contract must be page.feedback-aggregate.v3"):
        load_aggregate(tmp_path, "/feedback.json", expected_site_id="docs")


def test_distribution_v3_complete_snapshot_metadata_is_explicit(tmp_path):
    path = tmp_path / "feedback-v3-complete.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"complete":true,"pages":{}}\n',
        encoding="utf-8",
    )
    pages, metadata = load_aggregate(
        tmp_path,
        "/feedback-v3-complete.json",
        expected_site_id="docs",
        return_metadata=True,
    )
    assert pages == {}
    assert metadata == {"contract": "page.feedback-aggregate.v3", "complete": True}


def test_distribution_v3_sparse_snapshot_defaults_to_unknown_coverage(tmp_path):
    path = tmp_path / "feedback-v3-sparse.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs","pages":{}}\n',
        encoding="utf-8",
    )
    pages, metadata = load_aggregate(
        tmp_path,
        "/feedback-v3-sparse.json",
        expected_site_id="docs",
        return_metadata=True,
    )
    assert pages == {}
    assert metadata == {"contract": "page.feedback-aggregate.v3", "complete": False}


def test_distribution_v3_rejects_non_boolean_complete_marker(tmp_path):
    path = tmp_path / "feedback-v3-bad-complete.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"complete":"yes","pages":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match="complete must be a boolean"):
        load_aggregate(tmp_path, "/feedback-v3-bad-complete.json", expected_site_id="docs")


def test_distribution_v3_loads_exact_sign_counts(tmp_path):
    path = tmp_path / "feedback-v3.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"pages":{"guide/install":{"count":6,"score":2,'
        '"positive_count":3,"negative_count":2,"neutral_count":1}}}\n',
        encoding="utf-8",
    )
    assert load_aggregate(tmp_path, "/feedback-v3.json", expected_site_id="docs") == {
        "guide/install": {
            "count": 6,
            "score": 2,
            "positive_count": 3,
            "negative_count": 2,
            "neutral_count": 1,
        }
    }


@pytest.mark.parametrize(
    "entry,match",
    [
        ({"count": 3, "score": 0, "positive_count": 1, "negative_count": 1, "neutral_count": 0}, "sum exactly"),
        ({"count": 2, "score": 10, "positive_count": 1, "negative_count": 1, "neutral_count": 0}, "impossible"),
        ({"count": 1, "score": 1, "positive_count": -1, "negative_count": 1, "neutral_count": 1}, "non-negative"),
    ],
)
def test_distribution_v3_rejects_inconsistent_counts(tmp_path, entry, match):
    import json
    path = tmp_path / "bad-v3.json"
    path.write_text(
        json.dumps({
            "contract": "page.feedback-aggregate.v3",
            "site_id": "docs",
            "pages": {"guide/install": entry},
        }),
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match=match):
        load_aggregate(tmp_path, "/bad-v3.json", expected_site_id="docs")


def test_distribution_v3_can_be_revision_pinned(tmp_path):
    path = tmp_path / "feedback-v3.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"page_revision":"rev-2","pages":{"guide/install":{"count":2,"score":0,'
        '"positive_count":1,"negative_count":1,"neutral_count":0}}}\n',
        encoding="utf-8",
    )
    assert load_aggregate(
        tmp_path, "/feedback-v3.json", expected_site_id="docs", expected_page_revision="rev-2"
    )["guide/install"]["positive_count"] == 1


def test_distribution_v3_rejects_javascript_unsafe_numbers(tmp_path):
    import json
    path = tmp_path / "unsafe-v3.json"
    path.write_text(
        json.dumps({
            "contract": "page.feedback-aggregate.v3",
            "site_id": "docs",
            "pages": {
                "guide/install": {
                    "count": 9007199254740992,
                    "score": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 9007199254740992,
                }
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match="JavaScript-safe"):
        load_aggregate(tmp_path, "/unsafe-v3.json", expected_site_id="docs")


def test_distribution_v3_rejects_present_but_empty_page_revision(tmp_path):
    path = tmp_path / "empty-revision-v3.json"
    path.write_text(
        '{"contract":"page.feedback-aggregate.v3","site_id":"docs",'
        '"page_revision":"","pages":{"guide/install":{"count":1,"score":1,'
        '"positive_count":1,"negative_count":0,"neutral_count":0}}}\n',
        encoding="utf-8",
    )
    with pytest.raises(FeedbackConfigError, match="non-empty and canonical"):
        load_aggregate(tmp_path, "/empty-revision-v3.json", expected_site_id="docs")
