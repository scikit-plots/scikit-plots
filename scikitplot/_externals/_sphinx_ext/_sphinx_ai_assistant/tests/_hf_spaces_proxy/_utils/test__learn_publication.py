from __future__ import annotations

import json

import pytest

from _sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._learn_publication import (
    LearnPublicationTransportError,
    build_publication_policy,
    capability_document,
    parse_publication_request,
    publication_request_id,
    workflow_dispatch_body,
)


def policy(mode="stub"):
    return build_publication_policy(
        mode=mode,
        repository="scikit-plots/learn",
        default_branch="main",
        canonical_prefix="docs/source/learn-ai",
        workflow="ai-learn-publish.yml",
        max_request_bytes=49_152,
    )


def overview_request():
    draft = {
        "contract": "learn.page-overview-draft.v1",
        "body": "Reviewed overview.",
        "model": "stub/test",
        "generated_at": "2026-09-25T00:00:00.000Z",
        "base_revision": "tree-abc",
        "profile": {},
        "contexts": [],
        "source_ids": [],
        "guidance": "",
    }
    return {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "draft": draft,
        "base_revision": "tree-abc",
        "subject_id": "topic-example",
    }


def test_publication_policy_is_fixed_server_side_and_capability_has_no_secret():
    p = policy("github")
    cap = capability_document(p, credential_ready=True)
    assert cap["repository"] == "scikit-plots/learn"
    assert cap["canonical_prefix"] == "docs/source/learn-ai"
    assert cap["browser_repository_override"] is False
    assert cap["browser_credentials"] is False
    assert "token" not in json.dumps(cap).lower()


def test_publication_request_rejects_repo_branch_path_secret_and_bad_json():
    p = policy()
    request = overview_request()
    for key, value in {
        "repository": "attacker/repo",
        "branch": "other",
        "path": "../../x",
        "token": "secret",
        "workflow": "evil.yml",
    }.items():
        candidate = dict(request, **{key: value})
        with pytest.raises(LearnPublicationTransportError, match="unexpected fields"):
            parse_publication_request(json.dumps(candidate).encode(), p)
    with pytest.raises(LearnPublicationTransportError, match="duplicate JSON key"):
        parse_publication_request(b'{"contract":"learn.publication-request.v1","contract":"x","action":"test"}', p)
    with pytest.raises(LearnPublicationTransportError, match="non-finite"):
        parse_publication_request(b'{"contract":"learn.publication-request.v1","action":"publish","draft":{"contract":"learn.page-overview-draft.v1","body":NaN},"base_revision":"tree-abc","subject_id":"topic-example"}', p)


def test_workflow_dispatch_is_deterministic_bounded_and_contains_no_destination_override():
    p = policy("github")
    parsed = parse_publication_request(json.dumps(overview_request()).encode(), p)
    first_id, first = workflow_dispatch_body(p, parsed)
    second_id, second = workflow_dispatch_body(p, parsed)
    assert first_id == second_id == publication_request_id(parsed)
    assert first == second
    assert first["ref"] == "main"
    assert first["inputs"]["operation"] == "publish"
    assert first["inputs"]["request_id"] == first_id
    assert len(first["inputs"]["request_json"]) < 60_000
    transported = json.loads(first["inputs"]["request_json"])
    assert "repository" not in transported
    assert "workflow" not in transported
    assert "token" not in transported


def test_test_action_is_non_mutating_minimal_envelope():
    p = policy("github")
    assert parse_publication_request(
        b'{"contract":"learn.publication-request.v1","action":"test"}', p
    ) == {"contract": "learn.publication-request.v1", "action": "test"}
    with pytest.raises(LearnPublicationTransportError, match="unexpected fields"):
        parse_publication_request(
            b'{"contract":"learn.publication-request.v1","action":"test","repository":"x/y"}', p
        )


def test_record_transport_requires_stable_created_at_field():
    p = policy()
    request = overview_request()
    request["draft"] = {
        "contract": "learn.record-creation-draft.v1",
        "kind": "topic",
        "title": "T",
        "summary": "S",
        "domains": [],
        "sections": [],
        "evidence_gaps": [],
        "related_questions": [],
        "provenance": {"base_revision": "tree-abc"},
    }
    request.pop("subject_id")
    with pytest.raises(LearnPublicationTransportError, match="created_at"):
        parse_publication_request(json.dumps(request).encode(), p)


def test_publication_transport_preserves_bounded_public_contributor_credit():
    p = policy()
    request = overview_request()
    request["contributor"] = {"display_name": "DataFox"}
    parsed = parse_publication_request(json.dumps(request).encode(), p)
    assert parsed["contributor"] == {"display_name": "DataFox"}

    request["contributor"] = {"display_name": "Ada", "email": "private@example.org"}
    with pytest.raises(LearnPublicationTransportError, match=r"expected \{display_name\}"):
        parse_publication_request(json.dumps(request).encode(), p)

    request["contributor"] = {"display_name": "Ada\nInjected"}
    with pytest.raises(LearnPublicationTransportError, match="control characters"):
        parse_publication_request(json.dumps(request).encode(), p)


def test_feedback_transport_accepts_quick_and_full_scale_without_repository_authority():
    p = policy("github")
    base = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": "tree-0123456789abcdef",
        "subject_id": "topic-example",
        "section_id": "summary",
        "generation_id": "generation-0123456789abcdef",
        "feedback_id": "feedback-0123456789abcdef0123456789abcdef",
        "rating": 1,
        "feedback_mode": "quick",
        "contributor": {"display_name": ""},
    }
    parsed = parse_publication_request(json.dumps(base).encode(), p)
    assert parsed["action"] == "feedback"
    assert parsed["rating"] == 1
    assert parsed["feedback_mode"] == "quick"
    assert "created_at" not in parsed
    assert parsed["contributor"] == {"display_name": "Anonymous"}
    _, dispatch = workflow_dispatch_body(p, parsed)
    assert dispatch["inputs"]["operation"] == "publish"
    transported = json.loads(dispatch["inputs"]["request_json"])
    assert transported["action"] == "feedback"
    assert "repository" not in transported
    assert "token" not in transported

    for rating in range(-5, 6):
        candidate = dict(
            base,
            rating=rating,
            feedback_id=f"feedback-{rating + 5:032x}",
            feedback_mode="detailed",
        )
        assert parse_publication_request(json.dumps(candidate).encode(), p)["rating"] == rating

    invalid_quick = dict(base, rating=5, feedback_id="feedback-ffffffffffffffffffffffffffffffff")
    with pytest.raises(LearnPublicationTransportError, match="quick feedback"):
        parse_publication_request(json.dumps(invalid_quick).encode(), p)

    legacy = dict(base, created_at="2026-09-27T03:30:00Z")
    assert parse_publication_request(json.dumps(legacy).encode(), p)["created_at"] == legacy["created_at"]
    for bad_created_at in (
        "private@example.org",
        "2026-09-31T03:30:00Z",
        "2026-09-27T03:30:00+00:00",
    ):
        with pytest.raises(LearnPublicationTransportError, match="created_at"):
            parse_publication_request(
                json.dumps(dict(base, created_at=bad_created_at)).encode(), p
            )
    for bad_revision in (
        "tree-abc",
        "private@example.org",
        "tree-0123456789ABCDEf",
        "tree-0123456789abcdef00",
    ):
        with pytest.raises(LearnPublicationTransportError, match="base_revision"):
            parse_publication_request(
                json.dumps(dict(base, base_revision=bad_revision)).encode(), p
            )
    bad_mode = dict(base, feedback_mode="instant")
    with pytest.raises(LearnPublicationTransportError, match="feedback_mode"):
        parse_publication_request(json.dumps(bad_mode).encode(), p)

    for rating in (-6, 6, True, 1.5):
        candidate = dict(base, rating=rating)
        with pytest.raises(LearnPublicationTransportError, match="rating"):
            parse_publication_request(json.dumps(candidate).encode(), p)

    blank_comment = dict(base, comment="  \n  ")
    assert "comment" not in parse_publication_request(json.dumps(blank_comment).encode(), p)
    for bad_comment in ("bad\x00detail", "bad\x7fdetail"):
        candidate = dict(base, comment=bad_comment)
        with pytest.raises(LearnPublicationTransportError, match="comment: invalid text"):
            parse_publication_request(json.dumps(candidate).encode(), p)


def test_feedback_transport_requires_opaque_random_shaped_event_ids():
    p = policy("github")
    base = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": "tree-0123456789abcdef",
        "subject_id": "topic-example",
        "section_id": "summary",
        "generation_id": "generation-0123456789abcdef",
        "rating": 1,
        "feedback_mode": "quick",
        "contributor": {"display_name": ""},
    }
    current = dict(base, feedback_id="feedback-" + "ab" * 24)
    parsed = parse_publication_request(json.dumps(current).encode(), p)
    assert parsed["feedback_id"] == current["feedback_id"]
    assert len(parsed["feedback_id"]) == 57

    for bad in (
        "feedback-user-123",
        "feedback-0123456789abcdef",
        "feedback-" + "ab" * 20,
        "feedback-" + "ab" * 25,
    ):
        with pytest.raises(LearnPublicationTransportError, match="feedback_id"):
            parse_publication_request(
                json.dumps(dict(base, feedback_id=bad)).encode(), p
            )

    other = dict(current, feedback_id="feedback-" + "cd" * 24)
    first = parse_publication_request(json.dumps(current).encode(), p)
    second = parse_publication_request(json.dumps(other).encode(), p)
    assert publication_request_id(first) != publication_request_id(second)
    assert len(publication_request_id(first)) == 64
    assert all(key not in first for key in (
        "ip", "client_ip", "user_agent", "browser", "device", "timezone",
        "language", "session_id", "account_id", "telemetry",
    ))
