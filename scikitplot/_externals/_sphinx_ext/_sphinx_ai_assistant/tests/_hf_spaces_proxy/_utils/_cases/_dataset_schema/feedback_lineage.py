from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi import HTTPException

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app as proxy_app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _dataset_schema as schema

# deduplicate_dataset is deployed as a sibling script and intentionally supports
# direct execution, so import it through its package-local module path.
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import deduplicate_dataset as dd


def _eligible(
    fid: str,
    *,
    edit: int,
    chain: str,
    prev: str | None = None,
    prevs: list[str] | None = None,
    source: str = "feedback",
    ts: int = 1,
    rating: int = 1,
    key: str | None = None,
) -> dict:
    return {
        "schemaVersion": 5,
        "_source": source,
        "_ts": ts,
        "_dedup_key": key or f"receipt-{fid}-{source}",
        "conversationId": None,
        "feedbackId": fid,
        "feedbackChainId": chain,
        "recordType": "qa",
        "answerIndex": 0,
        "action": "review" if source == "feedback" else "rate",
        "prevFeedbackId": prev,
        "prevFeedbackIds": list(prevs or []),
        "editCount": edit,
        "status": "active",
        "trainingStatus": "eligible",
        "ratingValue": rating,
        "ratingSlug": "helpful" if rating > 0 else "not_helpful",
        "ratingTitle": "Helpful" if rating > 0 else "Not helpful",
        "ratingMode": "quick",
        "query": "Q",
        "answer": "A",
        "messages": None,
        "model": None,
        "page": "",
        "ts": ts,
    }


def test_schema_v5_adds_bounded_feedback_lineage_columns() -> None:
    assert schema.SCHEMA_VERSION == 5
    assert "feedbackChainId" in schema.CANONICAL_COLUMNS
    assert "prevFeedbackIds" in schema.CANONICAL_COLUMNS
    assert schema.MAX_FEEDBACK_LINEAGE_IDS == 1000


def test_feedback_review_normalizer_preserves_complete_lineage() -> None:
    payload = {
        "feedbackId": "f3",
        "feedbackChainId": "f1",
        "prevFeedbackId": "f2",
        "prevFeedbackIds": ["f1", "f2"],
        "editCount": 2,
        "answerIndex": 0,
        "ratingValue": 1,
        "ratingLabel": "helpful",
        "ratingTitle": "Helpful",
        "ratingMode": "quick",
        "ratingScaleMin": -1,
        "ratingScaleMax": 1,
        "message": "",
        "query": "Q",
        "answer": "A",
        "model": {"id": "m", "provider": "p", "model": "owner/m"},
        "page": "https://docs.example/page",
        "consentVersion": "2.0.0",
        "trainingConsentVersion": "1.0.0",
        "ts": 3,
    }
    row = schema.normalize_feedback_review_record(
        payload, server_ts_ms=4, receipt_id="receipt"
    )
    assert row["schemaVersion"] == 5
    assert row["feedbackId"] == "f3"
    assert row["feedbackChainId"] == "f1"
    assert row["prevFeedbackId"] == "f2"
    assert row["prevFeedbackIds"] == ["f1", "f2"]
    assert row["editCount"] == 2


def test_feedback_retraction_targets_canonical_parent_key_and_keeps_ancestry() -> None:
    row = schema.normalize_feedback_record(
        {
            "schemaVersion": 4,
            "action": "retract",
            "feedbackChainId": "f1",
            "prevFeedbackId": "f3",
            "prevFeedbackIds": ["f1", "f2", "f3"],
            "editCount": 3,
            "answerIndex": 0,
            "ts": 9,
        },
        server_ts_ms=10,
    )
    assert row["_dedup_key"] == "f3:feedback"
    assert row["feedbackChainId"] == "f1"
    assert row["prevFeedbackIds"] == ["f1", "f2", "f3"]
    assert row["action"] == "retract"


def test_contribution_qa_and_conversation_preserve_same_lineage() -> None:
    lineage = {
        "feedbackId": "f3",
        "feedbackChainId": "f1",
        "prevFeedbackId": "f2",
        "prevFeedbackIds": ["f1", "f2"],
        "editCount": 2,
    }
    envelope = {
        "schemaVersion": 4,
        "page": "",
        "consentVersion": "2.0.0",
        "model": None,
    }
    qa = schema.normalize_contribution_record(
        {
            "recordType": "qa",
            "answerIndex": 0,
            "query": "Q",
            "answer": "A",
            "ratingValue": 1,
            "ratingLabel": "helpful",
            "ratingMode": "quick",
            **lineage,
        },
        envelope=envelope,
        server_ts_ms=10,
        submission_id="r1",
    )
    assert {k: qa[k] for k in lineage} == lineage

    conv = schema.normalize_contribution_record(
        {
            "recordType": "conversation",
            "message": "",
            "messages": [
                {"role": "user", "content": "Q", "ts": 1},
                {
                    "role": "assistant",
                    "content": "A",
                    "ts": 2,
                    "feedback": {
                        "ratingValue": 1,
                        "ratingLabel": "helpful",
                        "ratingMode": "quick",
                        "note": "",
                        **lineage,
                    },
                },
            ],
        },
        envelope=envelope,
        server_ts_ms=10,
        submission_id="r2",
    )
    nested = conv["messages"][1]["feedback"]
    for key, value in lineage.items():
        assert nested[key] == value


def test_terminal_revision_wins_before_source_priority() -> None:
    rows = [
        _eligible("f1", edit=0, chain="f1", source="contribution", rating=-1),
        _eligible("f2", edit=1, chain="f1", prev="f1", prevs=["f1"], source="feedback", rating=1),
    ]
    clean = dd.deduplicate(rows)
    assert [r["feedbackId"] for r in clean] == ["f2"]
    assert clean[0]["_source"] == "feedback"
    assert dd._LAST_LINEAGE_STATS.superseded_records_removed == 1


def test_same_terminal_event_uses_source_priority_after_lineage_resolution() -> None:
    feedback = _eligible("f3", edit=2, chain="f1", prev="f2", prevs=["f1", "f2"], source="feedback", ts=50)
    contribution = deepcopy(feedback)
    contribution.update({"_source": "contribution", "action": "rate", "_dedup_key": "other", "_ts": 10})
    clean = dd.deduplicate([feedback, contribution])
    assert len(clean) == 1
    assert clean[0]["_source"] == "contribution"


def test_same_feedback_id_with_conflicting_ancestry_fails_closed() -> None:
    left = _eligible(
        "f2", edit=1, chain="f1", prev="f1", prevs=["f1"],
        source="feedback", key="review-a:feedback"
    )
    right = _eligible(
        "f2", edit=1, chain="other-root", prev="other-root",
        prevs=["other-root"], source="contribution", key="receipt:0"
    )
    assert dd.deduplicate([left, right]) == []


def test_same_revision_fork_fails_closed_instead_of_using_timestamp() -> None:
    clean = dd.deduplicate(
        [
            _eligible("f1", edit=0, chain="f1"),
            _eligible("f2a", edit=1, chain="f1", prev="f1", prevs=["f1"], ts=10),
            _eligible("f2b", edit=1, chain="f1", prev="f1", prevs=["f1"], ts=99),
        ]
    )
    assert clean == []
    assert dd._LAST_LINEAGE_STATS.forked_chains_excluded == 1


def test_deeper_descendant_does_not_hide_an_earlier_fork() -> None:
    clean = dd.deduplicate(
        [
            _eligible("f1", edit=0, chain="f1"),
            _eligible("f2a", edit=1, chain="f1", prev="f1", prevs=["f1"], ts=10),
            _eligible("f2b", edit=1, chain="f1", prev="f1", prevs=["f1"], ts=11),
            _eligible("f3", edit=2, chain="f1", prev="f2b", prevs=["f1", "f2b"], ts=12),
        ]
    )
    assert clean == []
    assert dd._LAST_LINEAGE_STATS.forked_chains_excluded == 1


def test_sparse_self_contained_terminal_chain_remains_valid() -> None:
    # Missing intermediate rows are expected in provider-review snapshots; the
    # terminal v5 row carries enough ancestry to remain independently valid.
    clean = dd.deduplicate(
        [
            _eligible("f1", edit=0, chain="f1"),
            _eligible("f3", edit=2, chain="f1", prev="f2", prevs=["f1", "f2"], ts=12),
        ]
    )
    assert [row["feedbackId"] for row in clean] == ["f3"]


def test_explicit_cycle_poisoning_excludes_the_claimed_chain() -> None:
    clean = dd.deduplicate(
        [
            _eligible("f1", edit=0, chain="f1"),
            _eligible("f2", edit=2, chain="f1", prev="f2", prevs=["f1", "f2"]),
        ]
    )
    assert clean == []
    assert dd._LAST_LINEAGE_STATS.malformed_records_excluded >= 1


def test_legacy_scalar_parent_chain_still_resolves_when_rows_are_available() -> None:
    first = schema.normalize_record(
        {
            "schemaVersion": 4,
            "_source": "feedback",
            "_ts": 1,
            "_dedup_key": "a",
            "feedbackId": "f1",
            "recordType": "qa",
            "action": "review",
            "editCount": 0,
            "trainingStatus": "eligible",
            "ratingValue": -1,
        }
    )
    second = schema.normalize_record(
        {
            "schemaVersion": 4,
            "_source": "feedback",
            "_ts": 2,
            "_dedup_key": "b",
            "feedbackId": "f2",
            "recordType": "qa",
            "action": "review",
            "prevFeedbackId": "f1",
            "editCount": 1,
            "trainingStatus": "eligible",
            "ratingValue": 1,
        }
    )
    clean = dd.deduplicate([first, second])
    assert [r["feedbackId"] for r in clean] == ["f2"]


def test_server_lineage_validator_accepts_current_chain_and_rejects_cycle() -> None:
    valid = {
        "feedbackId": "f3",
        "feedbackChainId": "f1",
        "prevFeedbackId": "f2",
        "prevFeedbackIds": ["f1", "f2"],
        "editCount": 2,
    }
    proxy_app._validate_feedback_lineage_fields(valid, label="x")

    bad = dict(valid)
    bad["prevFeedbackIds"] = ["f1", "f3"]
    bad["prevFeedbackId"] = "f3"
    with pytest.raises(HTTPException) as exc:
        proxy_app._validate_feedback_lineage_fields(bad, label="x")
    assert exc.value.status_code == 422
