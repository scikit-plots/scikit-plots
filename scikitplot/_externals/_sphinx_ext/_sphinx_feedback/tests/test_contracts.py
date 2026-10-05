from __future__ import annotations

import json

import pytest

from _sphinx_ext._sphinx_feedback._contracts import (
    EVENT_CONTRACT,
    FeedbackValidationError,
    build_feedback_event,
    canonical_event_bytes,
    canonical_feedback_bytes,
    decode_feedback_event,
    decode_feedback_request,
    feedback_event_request_hash,
    feedback_request_from_event,
    feedback_request_hash,
    page_digest,
    parse_feedback_event,
    parse_feedback_request,
    repository_event_bytes,
)


def request(**overrides):
    payload = {
        "contract": "page.feedback-request.v1",
        "action": "submit",
        "site_id": "docs-site",
        "page_id": "guide/install",
        "feedback_id": "feedback-" + "a" * 48,
        "rating": 1,
        "mode": "quick",
        "contributor": {"display_name": ""},
    }
    payload.update(overrides)
    return payload


def test_quick_request_normalizes():
    assert parse_feedback_request(request())["rating"] == 1


def test_detailed_request_normalizes_credit_comment_and_revision():
    got = parse_feedback_request(
        request(
            mode="detailed",
            rating=-3,
            comment="  line one\nline two  ",
            contributor={"display_name": "  Ada   Lovelace "},
            page_revision="abc123",
        )
    )
    assert got["comment"] == "line one\nline two"
    assert got["contributor"]["display_name"] == "Ada Lovelace"
    assert got["page_revision"] == "abc123"


@pytest.mark.parametrize("rating", range(-5, 6))
def test_detailed_accepts_full_rating_scale(rating):
    got = parse_feedback_request(request(mode="detailed", rating=rating))
    assert got["rating"] == rating


@pytest.mark.parametrize("rating", [-5, -2, 0, 2, 5])
def test_quick_rejects_non_binary_rating(rating):
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(rating=rating))


def test_bool_rating_rejected():
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(rating=True))


@pytest.mark.parametrize(
    "extra",
    ["ip", "user_agent", "referrer", "timezone", "locale", "session_id", "screen"],
)
def test_implicit_telemetry_fields_fail_closed(extra):
    payload = request()
    payload[extra] = "should-not-persist"
    with pytest.raises(FeedbackValidationError, match="unsupported field"):
        parse_feedback_request(payload)


def test_nested_contributor_unknown_field_rejected():
    with pytest.raises(FeedbackValidationError, match="unsupported field"):
        parse_feedback_request(request(contributor={"display_name": "Ada", "email": "x@y"}))


@pytest.mark.parametrize(
    "page_id",
    ["/guide", "guide/", "guide//x", "guide/../x", "guide/./x", "guide\\x", "guide?x=1", "guide#x", "guide\x00x"],
)
def test_page_id_rejects_url_and_traversal_forms(page_id):
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(page_id=page_id))


@pytest.mark.parametrize("site_id", ["", "has space", "bad/slash", "_leading", "trailing-"])
def test_site_id_is_stable_ascii(site_id):
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(site_id=site_id))


@pytest.mark.parametrize(
    "feedback_id",
    ["feedback-short", "feedback-" + "A" * 48, "feedback-" + "a" * 47, "event-" + "a" * 48],
)
def test_feedback_nonce_shape_is_strict(feedback_id):
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(feedback_id=feedback_id))


def test_public_credit_rejects_control_characters():
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(
            request(mode="detailed", contributor={"display_name": "Ada\nLovelace"})
        )


@pytest.mark.parametrize("credit", ["Anonymous", "anonymous", " ANONYMOUS "])
def test_public_credit_reserves_anonymous_system_sentinel(credit):
    with pytest.raises(FeedbackValidationError, match="reserved"):
        parse_feedback_request(
            request(mode="detailed", contributor={"display_name": credit})
        )


def test_wire_decoder_maps_pathological_integer_to_contract_error():
    huge = "9" * 5000
    raw = (
        '{"contract":"page.feedback-request.v1","action":"submit",'
        '"site_id":"docs-site","page_id":"guide/install",'
        '"feedback_id":"feedback-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        f'"rating":{huge},"mode":"quick","contributor":{{"display_name":""}}}}'
    ).encode("ascii")
    with pytest.raises(FeedbackValidationError, match="Invalid JSON"):
        decode_feedback_request(raw)


def test_comment_allows_deliberate_newlines_tabs():
    got = parse_feedback_request(request(mode="detailed", comment="a\tb\nc\r\nd"))
    assert got["comment"] == "a\tb\nc\r\nd"


def test_comment_rejects_other_control_characters():
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(mode="detailed", comment="bad\x0bcomment"))


def test_quick_cannot_smuggle_comment_or_credit():
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(comment="x"))
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(contributor={"display_name": "Ada"}))


def test_hash_is_key_order_independent_and_64_hex():
    a = request()
    b = dict(reversed(list(a.items())))
    assert feedback_request_hash(a) == feedback_request_hash(b)
    assert len(feedback_request_hash(a)) == 64


def test_canonical_bytes_round_trip_to_normalized_request():
    payload = request(mode="detailed", rating=0, contributor=None)
    decoded = json.loads(canonical_feedback_bytes(payload))
    assert decoded == parse_feedback_request(payload)


def test_event_projection_contains_no_implicit_identity_fields():
    event = build_feedback_event(request())
    assert event == {
        "contract": EVENT_CONTRACT,
        "site_id": "docs-site",
        "page_id": "guide/install",
        "feedback": {
            "id": "feedback-" + "a" * 48,
            "rating": 1,
            "mode": "quick",
            "contributor": "Anonymous",
        },
    }
    text = json.dumps(event)
    for forbidden in ["ip", "user_agent", "referrer", "timezone", "session", "browser"]:
        assert forbidden not in text


def test_page_digest_is_deterministic_site_scoped_and_path_safe():
    one = page_digest("docs-a", "guide/install")
    assert one == page_digest("docs-a", "guide/install")
    assert one != page_digest("docs-b", "guide/install")
    assert len(one) == 24
    assert all(ch in "0123456789abcdef" for ch in one)


def test_wrong_contract_and_action_rejected():
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(contract="other.v1"))
    with pytest.raises(FeedbackValidationError):
        parse_feedback_request(request(action="delete"))


def test_unicode_is_nfc_canonical_in_request_and_event():
    decomposed = "Cafe\u0301"
    req = request(
        mode="detailed",
        rating=2,
        comment=f"  {decomposed}  ",
        contributor={"display_name": f" {decomposed} "},
    )
    normalized = parse_feedback_request(req)
    assert normalized["comment"] == "Café"
    assert normalized["contributor"]["display_name"] == "Café"
    event = build_feedback_event(req)
    assert parse_feedback_event(event) == event
    assert canonical_event_bytes(event).endswith(b"\n")


def test_repository_event_bytes_are_pretty_without_changing_canonical_identity():
    item = build_feedback_event(
        request(
            mode="detailed",
            rating=2,
            comment="Readable JSON",
            contributor={"display_name": "Ada"},
        )
    )
    compact = canonical_event_bytes(item)
    pretty = repository_event_bytes(item)
    assert pretty.endswith(b"\n")
    assert pretty != compact
    assert b"\n  \"feedback\":" in pretty
    assert json.loads(pretty) == json.loads(compact) == item
    assert decode_feedback_event(pretty) == item
    assert decode_feedback_event(compact) == item


def test_durable_event_decoder_rejects_duplicate_keys_and_nonfinite_json():
    duplicate = (
        b'{"contract":"page.feedback-event.v1","site_id":"docs-site",'
        b'"page_id":"guide/install","feedback":{"id":'
        b'"feedback-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"rating":1,"rating":-1,"mode":"quick","contributor":"Anonymous"}}'
    )
    with pytest.raises(FeedbackValidationError, match="duplicate JSON field"):
        decode_feedback_event(duplicate)

    nonfinite = (
        b'{"contract":"page.feedback-event.v1","site_id":"docs-site",'
        b'"page_id":"guide/install","feedback":{"id":'
        b'"feedback-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"rating":NaN,"mode":"quick","contributor":"Anonymous"}}'
    )
    with pytest.raises(FeedbackValidationError, match="non-finite JSON"):
        decode_feedback_event(nonfinite)


def test_event_contract_rejects_unknown_or_noncanonical_fields():
    event = build_feedback_event(
        request(mode="detailed", rating=2, comment="hello", contributor={"display_name": "Ada"})
    )
    bad = json.loads(json.dumps(event))
    bad["feedback"]["email"] = "private@example.org"
    with pytest.raises(FeedbackValidationError, match="unsupported field"):
        parse_feedback_event(bad)
    bad = json.loads(json.dumps(event))
    bad["feedback"]["contributor"] = "  Ada  "
    with pytest.raises(FeedbackValidationError, match="canonicalized"):
        parse_feedback_event(bad)


def test_2000_non_ascii_codepoints_fit_canonical_request_limit():
    payload = request(mode="detailed", rating=5, comment="🤩" * 2000)
    assert len(canonical_feedback_bytes(payload)) <= 16 * 1024


def test_wire_decoder_rejects_duplicate_keys_and_nonfinite_json():
    duplicate = (
        b'{"contract":"page.feedback-request.v1","action":"submit",'
        b'"site_id":"docs-site","page_id":"guide/install",'
        b'"feedback_id":"feedback-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"rating":1,"rating":-1,"mode":"quick","contributor":{"display_name":""}}'
    )
    with pytest.raises(FeedbackValidationError, match="duplicate JSON field"):
        decode_feedback_request(duplicate)

    nonfinite = (
        b'{"contract":"page.feedback-request.v1","action":"submit",'
        b'"site_id":"docs-site","page_id":"guide/install",'
        b'"feedback_id":"feedback-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"rating":NaN,"mode":"quick","contributor":{"display_name":""}}'
    )
    with pytest.raises(FeedbackValidationError, match="non-finite JSON"):
        decode_feedback_request(nonfinite)


@pytest.mark.parametrize("credit", ["anonymous", "ANONYMOUS"])
def test_durable_event_reserves_case_variants_of_anonymous_sentinel(credit):
    event = build_feedback_event(
        request(mode="detailed", contributor={"display_name": "Ada"})
    )
    event["feedback"]["contributor"] = credit
    with pytest.raises(FeedbackValidationError, match="reserves 'Anonymous'"):
        parse_feedback_event(event)


def test_event_inverse_projection_preserves_request_commitment():
    for req in [
        request(),
        request(
            mode="detailed",
            rating=-4,
            comment="  Café\nexplanation  ",
            contributor={"display_name": " Ada   Lovelace "},
            page_revision="rev-42",
        ),
    ]:
        normalized = parse_feedback_request(req)
        event = build_feedback_event(req)
        assert feedback_request_from_event(event) == normalized
        assert feedback_event_request_hash(event) == feedback_request_hash(normalized)
