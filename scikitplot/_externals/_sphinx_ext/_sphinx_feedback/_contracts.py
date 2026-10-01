"""Strict, privacy-minimal contracts for generic page feedback."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

REQUEST_CONTRACT = "page.feedback-request.v1"
EVENT_CONTRACT = "page.feedback-event.v1"
RECEIPT_CONTRACT = "page.feedback-receipt.v1"
AGGREGATE_CONTRACT = "page.feedback-aggregate.v3"

# A 2,000-code-point Unicode comment can exceed 8 KiB once encoded. Keep the
# canonical envelope and transport limits aligned so the documented comment
# limit remains usable for non-ASCII text as well as ASCII prose.
MAX_REQUEST_BYTES = 16 * 1024
MAX_AGGREGATE_BYTES = 2 * 1024 * 1024
MAX_SITE_ID = 96
MAX_PAGE_ID = 512
MAX_PAGE_REVISION = 128
MAX_COMMENT = 2000
MAX_CONTRIBUTOR = 80

_SITE_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?\Z")
_FEEDBACK_ID_RE = re.compile(r"feedback-[0-9a-f]{48}\Z")
_REQUEST_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_TOP = frozenset(
    {
        "contract",
        "action",
        "site_id",
        "page_id",
        "page_revision",
        "feedback_id",
        "rating",
        "mode",
        "comment",
        "contributor",
    }
)
_ALLOWED_CONTRIBUTOR = frozenset({"display_name"})
_ALLOWED_EVENT_TOP = frozenset(
    {
        "contract",
        "site_id",
        "page_id",
        "page_revision",
        "feedback",
    },
)
_ALLOWED_EVENT_FEEDBACK = frozenset(
    {
        "id",
        "rating",
        "mode",
        "comment",
        "contributor",
    },
)


class FeedbackValidationError(ValueError):
    """Raised when a public feedback envelope violates the contract."""

    def __init__(self, message: str, *, code: str = "invalid_feedback") -> None:
        super().__init__(message)
        self.code = code


class FeedbackConflictError(RuntimeError):
    """Raised when an idempotency key is replayed with different content."""


def _reject_unknown_keys(
    value: dict[str, Any],
    allowed: frozenset[str],
    path: str,
) -> None:
    extras = sorted(str(key) for key in value if key not in allowed)
    if extras:
        raise FeedbackValidationError(
            f"{path}: unsupported field(s): {', '.join(extras)}",
            code="unexpected_field",
        )


def _nfc_text(value: Any, *, path: str, limit: int, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise FeedbackValidationError(f"{path}: must be a string")
    try:
        text = unicodedata.normalize("NFC", value)
        text.encode("utf-8", "strict")
    except (UnicodeError, ValueError) as exc:
        raise FeedbackValidationError(f"{path}: invalid Unicode text") from exc
    if len(text) > limit:
        raise FeedbackValidationError(f"{path}: exceeds {limit} characters")
    if not allow_empty and not text:
        raise FeedbackValidationError(f"{path}: must not be empty")
    return text


def _public_credit(value: Any) -> str:
    text = _nfc_text(
        value,
        path="contributor.display_name",
        limit=MAX_CONTRIBUTOR,
        allow_empty=True,
    )
    if any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in text
    ):
        raise FeedbackValidationError(
            "contributor.display_name: control characters are not allowed"
        )
    text = re.sub(r" +", " ", text.strip(" "))
    if len(text) > MAX_CONTRIBUTOR:
        raise FeedbackValidationError(
            f"contributor.display_name: exceeds {MAX_CONTRIBUTOR} characters"
        )
    return text


def _comment(value: Any) -> str:
    text = _nfc_text(
        value,
        path="comment",
        limit=MAX_COMMENT,
        allow_empty=True,
    ).strip(" \t\n\r")
    # Preserve line breaks and tabs in deliberate prose, but reject the rest of
    # C0/DEL. This mirrors the AI Learn reviewed-comment boundary.
    for ch in text:
        code = ord(ch)
        if code == 127 or (  # ruff: ignore[magic-value-comparison]
            code < 32 and ch not in "\t\n\r"  # ruff: ignore[magic-value-comparison]
        ):
            raise FeedbackValidationError("comment: unsupported control characters")
    return text


def normalize_page_id(value: Any) -> str:
    """Validate a canonical page identity without interpreting it as a URL/path."""
    text = _nfc_text(
        value, path="page_id", limit=MAX_PAGE_ID, allow_empty=False
    ).strip()
    if (
        text.startswith("/")
        or text.endswith("/")
        or "\\" in text
        or "?" in text
        or "#" in text
        or any(
            ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in text
        )
    ):
        raise FeedbackValidationError("page_id: invalid canonical page name")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise FeedbackValidationError("page_id: invalid path segment")
    return text


def normalize_page_revision(value: Any) -> str:
    if value is None or value == "":
        return ""
    text = _nfc_text(
        value,
        path="page_revision",
        limit=MAX_PAGE_REVISION,
        allow_empty=True,
    ).strip()
    if any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in text
    ):
        raise FeedbackValidationError(
            "page_revision: control characters are not allowed",
        )
    return text


def _site_id(value: Any) -> str:
    site_id = _nfc_text(
        value, path="site_id", limit=MAX_SITE_ID, allow_empty=False
    ).strip()
    if not _SITE_ID_RE.fullmatch(site_id):
        raise FeedbackValidationError("site_id: use a stable ASCII identifier")
    return site_id


def normalize_site_id(value: Any) -> str:
    """Validate and canonicalize a stable public site identifier."""
    return _site_id(value)


def _feedback_id(value: Any) -> str:
    if not isinstance(value, str) or not _FEEDBACK_ID_RE.fullmatch(value):
        raise FeedbackValidationError(
            "feedback_id: expected feedback- followed by 48 lowercase hexadecimal characters"
        )
    return value


def _rating(value: Any) -> int:
    _value = -5 <= value <= 5  # ruff: ignore[magic-value-comparison]
    if isinstance(value, bool) or not isinstance(value, int) or not _value:
        raise FeedbackValidationError("rating: expected an integer from -5 through +5")
    return value


def _mode(value: Any) -> str:
    if value not in {"quick", "detailed"}:
        raise FeedbackValidationError(
            "mode: expected 'quick' or 'detailed'",
        )
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FeedbackValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise FeedbackValidationError(f"non-finite JSON number is not allowed: {value}")


def decode_feedback_request(raw: bytes | str) -> dict[str, Any]:
    """
    Decode one wire request with strict JSON semantics, then validate it.

    Duplicate object keys and non-finite numeric spellings are rejected before
    contract normalization so different JSON parsers cannot disagree about the
    event that an idempotency key represents.
    """
    if not isinstance(raw, (bytes, str)):
        raise FeedbackValidationError("Feedback body must be JSON text")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except FeedbackValidationError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise FeedbackValidationError(
            "Invalid JSON",
            code="invalid_json",
        ) from exc
    return parse_feedback_request(payload)


def decode_feedback_event(raw: bytes | str) -> dict[str, Any]:
    """Decode one durable event with the same strict JSON semantics as requests."""
    if not isinstance(raw, (bytes, str)):
        raise FeedbackValidationError("Feedback event must be JSON text")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except FeedbackValidationError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise FeedbackValidationError(
            "Invalid feedback event JSON",
            code="invalid_event_json",
        ) from exc
    return parse_feedback_event(payload)


def parse_feedback_request(payload: Any) -> dict[str, Any]:
    """
    Validate and normalize one browser feedback request.

    Unknown fields are rejected deliberately. This prevents future browser,
    device, account, referrer, IP, or analytics fields from silently becoming
    durable feedback data.
    """
    if not isinstance(payload, dict):
        raise FeedbackValidationError("Feedback body must be a JSON object")
    _reject_unknown_keys(payload, _ALLOWED_TOP, "feedback")

    if payload.get("contract") != REQUEST_CONTRACT:
        raise FeedbackValidationError(
            "Unsupported feedback contract", code="unsupported_contract"
        )
    if payload.get("action") != "submit":
        raise FeedbackValidationError("Unsupported feedback action")

    site_id = normalize_site_id(payload.get("site_id"))
    page_id = normalize_page_id(payload.get("page_id"))
    page_revision = normalize_page_revision(payload.get("page_revision", ""))
    feedback_id = _feedback_id(payload.get("feedback_id"))
    rating = _rating(payload.get("rating"))
    mode = _mode(payload.get("mode"))

    raw_contributor = payload.get("contributor", {})
    if raw_contributor is None:
        raw_contributor = {}
    if not isinstance(raw_contributor, dict):
        raise FeedbackValidationError("contributor: must be an object")
    _reject_unknown_keys(raw_contributor, _ALLOWED_CONTRIBUTOR, "contributor")
    contributor = _public_credit(raw_contributor.get("display_name", ""))
    if contributor.lower() == "anonymous":
        raise FeedbackValidationError(
            "contributor.display_name: 'Anonymous' is reserved; leave public credit blank instead"
        )
    comment = _comment(payload.get("comment", ""))

    if mode == "quick":
        if rating not in {-1, 1}:
            raise FeedbackValidationError("quick feedback rating must be -1 or +1")
        if comment or contributor:
            raise FeedbackValidationError(
                "quick feedback must not include comment or contributor content"
            )

    normalized: dict[str, Any] = {
        "contract": REQUEST_CONTRACT,
        "action": "submit",
        "site_id": site_id,
        "page_id": page_id,
        "feedback_id": feedback_id,
        "rating": rating,
        "mode": mode,
        "contributor": {"display_name": contributor},
    }
    if page_revision:
        normalized["page_revision"] = page_revision
    if comment:
        normalized["comment"] = comment
    return normalized


def canonical_feedback_bytes(request: dict[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON used for request commitments/idempotency."""
    normalized = parse_feedback_request(request)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise FeedbackValidationError(
            f"feedback request exceeds {MAX_REQUEST_BYTES} canonical bytes"
        )
    return encoded


def feedback_request_hash(request: dict[str, Any]) -> str:
    """Return a deterministic commitment to the normalized feedback request."""
    return hashlib.sha256(canonical_feedback_bytes(request)).hexdigest()


def build_feedback_event(request: dict[str, Any]) -> dict[str, Any]:
    """Project a request into the minimal durable feedback event."""
    normalized = parse_feedback_request(request)
    feedback: dict[str, Any] = {
        "id": normalized["feedback_id"],
        "rating": normalized["rating"],
        "mode": normalized["mode"],
        "contributor": normalized["contributor"]["display_name"] or "Anonymous",
    }
    if normalized.get("comment"):
        feedback["comment"] = normalized["comment"]
    event: dict[str, Any] = {
        "contract": EVENT_CONTRACT,
        "site_id": normalized["site_id"],
        "page_id": normalized["page_id"],
        "feedback": feedback,
    }
    if normalized.get("page_revision"):
        event["page_revision"] = normalized["page_revision"]
    return event


def feedback_request_from_event(event: dict[str, Any]) -> dict[str, Any]:
    """
    Reconstruct the unique normalized request represented by a durable event.

    This inverse projection is intentionally strict. It gives storage/provider
    adapters an independent way to verify that a supplied request commitment
    actually commits to the exact durable event they are about to accept.
    """
    normalized = parse_feedback_event(event)
    feedback = normalized["feedback"]
    contributor = feedback["contributor"]
    request: dict[str, Any] = {
        "contract": REQUEST_CONTRACT,
        "action": "submit",
        "site_id": normalized["site_id"],
        "page_id": normalized["page_id"],
        "feedback_id": feedback["id"],
        "rating": feedback["rating"],
        "mode": feedback["mode"],
        "contributor": {
            "display_name": "" if contributor == "Anonymous" else contributor
        },
    }
    if normalized.get("page_revision"):
        request["page_revision"] = normalized["page_revision"]
    if feedback.get("comment"):
        request["comment"] = feedback["comment"]
    return parse_feedback_request(request)


def feedback_event_request_hash(event: dict[str, Any]) -> str:
    """Return the request commitment implied by one canonical durable event."""
    return feedback_request_hash(feedback_request_from_event(event))


def parse_feedback_event(  # ruff: ignore[too-many-branches]
    payload: Any,
) -> dict[str, Any]:
    """Validate one durable event and require canonical public text forms."""
    if not isinstance(payload, dict):
        raise FeedbackValidationError("feedback event must be an object")
    _reject_unknown_keys(payload, _ALLOWED_EVENT_TOP, "feedback event")
    if payload.get("contract") != EVENT_CONTRACT:
        raise FeedbackValidationError("Unsupported feedback event contract")

    site_id = normalize_site_id(payload.get("site_id"))
    page_id = normalize_page_id(payload.get("page_id"))
    page_revision = normalize_page_revision(payload.get("page_revision", ""))
    raw_feedback = payload.get("feedback")
    if not isinstance(raw_feedback, dict):
        raise FeedbackValidationError("feedback event feedback must be an object")
    _reject_unknown_keys(
        raw_feedback,
        _ALLOWED_EVENT_FEEDBACK,
        "feedback event.feedback",
    )
    feedback_id = _feedback_id(raw_feedback.get("id"))
    rating = _rating(raw_feedback.get("rating"))
    mode = _mode(raw_feedback.get("mode"))

    raw_contributor = raw_feedback.get("contributor")
    if raw_contributor == "Anonymous":
        contributor = "Anonymous"
    else:
        contributor = _public_credit(raw_contributor)
        if contributor.lower() == "anonymous":
            raise FeedbackValidationError(
                "feedback event contributor reserves 'Anonymous' for the anonymity sentinel"
            )
        if not contributor:
            raise FeedbackValidationError(
                "feedback event contributor must be 'Anonymous' or a non-empty public credit"
            )
        if contributor != raw_contributor:
            raise FeedbackValidationError(
                "feedback event contributor must already be canonicalized"
            )

    comment = _comment(raw_feedback.get("comment", ""))
    if "comment" in raw_feedback and comment != raw_feedback.get("comment"):
        raise FeedbackValidationError(
            "feedback event comment must already be canonicalized",
        )
    if mode == "quick":
        if rating not in {-1, 1}:
            raise FeedbackValidationError(
                "quick feedback rating must be -1 or +1",
            )
        if comment or contributor != "Anonymous":
            raise FeedbackValidationError(
                "quick feedback event must not contain comment or public credit",
            )

    normalized_feedback: dict[str, Any] = {
        "id": feedback_id,
        "rating": rating,
        "mode": mode,
        "contributor": contributor,
    }
    if comment:
        normalized_feedback["comment"] = comment
    normalized: dict[str, Any] = {
        "contract": EVENT_CONTRACT,
        "site_id": site_id,
        "page_id": page_id,
        "feedback": normalized_feedback,
    }
    if page_revision:
        normalized["page_revision"] = page_revision
    return normalized


def canonical_event_bytes(event: dict[str, Any]) -> bytes:
    """Return deterministic bytes for durable event equality checks."""
    normalized = parse_feedback_event(event)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def repository_event_bytes(event: dict[str, Any]) -> bytes:
    """
    Return deterministic human-readable JSON for repository event files.

    Repository presentation is intentionally separate from
    :func:`canonical_event_bytes`: whitespace must never become feedback-event
    identity or alter request/idempotency commitments.
    """
    normalized = parse_feedback_event(event)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def validate_request_hash(value: Any) -> str:
    """Validate a lowercase SHA-256 request commitment."""
    if not isinstance(value, str) or not _REQUEST_HASH_RE.fullmatch(value):
        raise FeedbackValidationError(
            "request_hash must be 64 lowercase hexadecimal characters",
        )
    return value


def page_digest(site_id: str, page_id: str) -> str:
    """Return a filesystem-safe opaque page bucket without hiding page_id in JSON."""
    site = normalize_site_id(site_id)
    page = normalize_page_id(page_id)
    raw = f"{site}\0{page}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]
